import math
from argparse import ArgumentParser

import torch
from pytorch_lightning import LightningModule, Trainer
#from pytorch_lightning.plugins.environments import LightningEnvironment
#from pytorch_lightning.plugins.training_type import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch import nn, Tensor
from torch.nn import functional as F

from pl_bolts.models.self_supervised.resnets import resnet18, resnet50
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    imagenet_normalization,
    stl10_normalization,
)

import distributed as du
import multiprocessingu as mpu
from typing import Union, Any, Dict
import os
import numpy as np

from pytorch_lightning.plugins.environments import ClusterEnvironment


class MyClusterEnvironment(ClusterEnvironment):
    def creates_children(self) -> bool:
        # return True if the cluster is managed (you don't launch processes yourself)
        return True
    def world_size(self) -> int:
        return int(os.environ["WORLD_SIZE"])
    def global_rank(self) -> int:
        return int(os.environ["RANK"])
    def local_rank(self) -> int:
        return int(os.environ["LOCAL_RANK"])
    def node_rank(self) -> int:
        return int(os.environ["NODE_RANK"])
    def master_address(self) -> str:
        master_node_params = os.environ['AZ_BATCHAI_WORKER_HOSTS'].split(':')
        return master_node_params[0]
        #return 'localhost'#get_ip_address()
    def master_port(self) -> int:
        return int(os.environ["MASTER_PORT"])
    def set_world_size(self, size: int) -> None:
        print("SLURMEnvironment.set_world_size was called, but setting world size is not allowed. Ignored.")
    def set_global_rank(self, rank: int) -> None:
        print("SLURMEnvironment.set_global_rank was called, but setting global rank is not allowed. Ignored.")

class SyncFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


class Projection(nn.Module):

    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim), nn.BatchNorm1d(self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


class SimCLR(LightningModule):

    def __init__(
        self,
        gpus: int,
        num_samples: int,
        batch_size: int,
        dataset: str,
        lam: float = 0.001,
        q: float =0.5,
        num_nodes: int = 1,
        arch: str = 'resnet50',
        hidden_mlp: int = 2048,
        feat_dim: int = 128,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        temperature: float = 0.1,
        first_conv: bool = True,
        maxpool1: bool = True,
        optimizer: str = 'adam',
        exclude_bn_bias: bool = False,
        start_lr: float = 0.,
        learning_rate: float = 1e-3,
        final_lr: float = 0.,
        weight_decay: float = 1e-6,
        **kwargs
    ):
        """
        Args:
            batch_size: the batch size
            num_samples: num samples in the dataset
            warmup_epochs: epochs to warmup the lr for
            lr: the optimizer learning rate
            opt_weight_decay: the optimizer weight decay
            loss_temperature: the loss temperature
        """
        super().__init__()
        self.save_hyperparameters()

        self.num_pos = 2
        self.lam = lam
        self.q = q

        self.gpus = gpus
        self.num_nodes = num_nodes
        self.arch = arch
        self.dataset = dataset
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        self.optim = optimizer
        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay
        self.temperature = temperature

        self.start_lr = start_lr
        self.final_lr = final_lr
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.encoder = self.init_model()

        self.projection = Projection(input_dim=self.hidden_mlp, hidden_dim=self.hidden_mlp, output_dim=self.feat_dim)

        # compute iters per epoch
        global_batch_size = self.num_nodes * self.gpus * self.batch_size if self.gpus > 0 else self.batch_size
        self.train_iters_per_epoch = self.num_samples // global_batch_size

    def init_model(self):
        if self.arch == 'resnet18':
            backbone = resnet18
        elif self.arch == 'resnet50':
            backbone = resnet50

        return backbone(first_conv=self.first_conv, maxpool1=self.maxpool1, return_all_feature_maps=False)

    def forward(self, x):
        # bolts resnet returns a list
        return self.encoder(x)[-1]

    def shared_step(self, batch):
        if self.dataset == 'stl10':
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        # final image in tuple is for online eval
        (img1, img2, _), y = batch

        # get h representations, bolts resnet returns a list
        h1 = self(img1)
        h2 = self(img2)

        # get z representations
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss = self.sym_loss(z1, z2, self.lam, self.q, self.temperature)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=['bias', 'bn']):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [{
            'params': params,
            'weight_decay': weight_decay
        }, {
            'params': excluded_params,
            'weight_decay': 0.,
        }]

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.weight_decay)
        else:
            params = self.parameters()

        if self.optim == 'lars':
            optimizer = LARS(
                params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.optim == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def compute_neg_mask(self):
        """
        We precompute the positive and negative masks to speed up the loss calculation
        Code modified from facebookresearch/vissl
        """
        # computed once at the begining of training
        
        total_images = self.num_nodes * self.gpus * self.batch_size * self.num_pos
        world_size = self.num_nodes * self.gpus
        batch_size = self.batch_size * self.num_pos
        orig_images = self.batch_size
        rank = int(os.environ["LOCAL_RANK"])

        neg_mask = torch.zeros(batch_size, total_images)
        all_indices = np.arange(total_images)
        pos_members = orig_images * world_size * np.arange(self.num_pos)
        for anchor in np.arange(self.num_pos):
            for img_idx in range(orig_images):
                delete_inds = orig_images * rank + img_idx + pos_members
                neg_inds = torch.tensor(np.delete(all_indices, delete_inds)).long()
                neg_mask[anchor * orig_images + img_idx, neg_inds] = 1
        neg_mask = neg_mask.cuda(non_blocking=True)

        return neg_mask

    def sym_loss(self, out_1, out_2, lam, q, temperature):
        """
            assume out_1 and out_2 are normalized
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = SyncFunction.apply(out_1)
            out_2_dist = SyncFunction.apply(out_2)
        else:
            out_1_dist = out_1
            out_2_dist = out_2


        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        similarity = torch.exp(torch.mm(out, out_dist.t()) / temperature)
        neg_mask = self.compute_neg_mask()
        neg = torch.sum(similarity * neg_mask, 1)

        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        # InfoNCE loss
        # loss = -(torch.mean(torch.log(pos / (pos + neg))))

        # RINCE loss
        neg = ((lam*(pos + neg))**q) / q
        pos = -(pos**q) / q
        loss = pos.mean() + neg.mean()

        return loss


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # model params
        parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
        # specify flags to store false
        parser.add_argument("--first_conv", action='store_false')
        parser.add_argument("--maxpool1", action='store_false')
        parser.add_argument("--hidden_mlp", default=2048, type=int, help="hidden layer dimension in projection head")
        parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")
        parser.add_argument("--online_ft", action='store_true')
        parser.add_argument("--fp32", action='store_true')

        # transform params
        parser.add_argument("--gaussian_blur", action="store_true", help="add gaussian blur")
        parser.add_argument("--jitter_strength", type=float, default=1.0, help="jitter strength")
        parser.add_argument("--dataset", type=str, default="cifar10", help="stl10, cifar10")
        parser.add_argument("--data_dir", type=str, default=".", help="path to download data")

        # training params
        parser.add_argument("--fast_dev_run", default=1, type=int)
        parser.add_argument("--num_nodes", default=1, type=int, help="number of nodes for training")
        parser.add_argument("--gpus", default=1, type=int, help="number of gpus to train on")
        parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
        parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/lars")
        parser.add_argument('--exclude_bn_bias', action='store_true', help="exclude bn/bias from weight decay")
        parser.add_argument("--max_epochs", default=100, type=int, help="number of total epochs to run")
        parser.add_argument("--max_steps", default=-1, type=int, help="max steps")
        parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
        parser.add_argument("--batch_size", default=128, type=int, help="batch size per gpu")

        parser.add_argument("--temperature", default=0.1, type=float, help="temperature parameter in training loss")
        parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
        parser.add_argument("--learning_rate", default=1e-3, type=float, help="base learning rate")
        parser.add_argument("--start_lr", default=0, type=float, help="initial warmup learning rate")
        parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")

        parser.add_argument("--save_path", type=str, help="path to save checkpoints")
        parser.add_argument("--lam", default=0.01, type=float, help="lambda for RINCE loss")
        parser.add_argument("--q", default=0.1, type=float, help="q for RINCE loss")

        return parser


def cli_main():
    from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
    from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, STL10DataModule
    from pl_bolts.models.self_supervised.simclr.transforms import SimCLREvalDataTransform, SimCLRTrainDataTransform

    parser = ArgumentParser()

    # model args
    parser = SimCLR.add_model_specific_args(parser)
    args = parser.parse_args()

    if args.dataset == 'stl10':
        dm = STL10DataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True, shuffle=True)

        dm.train_dataloader = dm.train_dataloader_mixed
        dm.val_dataloader = dm.val_dataloader_mixed
        args.num_samples = dm.num_unlabeled_samples

        args.maxpool1 = False
        args.first_conv = True
        args.input_height = dm.size()[-1]

        normalization = stl10_normalization()

        args.gaussian_blur = True
        args.jitter_strength = 1.
    elif args.dataset == 'cifar10':
        val_split = 5000
        if args.num_nodes * args.gpus * args.batch_size > val_split:
            val_split = args.num_nodes * args.gpus * args.batch_size

        dm = CIFAR10DataModule(
            data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, val_split=val_split, drop_last=True, shuffle=True
        )

        args.num_samples = dm.num_samples

        args.maxpool1 = False
        args.first_conv = False
        args.input_height = dm.size()[-1]
        args.temperature = 0.5

        normalization = cifar10_normalization()

        args.gaussian_blur = False
        args.jitter_strength = 0.5
    elif args.dataset == 'imagenet':
        args.maxpool1 = True
        args.first_conv = True
        normalization = imagenet_normalization()

        args.gaussian_blur = True
        args.jitter_strength = 1.

        args.num_nodes = 1
        args.gpus = 16  # per-node
        args.batch_size = int(4096 / (args.num_nodes * args.gpus))
        args.max_epochs = 800

        args.optimizer = 'lars'
        args.learning_rate = 4.8
        args.final_lr = 0.0048
        args.start_lr = 0.3
        args.online_ft = True

        dm = ImagenetDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True, shuffle=True)

        args.num_samples = dm.num_samples
        args.input_height = dm.size()[-1]
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    dm.train_transforms = SimCLRTrainDataTransform(
        input_height=args.input_height,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
    )

    dm.val_transforms = SimCLREvalDataTransform(
        input_height=args.input_height,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
    )

    model = SimCLR(**args.__dict__)

    online_evaluator = None
    if args.online_ft:
        # online eval
        online_evaluator = SSLOnlineEvaluator(
            drop_p=0.,
            hidden_dim=None,
            z_dim=args.hidden_mlp,
            num_classes=dm.num_classes,
            dataset=args.dataset,
        )

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(dirpath=args.save_path, save_last=True, save_top_k=100, monitor='val_loss', every_n_val_epochs=10)
    callbacks = [model_checkpoint, online_evaluator] if args.online_ft else [model_checkpoint]
    callbacks.append(lr_monitor)

    if args.num_nodes > 1:
        du.set_environment_variables_for_nccl_backend(
            du.get_global_size() == du.get_local_size(args.gpus),
            6105, # MASTER_PORT
            True#int(args.num_nodes) > 1,
        )

        trainer = Trainer(
            max_epochs=args.max_epochs,
            max_steps=None if args.max_steps == -1 else args.max_steps,
            gpus=args.gpus,
            num_nodes=args.num_nodes,
            distributed_backend='ddp' if args.gpus > 1 else None,
            sync_batchnorm=True if args.gpus > 1 else False,
            precision=32 if args.fp32 else 16,
            callbacks=callbacks,
            plugins=[MyClusterEnvironment()],
        )
    else:
        du.set_environment_variables_for_nccl_backend(
            du.get_global_size() == du.get_local_size(args.gpus),
            6105, # MASTER_PORT
            False#int(args.num_nodes) > 1,
        )

        trainer = Trainer(
            max_epochs=args.max_epochs,
            max_steps=None if args.max_steps == -1 else args.max_steps,
            gpus=args.gpus,
            num_nodes=args.num_nodes,
            distributed_backend='ddp' if args.gpus > 1 else None,
            sync_batchnorm=True if args.gpus > 1 else False,
            precision=32 if args.fp32 else 16,
            callbacks=callbacks,
        )

    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    cli_main()
