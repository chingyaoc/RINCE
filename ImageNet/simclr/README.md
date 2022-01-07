## SimCLR, RINCE version

### Introduction
This is a PyTorch implementation of RINCE with SimCLR. The code is modified from the implementation of [PyTorchLightning/lightning-bolts](https://github.com/PyTorchLightning/lightning-bolts/tree/master/pl_bolts/models/self_supervised/simclr). Please see their [documentation](https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html) for detailed instruction.


### ImageNet Experiments

Launch the ImageNet training with default hyperparameters:
```
python symclr_module.py
    --dataset imagenet
    --lam 0.01
    --q 0.1
    --save_path [path to store models]
    --data_dir [path to imagenet data]
```

Linear evaluation:
```
python simclr_finetuner.py
    --gpus 4
    --ckpt_path [path to checkpoint]
    --dataset imagenet
    --data_dir [path to imagenet data]
    --batch_size 256
    --num_workers 16
    --learning_rate 0.8
    --nesterov True
    --num_epochs 90
```
