# Robust Contrastive Learning against Noisy Views
This repository provides a PyTorch implementation of the **Robust InfoNCE** loss proposed in paper [Robust Contrastive Learning against Noisy Views](https://arxiv.org/pdf/2201.04309.pdf).

Requirements:
+ PyTorch >= 1.5.0


## Pseudo Code

The implementation only requires a small modification to the InfoNCE code.
```py    
# bsz : batch size (number of positive pairs)
# pos : exponent for positive example, shape=[bsz]
# neg : sum of exponents for negative examples, shape=[bsz]
# q, lam : hyperparameters of RINCE

info_nce_loss = -log(pos / (pos + neg))
rince_loss = -pos**q / q + (lam * (pos + neg))**q / q
```

## ImagNet Experiments

The code can be found in [ImageNet/moco-v3](https://github.com/chingyaoc/RINCE/tree/main/ImageNet/moco-v3) and [ImageNet/simclr](https://github.com/chingyaoc/RINCE/tree/main/ImageNet/simclr).

## Citation

If you find this repo useful for your research, please consider citing the paper

```
@article{chuang2022robust,
  title={Robust Contrastive Learning against Noisy Views},
  author={Chuang, Ching-Yao and Hjelm, R Devon and Wang, Xin and Vineet, Vibhav and Joshi, Neel and Torralba, Antonio and Jegelka, Stefanie and Song, Yale},
  journal={arXiv preprint arXiv:2201.04309},
  year={2022}
}
```
For any questions, please contact Ching-Yao Chuang (cychuang@mit.edu).

