# Physics-Informed Neural Operator (PITO)
Code for "[Physics-Informed Transformer operator for the prediction of three-dimensional turbulence](https://arxiv.org/abs/2601.19351)"
## Abstract
Data-driven turbulence prediction methods often face challenges related to data dependency and lack of physical interpretability. In this paper, we propose a physics-informed Transformer operator (PITO) and its implicit variant (PIITO) for predicting three-dimensional (3D) turbulence, which are developed based on the vision Transformer (ViT) architecture with an appropriate patch size. Given the current flow field, the Transformer operator computes its prediction for the next time step. By embedding the large-eddy simulation (LES) equations into the loss function, PITO and PIITO can learn solution operators without using labeled data. Furthermore, PITO can automatically learn the subgrid scale (SGS) coefficient using a single set of flow data during training. Both PITO and PIITO exhibit excellent stability and accuracy on the predictions of various statistical properties and flow structures for the situation of long-term extrapolation exceeding 25 times the training horizon in decaying homogeneous isotropic turbulence (HIT), and outperform the physics-informed Fourier neural operator (PIFNO). Furthermore, PITO exhibits a remarkable accuracy on the predictions of forced HIT where PIFNO fails. Notably, PITO and PIITO reduce graphics processing unit (GPU) memory consumption by 79.5\% and 91.3\% while requiring only 31.5\% and 3.1\% of the parameters, respectively, compared to PIFNO. Moreover, both PITO and PIITO models are much faster compared to traditional LES method.

## Datasets 
The dataset in used in the article are available for download at kaggle.
## Run Experiments
The config/directory contains all configuration files for the experiments:
| Configuration Directory | Description |
|------------------------|-------------|
| `config/DHIT/` | Decaying HIT at stationary initial condition |
| `config/DHIT_Random/` | Decaying HIT at random initial condition |
| `config/HIT/` | Forced HIT at stationary initial condition |
| `config/DHIT_parameters/` | Parameter sensitivity analysis for decaying HIT |
| `config/Cs_unknown/` | Automatic learning of the Smagorinsky coefficient (Cs) |

Run any experiment by specifying its configuration file: (e.g. DHIT)
```
python python train_pino.py --config config/DHIT/PITO.yaml
```
## Citation
```
@article{guo2026physics,
  title={Physics-Informed Transformer operator for the prediction of three-dimensional turbulence},
  author={Guo, Zhihong and Zhao, Sunan and Yang, Huiyu and Wang, Yunpeng and Wang, Jianchun},
  journal={arXiv preprint arXiv:2601.19351},
  year={2026}
}
```
