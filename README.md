# Refined Spatio-Temporal Pipeline Representation Network for Infrared Small Moving Target Detection

This repository provides the **PyTorch implementation of RPRNet**, proposed in  
**“Refined Spatio-Temporal Pipeline Representation Network for Infrared Small Moving Target Detection.”**

RPRNet is designed for infrared weak and small moving target detection under complex and dynamic background conditions by preserving and consistently propagating high-fidelity spatio-temporal pipeline representations.

---

## Abstract

Small Moving target detection aims to exploit spatio-temporal information to discriminate targets from background clutter. However, the extremely small target size, weak signal intensity, and highly dynamic backgrounds make accurate target extraction particularly challenging. Most existing methods rely on motion compensation and temporal alignment to capture temporal cues, but they often suffer from severe information loss in shallow network layers, which prevents subsequent modules from further improving detection performance based on degraded representations. To address these limitations, we propose a Refined Spatio-Temporal Pipeline Representation Network (RPRNet) for infrared weak and small moving target detection. The core idea is to preserve and consistently propagate joint spatio-temporal features, enabling refined pipeline-wise representation of target and background spatio-temporal characteristics for high-precision detection under dynamic backgrounds. Specifically, we first design a Spatio-Temporal Pipeline Representation Block (STRB), which adopts a unified three-dimensional spatio-temporal modeling mechanism to capture complex spatio-temporal variation patterns. We then introduce IG-GSM, a cross-scale spatio-temporal information preservation module that adaptively regulates high-resolution upsampling points for small targets, enhancing the continuity and consistency of target–background spatio-temporal pipeline representations. Finally, we propose SPGloss, which incorporates a progressive edge-expansion guidance mechanism to alleviate class imbalance while providing more stable supervision for weak targets, thereby maintaining the integrity of spatio-temporal pipelines. Experimental results on the NUDT-MIRSDT and TSIRMT datasets demonstrate that our network outperforms state-of-the-art (SOTA) methods.

## Architecture
<img width="1227" height="773" alt="image" src="https://github.com/user-attachments/assets/190838f5-f3cb-485e-95fc-367fff547297" />



## Datasets
For downloading the **NUDT-MIRSDT** and **TSIRMT** datasets, we provide the official links released by the original authors:  
- [NUDT-MIRSDT (DTUM)](https://github.com/TinaLRJ/Multi-frame-infrared-small-target-detection-DTUM)  
- [TSIRMT (LMAFormer)](https://github.com/lifier/LMAFormer?tab=readme-ov-file)



## HowToTrain
```bash
python train.py
```
You can switch datasets by modifying the `--dataset_names` argument. For other configurable parameters, please refer to `train.py`.



## HowToTest
```bash
python test.py
```
You can switch datasets by modifying the `--dataset_names` argument. For other configurable parameters, please refer to `test.py`.


## Results Summary
| Dataset |  Pd (%) |  Fa | 
|:------:|:------------------:|:-------------:|
| NUDT-MIRSDT   | 97.05              | 0.50e-6       |         
|   TSIRMT      | 87.00              | 140.50e-6     |


## Acknowledgement
We would like to thank [Yuan](https://github.com/xdFai/SCTransNet) et al. for making their code publicly available. Part of the code in this repository is collected and adapted from their work.
