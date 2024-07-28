# Two-Stage Shadow Inclusion (2SSI)

Source code for [Two-Stage Shadow Inclusion Estimation: An IV Approach for Causal Inference under Latent Confounding and Collider Bias](https://proceedings.mlr.press/v235/li24bu.html).

## Get Started

1. To install the necessary packages, run the following command-line code.
```
pip install -r requirements.txt
```

2. Run the demo (experiments on Demand) in `main.py`.

## Useful Links

- The mutual information estimator is adapted from [CLUB](https://github.com/Linear95/CLUB). 
- The overall framework of the code is adapted from [DFIV](https://github.com/liyuan9988/DeepFeatureIV). You can also find more baselines there.
- The data generation code is adapted from [DeepIV](https://github.com/jhartford/DeepIV).

## Citation

```
@InProceedings{pmlr-v235-li24bu,
  title={Two-Stage Shadow Inclusion Estimation: An {IV} Approach for Causal Inference under Latent Confounding and Collider Bias},
  author={Li, Baohong and Wu, Anpeng and Xiong, Ruoxuan and Kuang, Kun},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  pages={28949--28964},
  year={2024},
  organization={PMLR}
}
```