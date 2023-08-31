# DFT-NPZS-NAS: DFT based Prediction based Zero-shot NAS
[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE.md)

Authors.
Conference.

## Setup
- Clone repo.
- Install necessary packages.
```
$ pip install -r requirements.txt
```
-  Download databases 
```
$ wget downloaddata.sh
```

In our experiments, we do not implement directly the API benchmarks published in their repos (e.g., NAS-Bench-101, NAS-Bench-201, etc).
Instead, we create smaller-size databases by accessing their databases and only logging necessary content.

## Reproducing the results
You can reproduce our results by running the below script:
```shell
$ python train.py --benchmark [DARTS, NASNet, ENAS, PNAS, Amoeba, NB201, NB101, Macro]
```

## Evaluate
```shell
$ python test.py --checkpoint /path/to/checkpoint
```

## Search
```shell
$ python search.py --checkpoint /path/to/checkpoint
```
Please see specific file to modify the suitable argument parser

## Acknowledgement
Our source code is inspired by:
- [NAS-Bench-101: Towards Reproducible Neural Architecture Search](https://github.com/google-research/nasbench)
- [NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search](https://github.com/D-X-Y/NAS-Bench-201)
- [NAS-Bench-Macro: Prioritized Architecture Sampling with Monto-Carlo Tree Search](https://github.com/xiusu/NAS-Bench-Macro)
- [NDS: Designing Network Design Spaces](https://github.com/facebookresearch/pycls)
- [ZenNAS: : A Zero-Shot NAS for High-Performance Deep Image Recognition]([https://github.com/facebookresearch/pycls](https://github.com/idstcv/ZenNAS)https://github.com/idstcv/ZenNAS)
