# Adversarial Bone Length Attack on Action Recognition (AAAI 2022)

## Description
This code is the source code of our paper "Adversarial Bone Length Attack on Action Recognition" in AAAI 2022.

Our Paper is [here](https://arxiv.org/abs/2109.05830).

In this code, we use [NTU RGB+D dataset](https://arxiv.org/pdf/1604.02808.pdf) and [ST-GCN model](https://arxiv.org/abs/1801.07455).

## Abstruct
Skeleton-based action recognition models have recently been shown to be vulnerable to adversarial attacks. Compared to adversarial attacks on images, perturbations to skeletons are typically bounded to a lower dimension of approximately 100 per frame. This lower-dimensional setting makes it more difficult to generate imperceptible perturbations. Existing attacks resolve this by exploiting the temporal structure of the skeleton motion so that the perturbation dimension increases to thousands. In this paper, we show that adversarial attacks can be performed on skeleton-based action recognition models, even in a significantly low-dimensional setting without any temporal manipulation. Specifically, we restrict the perturbations to the lengths of the skeleton's bones, which allows an adversary to manipulate only approximately 30 effective dimensions. We conducted experiments on the NTU RGB+D and HDM05 datasets and demonstrate that the proposed attack successfully deceived models with sometimes greater than 90% success rate by small perturbations. Furthermore, we discovered an interesting phenomenon: in our low-dimensional setting, the adversarial training with the bone length attack shares a similar property with data augmentation, and it not only improves the adversarial robustness but also improves the classification accuracy on the original data. This is an interesting counterexample of the trade-off between adversarial robustness and clean accuracy, which has been widely observed in studies on adversarial training in the high-dimensional regime.

## Notice
This code was custumized [the source code of ST-GCN](https://github.com/yysijie/st-gcn).  
Many thanks to the authors for all their work.

## Installation
### Create Environment
We support `Anaconda` environment. To create virtual environment, run:
```
conda conda env create -f requirements.yaml
conda activate hoge
cd torchlignt/
python setup.py install; cd ..
```
### Download Dataset and Get Pretrained Models
To download Dataset and get pretrained models, please refer to [the code of ST-GCN](https://github.com/yysijie/st-gcn).
Then, you should save the dataset to `./data/NTU-RGB-D/xsub/` or `./data/NTU-RGB-D/xview/` and save the pretrained model to `./models/`.

## How to Attack
### Collect Data Correctly Classified by Model
In our experiments, we used success rate as a evaluation metric.
Therefore, we collect the data that the model can correctly classifies before implementation of our attack.
To collect these data, you should run the following command.
```
python main.py recognition -c config/st_gcn/ntu-xsub/collect_data.yaml
```
or 
```
python main.py recognition -c config/st_gcn/ntu-xview/collect_data.yaml
```
### Implementation of Adversarial Bone Lentgh Attack
To implement adversarial bone length attack that we proposed, run the following command.
```
python main.py recognition -c config/st_gcn/ntu-xsub/adversarial_attack.yaml
```
or
```
python main.py recognition -c config/st_gcn/ntu-xview/adversarial_attack.yaml
```
Then, you can get the adversarial examples in `./xsub/results_ad/` or `./xview/results_ad/`.
You can also see the result in `./xsub/logs_ad/` or `./xview/logs_ad/`.

## Citation
Nariki Tanaka, Hiroshi Kera, Kazuhiko Kawamoto, Adversarial Bone Length Attack on Action Recognition, AAAI 2022.

@article{Tanaka_Kera_Kawamoto_2022,
	author = {Tanaka, Nariki and Kera, Hiroshi and Kawamoto, Kazuhiko},
	journal = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
	number = {2},
	pages = {2335-2343},
	title = {Adversarial Bone Length Attack on Action Recognition},
	volume = {36},
	year = {2022}}
