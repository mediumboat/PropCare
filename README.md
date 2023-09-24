# Code repository of PropCare

This is the implementation of paper "Estimating Propensity for Causality-based Recommendation without Exposure Data".

## Description of each file:

baselines.py: including codes for DLCE model (or some other recommendation models)

evaluator.py: evaluation code to compute CP@10, CP@100 and CDCG

main: the entrance of the program

models.py: codes for our proposed PropCare model

train.py: training codes, also including data loading related codes

## Dataset & How to run:

1. Since the semi-simulated dataset is quite large (~2GB each), you should download from the original source.
Download the raw data and code for generating semi-simulated from https://arxiv.org/abs/2008.04563 (DH) and https://arxiv.org/abs/2012.09442 (ML), generate the dataset as guided in their `README` files.

2. Modify the path to dataset in `prepare_data` of train.py

3. Set all parameters in main.py

4. Execute "python -u main.py --dataset d" for DH_original dataset, "python -u main.py --dataset p" for DH_personalized dataset, "python -u main.py --dataset ml" for ML dataset.

## Contact
For any questions, please contact me (zzliu[DOT]2020[AT]phdcs[DOT]smu[DOT]edu[DOT]sg)
