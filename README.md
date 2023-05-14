# NeurIPS_PropCare_blind_review

This is the implementation of paper "Estimating Propensity for Causality-based Recommendation without Exposure Data". This code is for double-blind review only.

## Steps to run
1. Download code for generating semi-simulated from https://arxiv.org/abs/2008.04563 and https://arxiv.org/abs/2012.09442, generate the dataset as guided in their literatures.
2. Modify the path to dataset in `prepare_data` of train.py
3. Execute "python -u main.py --dataset d" for DH_original dataset, "python -u main.py --dataset p" for DH_personalized dataset, "python -u main.py --dataset ml" for ML dataset.

All results will be stored in "result_[dataset].txt"
