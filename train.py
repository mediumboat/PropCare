import pandas as pd
from pathlib import Path
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from models import Causal_Model
from evaluator import Evaluator
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import random
import os
import itertools

plotpath = "./results/"
if not os.path.isdir(plotpath):
    os.makedirs(plotpath)
def diff(list1, list2):
    return list(set(list2).difference(set(list1)))


def sparse_gather(indices, values, selected_indices, axis=0):
    """
    indices: [[idx_ax0, idx_ax1, idx_ax2, ..., idx_axk], ... []]
    values:  [ value1,                                 , ..., valuen]
    """
    mask = tf.equal(indices[:, axis][tf.newaxis, :], selected_indices[:, tf.newaxis])
    to_select = tf.where(mask)[:, 1]
    user_item = tf.gather(indices, to_select, axis=0)
    user = tf.gather(user_item, 0, axis=1)
    item = tf.gather(user_item, 1, axis=1)
    values = tf.gather(values, to_select, axis=0)
    return user, item, values


def count_freq(x):
    unique, counts = np.unique(x, return_counts=True)
    return np.asarray((unique, counts)).T


def prepare_data(flag):
    dataset = flag.dataset
    data_path = None
    if dataset == "d":
        print("dunn_cate (original) is used.")
        data_path = Path("../anc/UnbiasedLearningCausal/data/preprocessed/dunn_cat_mailer_10_10_1_1/original_rp0.40/")
    elif dataset == "p":
        print("dunn_cate (personalized) is used.")
        data_path = Path("../anc/UnbiasedLearningCausal/data/preprocessed/dunn_cat_mailer_10_10_1_1/rank_rp0.40_sf2.00_nr210/")
    elif dataset == "ml":
        data_path = Path("../ml/anc/data/synthetic/ML_100k_logrank100_offset5.0_scaling1.0")
        print("ML-100k is used")
    train_data = data_path / "data_train.csv"
    vali_data = data_path / "data_vali.csv"
    test_data = data_path / "data_test.csv"
    train_df = pd.read_csv(train_data)
    vali_df = pd.read_csv(vali_data)
    test_df = pd.read_csv(test_data)
    user_ids = np.sort(
        pd.concat([train_df["idx_user"], vali_df["idx_user"], test_df["idx_user"]]).unique().tolist())
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    item_ids = np.sort(
        pd.concat([train_df["idx_item"], vali_df["idx_item"], test_df["idx_item"]]).unique().tolist())
    item2item_encoded = {x: i for i, x in enumerate(item_ids)}
    train_df["idx_user"] = train_df["idx_user"].map(user2user_encoded)
    train_df["idx_item"] = train_df["idx_item"].map(item2item_encoded)
    vali_df["idx_user"] = vali_df["idx_user"].map(user2user_encoded)
    vali_df["idx_item"] = vali_df["idx_item"].map(item2item_encoded)
    test_df["idx_user"] = test_df["idx_user"].map(user2user_encoded)
    test_df["idx_item"] = test_df["idx_item"].map(item2item_encoded)
    num_users = len(user_ids)
    num_items = len(item_ids)
    print(num_items)
    if dataset == "d" or dataset == "p":
        num_times = len(train_df["idx_time"].unique().tolist())
    else: 
        num_times = 1
        train_df["idx_time"] = 0
        vali_df["idx_time"] = 0
        test_df["idx_time"] = 0
    train_df = train_df[["idx_user", "idx_item", "outcome", "idx_time", "propensity", "treated"]]
    train_df_positive = train_df[train_df["outcome"] > 0]
    counts = count_freq(train_df_positive['idx_item'].to_numpy())
    np_counts = np.zeros(num_items)
    print(np_counts.shape)
    np_counts[counts[:, 0].astype(int)] = counts[:, 1].astype(int)

    return train_df, vali_df, test_df, num_users, num_items, num_times, np_counts


def train_propensity(train_df, vali_df, test_df, flag, num_users, num_items, num_times, popular):
    from scipy.stats import kendalltau, pearsonr
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    model = Causal_Model(num_users, num_items, flag, None, None, popular)
    optim_val_car = 0
    # train_df = train_df[train_df["outcome"] > 0]
    for epoch in range(flag.epoch):
        print("Sampling negative items...", end=" ")
        j_list = []
        for i in train_df["idx_item"].to_numpy():
            j = np.random.randint(0, num_items)
            while j == i:
                j = np.random.randint(0, num_items)
            j_list.append(j)
        print("Done")
        j_list = np.reshape(np.array(j_list, dtype=train_df["idx_item"].to_numpy().dtype), train_df["idx_item"].to_numpy().shape)
        train_data = tf.data.Dataset.from_tensor_slices((train_df["idx_user"].to_numpy(), train_df["idx_item"].to_numpy(), j_list, train_df["outcome"].to_numpy()))
        with tqdm(total=len(train_df) // flag.batch_size + 1) as t:
            t.set_description('Training Epoch %i' % epoch)
            for user, item, item_j, value in train_data.shuffle(50000).batch(flag.batch_size):
                step = model.propensity_train((user, item, item_j, value))
                t.set_postfix(click_loss=step["click_loss"].numpy(), estimator_loss=step['estimator_loss'].numpy(), reg_loss=step['reg_loss'].numpy())
                t.update()
                model.click_loss_tracker.reset_states()
                model.estimator_loss_tracker.reset_states()
                model.reg_loss_tracker.reset_states()
        vali_data = tf.data.Dataset.from_tensor_slices((vali_df["idx_user"].to_numpy(), vali_df["idx_item"].to_numpy()))
        gamma_pred = None
        p_pred = None
        for u, i in vali_data.batch(5000):
            gamma_batch, p_batch, _, _ = model((u, i), training=False)
            if gamma_pred is None:
                gamma_pred = gamma_batch
            else:
                gamma_pred = tf.concat((gamma_pred, gamma_batch), axis=0)
            if p_pred is None:
                p_pred = p_batch
            else:
                p_pred = tf.concat((p_pred, p_batch), axis=0)
        p_true = np.squeeze(vali_df["propensity"].to_numpy())
        p_pred = np.squeeze(p_pred.numpy())
        p_pred = (p_pred - np.min(p_pred)) / (np.max(p_pred) - np.min(p_pred))
        tau_res, _ = kendalltau(p_pred, p_true)
        pearsonres, _ = pearsonr(p_pred, p_true)
        mse = mean_squared_error(y_pred=p_pred, y_true=p_true, squared=True)
        print("Pearson:", pearsonres)
        print("Tau:", tau_res)
        print("Propensity prediction square error:", mse)
        print("weight:", model.exp_weight.numpy())
        val_obj = pearsonres
        if abs(val_obj) > optim_val_car:
            optim_val_car = val_obj
            if not os.path.isdir(plotpath+ '/' + flag.add):
                os.makedirs(plotpath+ '/' + flag.add)
            model.save_weights(plotpath+ '/' + flag.add + "/saved_model")
            print("Model saved!")
    model.load_weights(plotpath+ '/' + flag.add + "/saved_model")
    return model

if __name__ == "__main__":
    pass