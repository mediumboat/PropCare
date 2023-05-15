import argparse
from train import prepare_data, train_propensity
from train import plotpath, Causal_Model
from baselines import DLMF
import numpy as np
import tensorflow as tf
from evaluator import Evaluator

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--dimension", default=128, type=int, help="number of features per user/item.")
parser.add_argument("--estimator_layer_units",
                    default=[64, 32, 16, 8],
                    type=list,
                    help="number of nodes each layer for MLP layers in Propensity and Relevance estimators")
parser.add_argument("--embedding_layer_units",
                    default=[256, 128, 64],
                    type=list,
                    help="number of nodes each layer for shared embedding layer.")
parser.add_argument("--click_layer_units",
                    default=[64, 32, 16, 8],
                    type=list,
                    help="number of nodes each layer for MLP layers in Click estimators")
parser.add_argument("--epoch", default=50, type=int,
                    help="Number of epochs in the training")

parser.add_argument("--lambda_1", default=10.0, type=float,
                    help="weight for popularity loss.")

parser.add_argument("--dataset", default='d', type=str,
                    help="the dataset used")

parser.add_argument("--batch_size", default=5096, type=int,
                    help="the batch size")
parser.add_argument("--repeat", default=3, type=int,
                    help="how many time to run the model")
parser.add_argument("--add", default='default', type=str,
                    help="additional information")
parser.add_argument("--p_weight", default=0.4, type=float,
                    help="weight for p_loss")
parser.add_argument("--pro_weight", default=1.0, type=float,
                    help="weight for propensity_loss")
parser.add_argument("--rele_weight", default=1.0, type=float,
                    help="weight for relevence_loss")
flag = parser.parse_args()


def main(flag=flag):
    cp10list = []
    cp100list = []
    cdcglist = []
    mselist = []
    maelist = []
    rmselist = []
    nrmselist = []
    taulist = []
    pearsonlist = []
    
    tmselist = []
    tmaelist = []
    trmselist = []
    tnrmselist = []
    ttaulist = []
    tpearsonlist = []
    tf1list = []
    tauclist = []
    random_seed = int(233)
    for epoch in range(flag.repeat):
        train_df, vali_df, test_df, num_users, num_items, num_times, popular = prepare_data(flag)
        random_seed += 1
        tf.random.set_seed(
            random_seed
        )
        model = train_propensity(train_df, vali_df, test_df, flag, num_users, num_items, num_times, popular)
        train_user = tf.convert_to_tensor(train_df["idx_user"].to_numpy(), dtype=tf.int32)
        train_item = tf.convert_to_tensor(train_df["idx_item"].to_numpy(), dtype=tf.int64)
        train_data = tf.data.Dataset.from_tensor_slices((train_user, train_item))

        test_user = tf.convert_to_tensor(test_df["idx_user"].to_numpy(), dtype=tf.int32)
        test_item = tf.convert_to_tensor(test_df["idx_item"].to_numpy(), dtype=tf.int64)
        test_data = tf.data.Dataset.from_tensor_slices((test_user, test_item))


#### EValuate on train data ##########
        p_pred = None
        for u, i in train_data.batch(5000):
            _, p_batch, _, _ = model((u, i), training=False)
            if p_pred is None:
                p_pred = p_batch
            else:
                p_pred = tf.concat((p_pred, p_batch), axis=0)
        p_pred = np.squeeze(p_pred.numpy())
        with open(plotpath+"/propensity_test_"+ flag.add+ "_" + flag.dataset +".txt", "w+") as f:
            np.savetxt(f, p_pred, fmt='%.4f', encoding="utf-8")
        # p_pred = (p_pred - np.min(p_pred)) / (np.max(p_pred) - np.min(p_pred))
        p_true = np.squeeze(train_df["propensity"].to_numpy())
        if flag.dataset == "ml":
            p_pred = p_pred * 0.2
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mse = mean_squared_error(y_pred=p_pred, y_true=p_true, squared=True)
        rmse = mean_squared_error(y_pred=p_pred, y_true=p_true, squared=False)
        mae = mean_absolute_error(y_pred=p_pred, y_true=p_true)
        nrmse = rmse / np.mean(p_true)
        mselist.append(mse)
        maelist.append(mae)
        rmselist.append(rmse)
        nrmselist.append(nrmse)
        from scipy.stats import kendalltau, pearsonr
        tau_res, _ = kendalltau(p_pred, p_true)
        taulist.append(tau_res)
        pearsonres, _ = pearsonr(p_pred, p_true)
        pearsonlist.append(pearsonres)

        t_true = np.squeeze(train_df["treated"].to_numpy())
        # p_pred = 0.25 * ((p_pred - np.mean(p_pred))/ (np.std(p_pred)))
        p_pred = np.clip((p_pred + 0.5), 0.0, 1.0)
        if flag.dataset == "d" or "p":
            flag.thres = 0.70
        elif flag.dataset == "ml":
            flag.thres = 0.65
            p_pred = p_pred / 5.0
        t_pred = np.where(p_pred >= flag.thres, 1.0, 0.0)
        # t_pred= np.random.binomial(1, p_pred)
        mse = mean_squared_error(y_pred=t_pred, y_true=t_true, squared=True)
        rmse = mean_squared_error(y_pred=t_pred, y_true=t_true, squared=False)
        mae = mean_absolute_error(y_pred=t_pred, y_true=t_true)
        nrmse = rmse / np.mean(t_true)
        tmselist.append(mse)
        tmaelist.append(mae)
        trmselist.append(rmse)
        tnrmselist.append(nrmse)
        tau_res, _ = kendalltau(t_pred, t_true)
        ttaulist.append(tau_res)
        pearsonres, _ = pearsonr(t_pred, t_true)
        tpearsonlist.append(pearsonres)
        from sklearn.metrics import f1_score, roc_auc_score
        tf1 = f1_score(y_true=t_true, y_pred=t_pred, average='macro')
        tauc = roc_auc_score(y_true=t_true, y_score=t_pred, average='macro')
        tf1list.append(tf1)
        tauclist.append(tauc)
        
 #### Train on training data ##########       
        gamma_pred = None
        p_pred = None
        for u, i in train_data.batch(5000):
            _, p_batch, _ ,_ = model((u, i), training=False)
            if p_pred is None:
                p_pred = p_batch
            else:
                p_pred = tf.concat((p_pred, p_batch), axis=0)
        p_pred = p_pred.numpy()
        with open(plotpath+"/propensity_train_"+ flag.add+ "_" + flag.dataset +".txt", "w+") as f:
            np.savetxt(f, p_pred, fmt='%.4f', encoding="utf-8")
        p_pred_t = 0.25 * ((p_pred - np.mean(p_pred))/ (np.std(p_pred)))
        p_pred_t = np.clip((p_pred + 0.5), 0.0, 1.0)
        if flag.dataset == "d" or "p":
            flag.thres = 0.70
        elif flag.dataset == "ml":
            flag.thres = 0.65
        if flag.dataset == "d" or "p":
            p_pred = p_pred * 0.8
        if flag.dataset == "ml":
            p_pred = p_pred * 0.2
        t_pred = np.where(p_pred_t >= flag.thres, 1.0, 0.0)
        # t_pred= np.random.binomial(1, p_pred)
        train_df["propensity"] = np.clip(p_pred, 0.0001, 0.9999)
        train_df["treated"] = t_pred
        
        if flag.dataset == "d":
            cap = 0.03
            lr = 0.001
            rf = 0.01
            itr = 1000e6
        if flag.dataset == "p":
            lr = 0.001
            cap = 0.5
            rf = 0.001
            itr = 1000e6
        if flag.dataset == "ml":
            lr = 0.001
            cap = 0.3
            rf = 0.1
            itr = 1000e6
        recommender = DLMF(num_users, num_items, capping_T=cap, capping_C=cap, learn_rate=lr, reg_factor=rf)
        recommender.train(train_df, iter=itr)
        cp10_tmp_list = []
        cp100_tmp_list = []
        cdcg_tmp_list = []
        if flag.dataset == 'd' or 'p':
            for t in range(num_times):
                test_df_t = test_df[test_df["idx_time"] == t]
                test_df_t["pred"] = recommender.predict(test_df_t)
                # print(test_df_t["pred"])
                evaluator = Evaluator()
                cp10_tmp_list.append(evaluator.evaluate(test_df_t, 'CPrec', 10))
                cp100_tmp_list.append(evaluator.evaluate(test_df_t, 'CPrec', 100))
                cdcg_tmp_list.append(evaluator.evaluate(test_df_t, 'CDCG', 100000))
        else:
            for t in [0]:
                test_df_t = test_df[test_df["idx_time"] == t]
                test_df_t["pred"] = recommender.predict(test_df_t)
                # print(test_df_t["pred"])
                evaluator = Evaluator()
                cp10_tmp_list.append(evaluator.evaluate(test_df_t, 'CPrec', 10))
                cp100_tmp_list.append(evaluator.evaluate(test_df_t, 'CPrec', 100))
                cdcg_tmp_list.append(evaluator.evaluate(test_df_t, 'CDCG', 100000))
        cp10 = np.mean(cp10_tmp_list)
        cp100 = np.mean(cp100_tmp_list)
        cdcg = np.mean(cdcg_tmp_list)
        cp10list.append(cp10)
        cp100list.append(cp100)
        cdcglist.append(cdcg)
    
    with open(plotpath+"/result_" + flag.dataset +".txt", "a+") as f:
        print("propensity MSE:", np.mean(mselist), np.std(mselist), file=f)
        print("propensity MAE:", np.mean(maelist), np.std(maelist), file=f)
        print("propensity RMSE:", np.mean(rmselist), np.std(rmselist), file=f)
        print("propensity NRMSE:", np.mean(nrmselist), np.std(nrmselist), file=f)
        print("Tau:", np.mean(taulist), np.std(taulist), file=f)
        print("Pearson:", np.mean(pearsonlist), np.std(pearsonlist), file=f)
        print("Exposure MSE:", np.mean(tmselist), np.std(tmselist), file=f)
        print("Exposure MAE:", np.mean(tmaelist), np.std(tmaelist), file=f)
        print("Exposure RMSE:", np.mean(trmselist), np.std(trmselist), file=f)
        print("Exposure NRMSE:", np.mean(tnrmselist), np.std(tnrmselist), file=f)
        print("Exposure Tau:", np.mean(ttaulist), np.std(ttaulist), file=f)
        print("Exposure Pearson:", np.mean(tpearsonlist), np.std(tpearsonlist), file=f)
        print("Exposure F1:", np.mean(tf1list), np.std(tf1list), file=f)
        print("Exposure AUC:", np.mean(tauclist), np.std(tauclist), file=f)
        print("CP10:", np.mean(cp10list), np.std(cp10list), file=f)
        print("CP100:", np.mean(cp100list), np.std(cp100list), file=f)
        print("CDCG:", np.mean(cdcglist), np.std(cdcglist), file=f)


            
if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass
    main(flag)
