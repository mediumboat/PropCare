import numpy as np
from pathlib import Path
from numpy.random.mtrand import RandomState
import random
import pandas as pd
from evaluator import Evaluator


class Recommender(object):

    def __init__(self, num_users, num_items,
                 colname_user = 'idx_user', colname_item = 'idx_item',
                 colname_outcome = 'outcome', colname_prediction='pred',
                 colname_treatment='treated', colname_propensity='propensity'):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.colname_user = colname_user
        self.colname_item = colname_item
        self.colname_outcome = colname_outcome
        self.colname_prediction = colname_prediction
        self.colname_treatment = colname_treatment
        self.colname_propensity = colname_propensity

    def train(self, df, iter=100):
        pass

    def predict(self, df):
        pass

    def recommend(self, df, num_rec=10):
        pass

    def func_sigmoid(self, x):
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            return np.exp(x) / (1.0 + np.exp(x))

    def sample_time(self):
        return random.randrange(self.num_times)

    def sample_user(self, idx_time, TP=True, TN=True, CP=True, CN=True):
        while True:
            flag_condition = 1
            u = random.randrange(self.num_users)
            if TP:
                if u not in self.dict_treatment_positive_sets[idx_time]:
                    flag_condition = 0
            if TN:
                if u not in self.dict_treatment_negative_sets[idx_time]:
                    flag_condition = 0
            if CP:
                if u not in self.dict_control_positive_sets[idx_time]:
                    flag_condition = 0
            if CN:
                if u not in self.dict_control_negative_sets[idx_time]:
                    flag_condition = 0
            if flag_condition > 0:
                return u

    def sample_treatment(self, idx_time, idx_user):
        return random.choice(self.dict_treatment_sets[idx_time][idx_user])

    def sample_control(self, idx_time, idx_user):
        while True:
            flag_condition = 1
            i = random.randrange(self.num_items)
            if idx_user in self.dict_treatment_positive_sets[idx_time]:
                if i in self.dict_treatment_positive_sets[idx_time][idx_user]:
                    flag_condition = 0
            if idx_user in self.dict_treatment_negative_sets[idx_time]:
                if i in self.dict_treatment_negative_sets[idx_time][idx_user]:
                    flag_condition = 0
            if flag_condition > 0:
                return i

    # in case control is rare
    def sample_control2(self, idx_time, idx_user):
        cand_control = np.arange(self.num_items)
        cand_control = cand_control[np.isin(cand_control, self.dict_treatment_sets[idx_time][idx_user])]
        return random.choice(cand_control)

    def sample_treatment_positive(self, idx_time, idx_user):
        return random.choice(self.dict_treatment_positive_sets[idx_time][idx_user])

    def sample_treatment_negative(self, idx_time, idx_user):
        return random.choice(self.dict_treatment_negative_sets[idx_time][idx_user])

    def sample_control_positive(self, idx_time, idx_user):
        return random.choice(self.dict_control_positive_sets[idx_time][idx_user])

    def sample_control_negative(self, idx_time, idx_user):
        while True:
            flag_condition = 1
            i = random.randrange(self.num_items)
            if idx_user in self.dict_treatment_positive_sets[idx_time]:
                if i in self.dict_treatment_positive_sets[idx_time][idx_user]:
                    flag_condition = 0
            if idx_user in self.dict_treatment_negative_sets[idx_time]:
                if i in self.dict_treatment_negative_sets[idx_time][idx_user]:
                    flag_condition = 0
            if idx_user in self.dict_control_positive_sets[idx_time]:
                if i in self.dict_control_positive_sets[idx_time][idx_user]:
                    flag_condition = 0
            if flag_condition > 0:
                return i

    # TP: treatment-positive
    # CP: control-positive
    # TN: treatment-negative
    # TN: control-negative
    def sample_triplet(self):
        t = self.sample_time()
        if random.random() <= self.alpha:  # CN as positive
            if random.random() <= 0.5:  # TP as positive
                if random.random() <= 0.5:  # TP vs. TN
                    u = self.sample_user(t, TP=True, TN=True, CP=False, CN=False)
                    i = self.sample_treatment_positive(t, u)
                    j = self.sample_treatment_negative(t, u)
                else:  # TP vs. CP
                    u = self.sample_user(t, TP=True, TN=False, CP=True, CN=False)
                    i = self.sample_treatment_positive(t, u)
                    j = self.sample_control_positive(t, u)
            else:  # CN as positive
                if random.random() <= 0.5:  # CN vs. TN
                    u = self.sample_user(t, TP=False, TN=True, CP=False, CN=True)
                    i = self.sample_control_negative(t, u)
                    j = self.sample_treatment_negative(t, u)
                else:  # CN vs. CP
                    u = self.sample_user(t, TP=False, TN=False, CP=True, CN=True)
                    i = self.sample_control_negative(t, u)
                    j = self.sample_control_positive(t, u)
        else:  # CN as negative
            if random.random() <= 0.333:  # TP vs. CN
                u = self.sample_user(t, TP=True, TN=False, CP=False, CN=True)
                i = self.sample_treatment_positive(t, u)
                j = self.sample_control_negative(t, u)
            elif random.random() <= 0.5:  # TP vs. TN
                u = self.sample_user(t, TP=True, TN=True, CP=False, CN=False)
                i = self.sample_treatment_positive(t, u)
                j = self.sample_treatment_negative(t, u)
            else:  # TP vs. CP
                u = self.sample_user(t, TP=True, TN=False, CP=True, CN=False)
                i = self.sample_treatment_positive(t, u)
                j = self.sample_control_positive(t, u)

        return u, i, j

    def sample_pair(self):
        t = self.sample_time()
        if random.random() < 0.5: # pick treatment
            if random.random() > self.ratio_nega: # TP
                u = self.sample_user(t, TP=True, TN=False, CP=False, CN=False)
                i = self.sample_treatment_positive(t, u)
                flag_positive = 1
            else: # TN
                u = self.sample_user(t, TP=False, TN=True, CP=False, CN=False)
                i = self.sample_treatment_negative(t, u)
                flag_positive = 0
        else: # pick control
            if random.random() > self.ratio_nega:  # CP
                u = self.sample_user(t, TP=False, TN=False, CP=True, CN=False)
                i = self.sample_control_positive(t, u)
                flag_positive = 0
            else:  # CN
                u = self.sample_user(t, TP=False, TN=False, CP=False, CN=True)
                i = self.sample_control_negative(t, u)
                if random.random() <= self.alpha:  # CN as positive
                    flag_positive = 1
                else:
                    flag_positive = 0

        return u, i, flag_positive

    # getter
    def get_propensity(self, idx_user, idx_item):
        return self.dict_propensity[idx_user][idx_item]


class LMF(Recommender):
    def __init__(self, num_users, num_items,
                 metric='AUC', ratio_nega=0.8,
                 dim_factor=200, with_bias=False,
                 learn_rate = 0.01, reg_factor = 0.01, reg_bias = 0.01, sd_init = 0.1,
                 reg_factor_j=0.01, reg_bias_j=0.01,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred',
                 colname_treatment='treated', colname_propensity='propensity'):
        super().__init__(num_users=num_users, num_items=num_items,
                         colname_user=colname_user, colname_item=colname_item,
                         colname_outcome=colname_outcome, colname_prediction=colname_prediction,
                         colname_treatment=colname_treatment, colname_propensity=colname_propensity)
        self.metric = metric
        self.ratio_nega = ratio_nega
        self.dim_factor = dim_factor
        self.rng = RandomState(seed=None)
        self.with_bias = with_bias

        self.learn_rate = learn_rate
        self.reg_bias = reg_bias
        self.reg_factor = reg_factor
        self.sd_init = sd_init
        self.reg_bias_j = reg_bias_j
        self.reg_factor_j = reg_factor_j
        self.flag_prepared = False

        self.user_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_users, self.dim_factor))
        self.item_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_items, self.dim_factor))
        if self.with_bias:
            self.user_biases = np.zeros(self.num_users)
            self.item_biases = np.zeros(self.num_items)
            self.global_bias = 0.0

    def prepare_dictionary(self, df, colname_time='idx_time'):
        print("start prepare dictionary")
        self.colname_time = colname_time
        self.num_times = np.max(df.loc[:, self.colname_time]) + 1
        self.dict_positive_sets = dict()

        df_posi = df.loc[df.loc[:, self.colname_outcome] > 0]

        for t in np.arange(self.num_times):
            df_t = df_posi.loc[df_posi.loc[:, self.colname_time] == t]
            self.dict_positive_sets[t] = dict()
            for u in np.unique(df_t.loc[:, self.colname_user]):
                self.dict_positive_sets[t][u] = \
                    np.unique(df_t.loc[df_t.loc[:, self.colname_user] == u, self.colname_item].values)

        self.flag_prepared = True
        print("prepared dictionary!")


    def train(self, df, iter = 10):

        df_train = df.loc[df.loc[:, self.colname_outcome] > 0, :]  # need only positive outcomes
        if not self.flag_prepared: # prepare dictionary
            self.prepare_dictionary(df)

        err = 0
        current_iter = 0
        while True:
            df_train = df_train.sample(frac=1)
            users = df_train.loc[:, self.colname_user].values
            items = df_train.loc[:, self.colname_item].values
            times = df_train.loc[:, self.colname_time].values

            if self.metric == 'AUC': # BPR
                for n in np.arange(len(df_train)):
                    u = users[n]
                    i = items[n]
                    t = times[n]

                    while True:
                        j = random.randrange(self.num_items)
                        if not j in self.dict_positive_sets[t][u]:
                            break

                    u_factor = self.user_factors[u, :]
                    i_factor = self.item_factors[i, :]
                    j_factor = self.item_factors[j, :]

                    diff_rating = np.sum(u_factor * (i_factor - j_factor))

                    if self.with_bias:
                        diff_rating += (self.item_biases[i] - self.item_biases[j])

                    coeff = self.func_sigmoid(-diff_rating)

                    err += coeff

                    self.user_factors[u, :] += \
                        self.learn_rate * (coeff * (i_factor - j_factor) - self.reg_factor * u_factor)
                    self.item_factors[i, :] += \
                        self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor)
                    self.item_factors[j, :] += \
                        self.learn_rate * (-coeff * u_factor - self.reg_factor_j * j_factor)

                    if self.with_bias:
                        self.item_biases[i] += \
                            self.learn_rate * (coeff - self.reg_bias * self.item_biases[i])
                        self.item_biases[j] += \
                            self.learn_rate * (-coeff - self.reg_bias_j * self.item_biases[j])

            current_iter += 1
            if current_iter >= iter:
                return err/iter

            elif self.metric == 'logloss': # essentially WRMF with downsampling
                for n in np.arange(len(df_train)):
                    u = users[n]
                    i = items[n]
                    t = times[n]
                    flag_positive = 1

                    if np.random.rand() < self.ratio_nega:
                        flag_positive = 0
                        i = np.random.randint(self.num_items)
                        while True:
                            if not i in self.dict_positive_sets[t][u]:
                                break
                            else:
                                i = np.random.randint(self.num_items)

                    u_factor = self.user_factors[u, :]
                    i_factor = self.item_factors[i, :]

                    rating = np.sum(u_factor * i_factor)

                    if self.with_bias:
                        rating += self.item_biases[i] + self.user_biases[u] + self.global_bias

                    if flag_positive > 0:
                        coeff = 1 / (1 + np.exp(rating))
                    else:
                        coeff = -1 / (1 + np.exp(-rating))

                    err += np.abs(coeff)

                    self.user_factors[u, :] += \
                        self.learn_rate * (coeff * i_factor - self.reg_factor * u_factor)
                    self.item_factors[i, :] += \
                        self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor)

                    if self.with_bias:
                        self.item_biases[i] += \
                            self.learn_rate * (coeff - self.reg_bias * self.item_biases[i])
                        self.user_biases[u] += \
                            self.learn_rate * (coeff - self.reg_bias * self.user_biases[u])
                        self.global_bias += \
                            self.learn_rate * (coeff)

                    current_iter += 1
                    if current_iter >= iter:
                        return err / iter

    def predict(self, df):
        users = df[self.colname_user].values
        items = df[self.colname_item].values
        pred = np.zeros(len(df))
        for n in np.arange(len(df)):
            pred[n] = np.inner(self.user_factors[users[n], :], self.item_factors[items[n], :])
            if self.with_bias:
                pred[n] += self.item_biases[items[n]]
                pred[n] += self.user_biases[users[n]]
                pred[n] += self.global_bias

        # pred = 1 / (1 + np.exp(-pred))
        return pred



def prepare_data(dataset="dunn_cate"):
    if dataset == "dunn_cate":
        data_path = Path("../anc/UnbiasedLearningCausal/data/preprocessed/dunn_cat_mailer_10_10_1_1/original_rp0.40")
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
        num_users = len(user2user_encoded)
        num_items = len(item2item_encoded)
        return train_df, vali_df, test_df, num_users, num_items

class PopularBase(Recommender):

    def __init__(self, num_users, num_items,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred'):
        super().__init__(num_users=num_users, num_items=num_items,
                         colname_user=colname_user, colname_item=colname_item,
                         colname_outcome=colname_outcome, colname_prediction=colname_prediction)

    def train(self, df, iter = 1):
        df_cnt = df.groupby(self.colname_item, as_index=False)[self.colname_outcome].sum()
        df_cnt['prob'] = df_cnt[self.colname_outcome] /self.num_users
        self.df_cnt = df_cnt

    def predict(self, df):
        df = pd.merge(df, self.df_cnt, on=self.colname_item, how='left')
        return df.loc[:, 'prob'].values

class DLMF2(Recommender): # This version consider a scale factor alpha
    def __init__(self, num_users, num_items,
                 metric='AR_logi', capping_T=0.01, capping_C=0.01,
                 dim_factor=200, with_bias=False, with_IPS=True,
                 only_treated=False,
                 learn_rate = 0.01, reg_factor = 0.01, reg_bias = 0.01,
                 sd_init = 0.1, reg_factor_j = 0.01, reg_bias_j = 0.01,
                 coeff_T = 1.0, coeff_C = 1.0,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred',
                 colname_treatment='treated', colname_propensity='propensity'):
        super().__init__(num_users=num_users, num_items=num_items,
                         colname_user=colname_user, colname_item=colname_item,
                         colname_outcome=colname_outcome, colname_prediction=colname_prediction,
                         colname_treatment=colname_treatment, colname_propensity=colname_propensity)
        self.metric = metric
        self.capping_T = capping_T
        self.capping_C = capping_C
        self.with_IPS = with_IPS
        self.dim_factor = dim_factor
        self.rng = RandomState(seed=None)
        self.with_bias = with_bias
        self.coeff_T = coeff_T
        self.coeff_C = coeff_C
        self.learn_rate = learn_rate
        self.reg_bias = reg_factor
        self.reg_factor = reg_factor
        self.reg_bias_j = reg_factor
        self.reg_factor_j = reg_factor
        self.sd_init = sd_init
        self.only_treated = only_treated

        self.user_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_users, self.dim_factor))
        self.item_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_items, self.dim_factor))
        if self.with_bias:
            self.user_biases = np.zeros(self.num_users)
            self.item_biases = np.zeros(self.num_items)
            self.global_bias = 0.0
        self.alpha = 0.5
        
    def train(self, df, iter = 100):
        df_train = df.loc[df.loc[:, self.colname_outcome] > 0, :] # need only positive outcomes
        if self.only_treated: # train only with treated positive (DLTO)
            df_train = df_train.loc[df_train.loc[:, self.colname_treatment] > 0, :]

        if self.capping_T is not None:
            bool_cap = np.logical_and(df_train.loc[:, self.colname_propensity] < self.capping_T, df_train.loc[:, self.colname_treatment] == 1)
            if np.sum(bool_cap) > 0:
                df_train.loc[bool_cap, self.colname_propensity] = self.capping_T
        if self.capping_C is not None:      
            bool_cap = np.logical_and(df_train.loc[:, self.colname_propensity] > 1 - self.capping_C, df_train.loc[:, self.colname_treatment] == 0)
            if np.sum(bool_cap) > 0:
                df_train.loc[bool_cap, self.colname_propensity] = 1 - self.capping_C

        # if self.with_IPS: # point estimate of individual treatment effect (ITE) <- for binary outcome abs(ITE) = IPS
        #     df_train.loc[:, 'ITE'] =  df_train.loc[:, self.colname_treatment] * df_train.loc[:, self.colname_outcome]/df_train.loc[:, self.colname_propensity] - \
        #                               (1 - df_train.loc[:, self.colname_treatment]) * df_train.loc[:, self.colname_outcome]/(1 - df_train.loc[:, self.colname_propensity])
        #     z_y_1 = df_train.loc[:, self.colname_treatment] * df_train.loc[:, self.colname_outcome]
        #     z_y_1 = z_y_1.values
        #     z_y_0 = (1 - df_train.loc[:, self.colname_treatment]) * df_train.loc[:, self.colname_outcome]
        #     z_y_0 = z_y_0.values

        else:
            df_train.loc[:, 'ITE'] =  df_train.loc[:, self.colname_treatment] * df_train.loc[:, self.colname_outcome]  - \
                                      (1 - df_train.loc[:, self.colname_treatment]) * df_train.loc[:, self.colname_outcome]
        self.propensity = df_train.loc[:, self.colname_propensity].values
        err = 0
        current_iter = 0
        while True:
            df_train = df_train.sample(frac=1)
            users = df_train.loc[:, self.colname_user].values
            items = df_train.loc[:, self.colname_item].values
            propensity_current = self.propensity * self.alpha
            treat = df_train.loc[:, self.colname_treatment].values
            outcome = df_train.loc[:, self.colname_outcome].values
            ITE = df_train.loc[:, self.colname_treatment] * df_train.loc[:, self.colname_outcome]/propensity_current - \
                                      (1 - df_train.loc[:, self.colname_treatment]) * df_train.loc[:, self.colname_outcome]/(1 - propensity_current)
            z_y_1 = df_train.loc[:, self.colname_treatment] * df_train.loc[:, self.colname_outcome]
            z_y_1 = z_y_1.values
            z_y_0 = (1 - df_train.loc[:, self.colname_treatment]) * df_train.loc[:, self.colname_outcome]
            z_y_0 = z_y_0.values


            ITE = ITE.values

            if self.metric in ['AR_logi', 'AR_sig', 'AR_hinge']:
                for n in np.arange(len(df_train)):

                    u = users[n]
                    i = items[n]
                    propensity_current = self.propensity[n] * self.alpha
                    ITE = treat[n] * outcome[n]/propensity_current - \
                                      (1 - treat[n]) * outcome[n]/(1 - propensity_current)
                    z_y_1 = treat[n] * outcome[n]
                    z_y_0 = (1 - treat[n]) * outcome[n]
                    while True:
                        j = random.randrange(self.num_items)
                        if i != j:
                            break

                    u_factor = self.user_factors[u, :]
                    i_factor = self.item_factors[i, :]
                    j_factor = self.item_factors[j, :]

                    diff_rating = np.sum(u_factor * (i_factor - j_factor))
                    if self.with_bias:
                        diff_rating += (self.item_biases[i] - self.item_biases[j])

                    if self.metric == 'AR_logi':
                        if ITE >= 0:
                            coeff = ITE * self.coeff_T * self.func_sigmoid(-self.coeff_T * diff_rating) # Z=1, Y=1
                            const_value = z_y_1 * self.coeff_T * self.func_sigmoid(-self.coeff_T * diff_rating)
                        else:
                            coeff = ITE * self.coeff_C * self.func_sigmoid(self.coeff_C * diff_rating) # Z=0, Y=1
                            const_value = z_y_0 * self.coeff_C * self.func_sigmoid(self.coeff_C * diff_rating)

                    elif self.metric == 'AR_sig':
                        if ITE[n] >= 0:
                            coeff = ITE[n] * self.coeff_T * self.func_sigmoid(self.coeff_T * diff_rating) * self.func_sigmoid(-self.coeff_T * diff_rating)
                        else:
                            coeff = ITE[n] * self.coeff_C * self.func_sigmoid(self.coeff_C * diff_rating) * self.func_sigmoid(-self.coeff_C * diff_rating)

                    elif self.metric == 'AR_hinge':
                        if ITE[n] >= 0:
                            if self.coeff_T > 0 and diff_rating < 1.0/self.coeff_T:
                                coeff = ITE[n] * self.coeff_T 
                            else:
                                coeff = 0.0
                        else:
                            if self.coeff_C > 0 and diff_rating > -1.0/self.coeff_C:
                                coeff = ITE[n] * self.coeff_C
                            else:
                                coeff = 0.0

                    err += np.abs(coeff)

                    self.user_factors[u, :] += \
                        self.learn_rate * (coeff * (i_factor - j_factor) - self.reg_factor * u_factor)
                    self.item_factors[i, :] += \
                        self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor)
                    self.item_factors[j, :] += \
                        self.learn_rate * (-coeff * u_factor - self.reg_factor_j * j_factor)
                    self.alpha += self.learn_rate * (const_value / np.power(self.alpha, 2))
                    print(self.alpha)

                    if self.with_bias:
                        self.item_biases[i] += \
                            self.learn_rate * (coeff - self.reg_bias * self.item_biases[i])
                        self.item_biases[j] += \
                            self.learn_rate * (-coeff - self.reg_bias_j * self.item_biases[j])

                    current_iter += 1
                    if current_iter % 100000 == 0:
                        print(str(current_iter)+"/"+str(iter))
                    if current_iter >= iter:
                        return self.alpha

    def predict(self, df):
        users = df[self.colname_user].values
        items = df[self.colname_item].values
        pred = np.zeros(len(df))
        for n in np.arange(len(df)):
            pred[n] = np.inner(self.user_factors[users[n], :], self.item_factors[items[n], :])
            if self.with_bias:
                pred[n] += self.item_biases[items[n]]
                pred[n] += self.user_biases[users[n]]
                pred[n] += self.global_bias

        # pred = 1 / (1 + np.exp(-pred))
        return pred



class DLMF(Recommender):
    def __init__(self, num_users, num_items,
                 metric='AR_logi', capping_T=0.01, capping_C=0.01,
                 dim_factor=200, with_bias=False, with_IPS=True,
                 only_treated=False,
                 learn_rate = 0.01, reg_factor = 0.01, reg_bias = 0.01,
                 sd_init = 0.1, reg_factor_j = 0.01, reg_bias_j = 0.01,
                 coeff_T = 1.0, coeff_C = 1.0,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred',
                 colname_treatment='treated', colname_propensity='propensity'):
        super().__init__(num_users=num_users, num_items=num_items,
                         colname_user=colname_user, colname_item=colname_item,
                         colname_outcome=colname_outcome, colname_prediction=colname_prediction,
                         colname_treatment=colname_treatment, colname_propensity=colname_propensity)
        self.metric = metric
        self.capping_T = capping_T
        self.capping_C = capping_C
        self.with_IPS = with_IPS
        self.dim_factor = dim_factor
        self.rng = RandomState(seed=None)
        self.with_bias = with_bias
        self.coeff_T = coeff_T
        self.coeff_C = coeff_C
        self.learn_rate = learn_rate
        self.reg_bias = reg_factor
        self.reg_factor = reg_factor
        self.reg_bias_j = reg_factor
        self.reg_factor_j = reg_factor
        self.sd_init = sd_init
        self.only_treated = only_treated

        self.user_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_users, self.dim_factor))
        self.item_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_items, self.dim_factor))
        if self.with_bias:
            self.user_biases = np.zeros(self.num_users)
            self.item_biases = np.zeros(self.num_items)
            self.global_bias = 0.0

    def train(self, df, iter = 100):
        df_train = df.loc[df.loc[:, self.colname_outcome] > 0, :] # need only positive outcomes
        if self.only_treated: # train only with treated positive (DLTO)
            df_train = df_train.loc[df_train.loc[:, self.colname_treatment] > 0, :]

        if self.capping_T is not None:
            bool_cap = np.logical_and(df_train.loc[:, self.colname_propensity] < self.capping_T, df_train.loc[:, self.colname_treatment] == 1)
            if np.sum(bool_cap) > 0:
                df_train.loc[bool_cap, self.colname_propensity] = self.capping_T
        if self.capping_C is not None:      
            bool_cap = np.logical_and(df_train.loc[:, self.colname_propensity] > 1 - self.capping_C, df_train.loc[:, self.colname_treatment] == 0)
            if np.sum(bool_cap) > 0:
                df_train.loc[bool_cap, self.colname_propensity] = 1 - self.capping_C

        if self.with_IPS: # point estimate of individual treatment effect (ITE) <- for binary outcome abs(ITE) = IPS
            df_train.loc[:, 'ITE'] =  df_train.loc[:, self.colname_treatment] * df_train.loc[:, self.colname_outcome]/df_train.loc[:, self.colname_propensity] - \
                                      (1 - df_train.loc[:, self.colname_treatment]) * df_train.loc[:, self.colname_outcome]/(1 - df_train.loc[:, self.colname_propensity])
            z_y_1 = df_train.loc[:, self.colname_treatment] * df_train.loc[:, self.colname_outcome]
            z_y_1 = z_y_1.values
            z_y_0 = (1 - df_train.loc[:, self.colname_treatment]) * df_train.loc[:, self.colname_outcome]
            z_y_0 = z_y_0.values

        else:
            df_train.loc[:, 'ITE'] =  df_train.loc[:, self.colname_treatment] * df_train.loc[:, self.colname_outcome]  - \
                                      (1 - df_train.loc[:, self.colname_treatment]) * df_train.loc[:, self.colname_outcome]

        err = 0
        current_iter = 0
        while True:
            df_train = df_train.sample(frac=1)
            users = df_train.loc[:, self.colname_user].values
            items = df_train.loc[:, self.colname_item].values
            ITE = df_train.loc[:, 'ITE'].values

            if self.metric in ['AR_logi', 'AR_sig', 'AR_hinge']:
                for n in np.arange(len(df_train)):

                    u = users[n]
                    i = items[n]

                    while True:
                        j = random.randrange(self.num_items)
                        if i != j:
                            break

                    u_factor = self.user_factors[u, :]
                    i_factor = self.item_factors[i, :]
                    j_factor = self.item_factors[j, :]

                    diff_rating = np.sum(u_factor * (i_factor - j_factor))
                    if self.with_bias:
                        diff_rating += (self.item_biases[i] - self.item_biases[j])

                    if self.metric == 'AR_logi':
                        if ITE[n] >= 0:
                            coeff = ITE[n] * self.coeff_T * self.func_sigmoid(-self.coeff_T * diff_rating) # Z=1, Y=1
                            const_value = z_y_1[n] * self.coeff_T * self.func_sigmoid(-self.coeff_T * diff_rating)
                        else:
                            coeff = ITE[n] * self.coeff_C * self.func_sigmoid(self.coeff_C * diff_rating) # Z=0, Y=1
                            const_value = z_y_0[n] * self.coeff_C * self.func_sigmoid(self.coeff_C * diff_rating)

                    elif self.metric == 'AR_sig':
                        if ITE[n] >= 0:
                            coeff = ITE[n] * self.coeff_T * self.func_sigmoid(self.coeff_T * diff_rating) * self.func_sigmoid(-self.coeff_T * diff_rating)
                        else:
                            coeff = ITE[n] * self.coeff_C * self.func_sigmoid(self.coeff_C * diff_rating) * self.func_sigmoid(-self.coeff_C * diff_rating)

                    elif self.metric == 'AR_hinge':
                        if ITE[n] >= 0:
                            if self.coeff_T > 0 and diff_rating < 1.0/self.coeff_T:
                                coeff = ITE[n] * self.coeff_T 
                            else:
                                coeff = 0.0
                        else:
                            if self.coeff_C > 0 and diff_rating > -1.0/self.coeff_C:
                                coeff = ITE[n] * self.coeff_C
                            else:
                                coeff = 0.0

                    err += np.abs(coeff)

                    self.user_factors[u, :] += \
                        self.learn_rate * (coeff * (i_factor - j_factor) - self.reg_factor * u_factor)
                    self.item_factors[i, :] += \
                        self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor)
                    self.item_factors[j, :] += \
                        self.learn_rate * (-coeff * u_factor - self.reg_factor_j * j_factor)

                    if self.with_bias:
                        self.item_biases[i] += \
                            self.learn_rate * (coeff - self.reg_bias * self.item_biases[i])
                        self.item_biases[j] += \
                            self.learn_rate * (-coeff - self.reg_bias_j * self.item_biases[j])

                    current_iter += 1
                    if current_iter % 100000 == 0:
                        print(str(current_iter)+"/"+str(iter))
                    if current_iter >= iter:
                        return err/iter

    def predict(self, df):
        users = df[self.colname_user].values
        items = df[self.colname_item].values
        pred = np.zeros(len(df))
        for n in np.arange(len(df)):
            pred[n] = np.inner(self.user_factors[users[n], :], self.item_factors[items[n], :])
            if self.with_bias:
                pred[n] += self.item_biases[items[n]]
                pred[n] += self.user_biases[users[n]]
                pred[n] += self.global_bias

        # pred = 1 / (1 + np.exp(-pred))
        return pred


class RandomBase(Recommender):

    def __init__(self, num_users, num_items,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred'):
        super().__init__(num_users=num_users, num_items=num_items,
                         colname_user=colname_user, colname_item=colname_item,
                         colname_outcome=colname_outcome, colname_prediction=colname_prediction)

    def train(self, df, iter = 1):
        pass

    def predict(self, df):
        return np.random.rand(df.shape[0])


class RandomBase(Recommender):

    def __init__(self, num_users, num_items,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred'):
        super().__init__(num_users=num_users, num_items=num_items,
                         colname_user=colname_user, colname_item=colname_item,
                         colname_outcome=colname_outcome, colname_prediction=colname_prediction)

    def train(self, df, iter = 1):
        pass

    def predict(self, df):
        return np.random.rand(df.shape[0])

class MF(Recommender):
    def __init__(self, num_users, num_items,
                 metric='RMSE',
                 dim_factor=200, with_bias=False,
                 learn_rate = 0.01, reg_factor = 0.01, reg_bias = 0.01, sd_init = 0.1,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred',
                 colname_treatment='treated', colname_propensity='propensity'):
        super().__init__(num_users=num_users, num_items=num_items,
                         colname_user=colname_user, colname_item=colname_item,
                         colname_outcome=colname_outcome, colname_prediction=colname_prediction,
                         colname_treatment=colname_treatment, colname_propensity=colname_propensity)
        self.metric = metric
        self.dim_factor = dim_factor
        self.rng = RandomState(seed=None)
        self.with_bias = with_bias

        self.learn_rate = learn_rate
        self.reg_bias = reg_bias
        self.reg_factor = reg_factor
        self.sd_init = sd_init

        self.flag_prepared = False

        self.user_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_users, self.dim_factor))
        self.item_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_items, self.dim_factor))
        if self.with_bias:
            self.user_biases = np.zeros(self.num_users)
            self.item_biases = np.zeros(self.num_items)
            self.global_bias = 0.0

    def prepare_dictionary(self, df, colname_time='idx_time'):
        print("start prepare dictionary")
        self.colname_time = colname_time
        self.num_times = np.max(df.loc[:, self.colname_time]) + 1
        self.dict_positive_sets = dict()
    
        df_posi = df.loc[df.loc[:, self.colname_outcome] > 0]
    
        for t in np.arange(self.num_times):
            df_t = df_posi.loc[df_posi.loc[:, self.colname_time] == t]
            self.dict_positive_sets[t] = dict()
            for u in np.unique(df_t.loc[:, self.colname_user]):
                self.dict_positive_sets[t][u] = \
                    np.unique(df_t.loc[df_t.loc[:, self.colname_user] == u, self.colname_item].values)
    
        self.flag_prepared = True
        print("prepared dictionary!")


    def train(self, df, iter = 100):

        # by default, rating prediction
        # outcome = rating
        df_train = df.loc[~np.isnan(df.loc[:, self.colname_outcome]), :]

        # # in case of binary implicit feedback
        if self.metric == 'logloss':
            df_train = df.loc[df.loc[:, self.colname_outcome] > 0, :]  # need only positive outcomes
            if not self.flag_prepared: # prepare dictionary
                self.prepare_dictionary(df)
        else:
            df_train = df.loc[~np.isnan(df.loc[:, self.colname_outcome]), :]

        err = 0
        current_iter = 0
        while True:
            if self.metric == 'RMSE':
                df_train = df_train.sample(frac=1)
                users = df_train.loc[:, self.colname_user].values
                items = df_train.loc[:, self.colname_item].values
                outcomes = df_train.loc[:, self.colname_outcome].values

                for n in np.arange(len(df_train)):
                    u = users[n]
                    i = items[n]
                    r = outcomes[n]

                    u_factor = self.user_factors[u, :]
                    i_factor = self.item_factors[i, :]

                    rating = np.sum(u_factor * i_factor)
                    if self.with_bias:
                        rating += self.item_biases[i] + self.user_biases[u] + self.global_bias

                    coeff = r - rating
                    err += np.abs(coeff)

                    self.user_factors[u, :] += \
                        self.learn_rate * (coeff * i_factor - self.reg_factor * u_factor)
                    self.item_factors[i, :] += \
                        self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor)

                    if self.with_bias:
                        self.item_biases[i] += \
                            self.learn_rate * (coeff - self.reg_bias * self.item_biases[i])
                        self.user_biases[u] += \
                            self.learn_rate * (coeff - self.reg_bias * self.user_biases[u])
                        self.global_bias += \
                            self.learn_rate * (coeff)

                    current_iter += 1
                    if current_iter >= iter:
                        return err / iter

            elif self.metric == 'logloss': # logistic matrix factorization
                df_train = df_train.sample(frac=1)
                users = df_train.loc[:, self.colname_user].values
                items = df_train.loc[:, self.colname_item].values
                outcomes = df_train.loc[:, self.colname_outcome].values

                for n in np.arange(len(df_train)):
                    u = users[n]
                    i = items[n]
                    r = outcomes[n]

                    u_factor = self.user_factors[u, :]
                    i_factor = self.item_factors[i, :]

                    rating = np.sum(u_factor * i_factor)
                    if self.with_bias:
                        rating += self.item_biases[i] + self.user_biases[u] + self.global_bias

                    if r > 0:
                        coeff = self.func_sigmoid(-rating)
                    else:
                        coeff = - self.func_sigmoid(rating)

                    self.user_factors[u, :] += \
                        self.learn_rate * (coeff * i_factor - self.reg_factor * u_factor)
                    self.item_factors[i, :] += \
                        self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor)

                    if self.with_bias:
                        self.item_biases[i] += \
                            self.learn_rate * (coeff - self.reg_bias * self.item_biases[i])
                        self.user_biases[u] += \
                            self.learn_rate * (coeff - self.reg_bias * self.user_biases[u])
                        self.global_bias += \
                            self.learn_rate * (coeff)

                    current_iter += 1
                    if current_iter >= iter:
                        return err / iter



    def predict(self, df):
        users = df[self.colname_user].values
        items = df[self.colname_item].values
        pred = np.zeros(len(df))
        for n in np.arange(len(df)):
            pred[n] = np.inner(self.user_factors[users[n], :], self.item_factors[items[n], :])
            if self.with_bias:
                pred[n] += self.item_biases[items[n]]
                pred[n] += self.user_biases[users[n]]
                pred[n] += self.global_bias

        if self.metric == 'logloss':
            pred = 1 / (1 + np.exp(-pred))
        return pred



class CausalNeighborBase(Recommender):

    def __init__(self, num_users, num_items,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred',
                 measure_simil='cosine', way_simil='treatment',
                 way_neighbor='user', num_neighbor=3000,
                 way_self='exclude',
                 weight_treated_outcome=0.5,
                 shrinkage_T=10.0, shrinkage_C=10.0,
                 scale_similarity=0.33, normalize_similarity=False):

        super().__init__(num_users=num_users, num_items=num_items,
                         colname_user=colname_user, colname_item=colname_item,
                         colname_outcome=colname_outcome, colname_prediction=colname_prediction)
        self.measure_simil = measure_simil
        self.way_simil = way_simil
        self.way_neighbor = way_neighbor
        self.num_neighbor = num_neighbor
        self.scale_similarity = scale_similarity
        self.normalize_similarity = normalize_similarity
        self.weight_treated_outcome = weight_treated_outcome
        self.shrinkage_T = shrinkage_T
        self.shrinkage_C = shrinkage_C
        self.way_self = way_self # exclude/include/only


    def simil(self, set1, set2, measure_simil):
        if measure_simil == "jaccard":
            return self.simil_jaccard(set1, set2)
        elif measure_simil == "cosine":
            return self.simil_cosine(set1, set2)

    def train(self, df, iter=1):
        df_posi = df.loc[df.loc[:, self.colname_outcome] > 0]
        print("len(df_posi): {}".format(len(df_posi)))

        dict_items2users = dict() # map an item to users who consumed the item
        for i in np.arange(self.num_items):
            dict_items2users[i] = np.unique(df_posi.loc[df_posi.loc[:, self.colname_item] == i, self.colname_user].values)
        self.dict_items2users = dict_items2users
        print("prepared dict_items2users")

        dict_users2items = dict()  # map an user to items which are consumed by the user
        for u in np.arange(self.num_users):
            dict_users2items[u] = np.unique(df_posi.loc[df_posi.loc[:, self.colname_user] == u, self.colname_item].values)
        self.dict_users2items = dict_users2items
        print("prepared dict_users2items")

        df_treated = df.loc[df.loc[:, self.colname_treatment] > 0]  # calc similarity by treatment assignment
        print("len(df_treated): {}".format(len(df_treated)))

        dict_items2users_treated = dict() # map an item to users who get treatment of the item
        for i in np.arange(self.num_items):
            dict_items2users_treated[i] = np.unique(df_treated.loc[df_treated.loc[:, self.colname_item] == i, self.colname_user].values)
        self.dict_items2users_treated = dict_items2users_treated
        print("prepared dict_items2users_treated")

        dict_users2items_treated = dict()  # map an user to items which are treated to the user
        for u in np.arange(self.num_users):
            dict_users2items_treated[u] = np.unique(df_treated.loc[df_treated.loc[:, self.colname_user] == u, self.colname_item].values)
        self.dict_users2items_treated = dict_users2items_treated
        print("prepared dict_users2items_treated")

        if self.way_simil == 'treatment':
            if self.way_neighbor == 'user':
                dict_simil_users = {}
                sum_simil = np.zeros(self.num_users)
                for u1 in np.arange(self.num_users):
                    if u1 % round(self.num_users/10) == 0:
                        print("progress of similarity computation: {:.1f} %".format(100 * u1/self.num_users))

                    items_u1 = self.dict_users2items_treated[u1]
                    dict_neighbor = {}
                    if len(items_u1) > 0:
                        cand_u2 = np.unique(df_treated.loc[np.isin(df_treated.loc[:, self.colname_item], items_u1), self.colname_user].values)
                        for u2 in cand_u2:
                            if u2 != u1:
                                items_u2 = self.dict_users2items_treated[u2]
                                dict_neighbor[u2] = self.simil(items_u1, items_u2, self.measure_simil)

                        # print("len(dict_neighbor): {}".format(len(dict_neighbor)))
                        if len(dict_neighbor) > self.num_neighbor:
                            dict_neighbor = self.trim_neighbor(dict_neighbor, self.num_neighbor)
                        if self.scale_similarity != 1.0:
                            dict_neighbor = self.rescale_neighbor(dict_neighbor, self.scale_similarity)
                        if self.normalize_similarity:
                            dict_neighbor = self.normalize_neighbor(dict_neighbor)
                        dict_simil_users[u1] = dict_neighbor
                        sum_simil[u1] = np.sum(np.array(list(dict_neighbor.values())))
                    else:
                        dict_simil_users[u1] = dict_neighbor
                self.dict_simil_users = dict_simil_users
                self.sum_simil = sum_simil

            elif self.way_neighbor == 'item':
                dict_simil_items = {}
                sum_simil = np.zeros(self.num_items)
                for i1 in np.arange(self.num_items):
                    if i1 % round(self.num_items/10) == 0:
                        print("progress of similarity computation: {:.1f} %".format(100 * i1 / self.num_items))

                    users_i1 = self.dict_items2users_treated[i1]
                    dict_neighbor = {}
                    if len(users_i1) > 0:
                        cand_i2 = np.unique(
                            df_treated.loc[np.isin(df_treated.loc[:, self.colname_user], users_i1), self.colname_item].values)
                        for i2 in cand_i2:
                            if i2 != i1:
                                users_i2 = self.dict_items2users_treated[i2]
                                dict_neighbor[i2] = self.simil(users_i1, users_i2, self.measure_simil)

                        if len(dict_neighbor) > self.num_neighbor:
                            dict_neighbor = self.trim_neighbor(dict_neighbor, self.num_neighbor)
                        if self.scale_similarity != 1.0:
                            dict_neighbor = self.rescale_neighbor(dict_neighbor, self.scale_similarity)
                        if self.normalize_similarity:
                            dict_neighbor = self.normalize_neighbor(dict_neighbor)
                        dict_simil_items[i1] = dict_neighbor
                        sum_simil[i1] = np.sum(np.array(list(dict_neighbor.values())))
                    else:
                        dict_simil_items[i1] = dict_neighbor
                self.dict_simil_items = dict_simil_items
                self.sum_simil = sum_simil
        else:
            if self.way_neighbor == 'user':
                dict_simil_users = {}
                sum_simil = np.zeros(self.num_users)
                for u1 in np.arange(self.num_users):
                    if u1 % round(self.num_users/10) == 0:
                        print("progress of similarity computation: {:.1f} %".format(100 * u1 / self.num_users))

                    items_u1 = self.dict_users2items[u1]
                    dict_neighbor = {}
                    if len(items_u1) > 0:
                        cand_u2 = np.unique(
                            df_posi.loc[np.isin(df_posi.loc[:, self.colname_item], items_u1), self.colname_user].values)
                        for u2 in cand_u2:
                            if u2 != u1:
                                items_u2 = self.dict_users2items[u2]
                                dict_neighbor[u2] = self.simil(items_u1, items_u2, self.measure_simil)

                        if len(dict_neighbor) > self.num_neighbor:
                            dict_neighbor = self.trim_neighbor(dict_neighbor, self.num_neighbor)
                        if self.scale_similarity != 1.0:
                            dict_neighbor = self.rescale_neighbor(dict_neighbor, self.scale_similarity)
                        if self.normalize_similarity:
                            dict_neighbor = self.normalize_neighbor(dict_neighbor)
                        dict_simil_users[u1] = dict_neighbor
                        sum_simil[u1] = np.sum(np.array(list(dict_neighbor.values())))
                    else:
                        dict_simil_users[u1] = dict_neighbor
                self.dict_simil_users = dict_simil_users
                self.sum_simil = sum_simil

            elif self.way_neighbor == 'item':
                dict_simil_items = {}
                sum_simil = np.zeros(self.num_items)
                for i1 in np.arange(self.num_items):
                    if i1 % round(self.num_items/10) == 0:
                        print("progress of similarity computation: {:.1f} %".format(100 * i1 / self.num_items))

                    users_i1 = self.dict_items2users[i1]
                    dict_neighbor = {}
                    if len(users_i1) > 0:
                        cand_i2 = np.unique(
                            df_posi.loc[np.isin(df_posi.loc[:, self.colname_user], users_i1), self.colname_item].values)
                        for i2 in cand_i2:
                            if i2 != i1:
                                users_i2 = self.dict_items2users[i2]
                                dict_neighbor[i2] = self.simil(users_i1, users_i2, self.measure_simil)

                        if len(dict_neighbor) > self.num_neighbor:
                            dict_neighbor = self.trim_neighbor(dict_neighbor, self.num_neighbor)
                        if self.scale_similarity != 1.0:
                            dict_neighbor = self.rescale_neighbor(dict_neighbor, self.scale_similarity)
                        if self.normalize_similarity:
                            dict_neighbor = self.normalize_neighbor(dict_neighbor)
                        dict_simil_items[i1] = dict_neighbor
                        sum_simil[i1] = np.sum(np.array(list(dict_neighbor.values())))
                    else:
                        dict_simil_items[i1] = dict_neighbor
                self.dict_simil_items = dict_simil_items
                self.sum_simil = sum_simil


    def trim_neighbor(self, dict_neighbor, num_neighbor):
        return dict(sorted(dict_neighbor.items(), key=lambda x:x[1], reverse = True)[:num_neighbor])

    def normalize_neighbor(self, dict_neighbor):
        sum_simil = 0.0
        for v in dict_neighbor.values():
            sum_simil += v
        for k, v in dict_neighbor.items():
            dict_neighbor[k] = v/sum_simil
        return dict_neighbor

    def rescale_neighbor(self, dict_neighbor, scaling_similarity=1.0):
        for k, v in dict_neighbor.items():
            dict_neighbor[k] = np.power(v, scaling_similarity)
        return dict_neighbor


    def predict(self, df):
        users = df[self.colname_user].values
        items = df[self.colname_item].values
        pred = np.zeros(len(df))
        if self.way_neighbor == 'user':
            for n in np.arange(len(df)):
                u1 = users[n]
                simil_users = np.fromiter(self.dict_simil_users[u1].keys(), dtype=int)
                i_users_posi = self.dict_items2users[items[n]]  # users who consumed i=items[n]
                i_users_treated = self.dict_items2users_treated[items[n]]  # users who are treated i=items[n]
                if n % round(len(df)/10) == 0:
                    print("progress of prediction computation: {:.1f} %".format(100 * n / len(df)))
                    # print("simil_users")
                    # print(simil_users)
                    # print(type(simil_users))
                    # print(np.any(np.isin(simil_users, i_users_posi)))

                # initialize for this u1-i pair
                value_T = 0.0
                denom_T = 0.0
                value_C = 0.0
                denom_C = 0.0

                if np.any(np.isin(simil_users, i_users_posi)):
                    simil_users = simil_users[np.isin(simil_users, np.unique(np.append(i_users_treated,i_users_posi)))]
                    for u2 in simil_users:
                        if u2 in i_users_treated:
                            denom_T += self.dict_simil_users[u1][u2]
                            if u2 in i_users_posi:
                                value_T += self.dict_simil_users[u1][u2]
                        else:
                            value_C += self.dict_simil_users[u1][u2]
                            # denom_C += self.dict_simil_users[u1][u2]
                            # if u2 in i_users_posi:
                            #     value_C += self.dict_simil_users[u1][u2]
                    denom_C = self.sum_simil[u1] - denom_T # denom_T + denom_C = sum_simil

                if self.way_self == 'include': # add data of self u-i
                    if u1 in i_users_treated:
                        denom_T += 1.0
                        if u1 in i_users_posi:
                            value_T += 1.0
                    else:
                        denom_C += 1.0
                        if u1 in i_users_posi:
                            value_C += 1.0

                if self.way_self == 'only': # force data to self u-i
                    if u1 in i_users_treated:
                        denom_T = 1.0
                        if u1 in i_users_posi:
                            value_T = 1.0
                        else:
                            value_T = 0.0
                    else:
                        denom_C = 1.0
                        if u1 in i_users_posi:
                            value_C = 1.0
                        else:
                            value_C = 0.0

                if value_T > 0:
                    pred[n] += 2 * self.weight_treated_outcome * value_T / (self.shrinkage_T + denom_T)
                if value_C > 0:
                    pred[n] -= 2 * (1 - self.weight_treated_outcome) * value_C / (self.shrinkage_C + denom_C)
            print(pred[:5])
            print(np.mean(pred))
            print(np.max(pred))
            print(np.min(pred))

        elif self.way_neighbor == 'item':
            for n in np.arange(len(df)):
                i1 = items[n]
                simil_items = np.fromiter(self.dict_simil_items[i1].keys(), dtype=int)
                u_items_posi = self.dict_users2items[users[n]]  # items that is consumed by u=users[n]
                u_items_treated = self.dict_users2items_treated[users[n]] # items that is treated for u=users[n]
                if n % round(len(df)/10) == 0:
                    print("progress of prediction computation: {:.1f} %".format(100 * n / len(df)))

                # initialize for this u-i1 pair
                value_T = 0.0
                denom_T = 0.0
                value_C = 0.0
                denom_C = 0.0

                if np.any(np.isin(simil_items, u_items_posi)):
                    simil_items = simil_items[np.isin(simil_items, np.unique(np.append(u_items_posi, u_items_treated)))]
                    for i2 in simil_items:
                        if i2 in u_items_treated: # we assume that treated items are less than untreated items
                            denom_T += self.dict_simil_items[i1][i2]
                            if i2 in u_items_posi:
                                value_T += self.dict_simil_items[i1][i2]
                        else:
                            value_C += self.dict_simil_items[i1][i2]
                            # denom_C += self.dict_simil_items[i1][i2]
                            # if i2 in u_items_posi:
                            #     value_C += self.dict_simil_items[i1][i2]
                    denom_C = self.sum_simil[i1] - denom_T  # denom_T + denom_C = sum_simil

                if self.way_self == 'include': # add data of self u-i
                    if i1 in u_items_treated:
                        denom_T += 1.0
                        if i1 in u_items_posi:
                            value_T += 1.0
                    else:
                        denom_C += 1.0
                        if i1 in u_items_posi:
                            value_C += 1.0

                if self.way_self == 'only': # force data to self u-i
                    if i1 in u_items_treated:
                        denom_T = 1.0
                        if i1 in u_items_posi:
                            value_T = 1.0
                        else:
                            value_T = 0.0
                    else:
                        denom_C = 1.0
                        if i1 in u_items_posi:
                            value_C = 1.0
                        else:
                            value_C = 0.0

                if value_T > 0:
                    pred[n] += 2 * self.weight_treated_outcome * value_T / (self.shrinkage_T + denom_T)
                if value_C > 0:
                    pred[n] -= 2 * (1 - self.weight_treated_outcome) * value_C / (self.shrinkage_C + denom_C)

        return pred


    def simil_jaccard(self, x, y):
        return len(np.intersect1d(x, y))/len(np.union1d(x, y))

    def simil_cosine(self, x, y):
        return len(np.intersect1d(x, y))/np.sqrt(len(x)*len(y))


if __name__ == "__main__":
    train_df, vali_df, test_df, num_users, num_items = prepare_data()
    recommender = DLMF(num_users, num_items, capping_T=0.3, capping_C=0.3, learn_rate=0.01, reg_factor=0.01)
    err = recommender.train(train_df,iter=1)
    test_df = test_df[test_df["idx_time"] == np.random.randint(10)]
    test_df["pred"] = recommender.predict(test_df)
    print(test_df["pred"][:10])
    print(np.mean(test_df["pred"].to_numpy()), np.std(test_df["pred"].to_numpy()))
    evaluator = Evaluator()
    print("CP@10:", evaluator.evaluate(test_df, 'CPrec', 10))
    print("CP@100:", evaluator.evaluate(test_df, 'CPrec', 100))
    print("CDCG:", evaluator.evaluate(test_df, 'CDCG', 100000))