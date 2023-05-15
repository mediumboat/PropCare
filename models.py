from abc import ABC
import tensorflow as tf
import sys
from tensorflow import keras
from tensorflow.keras.losses import MSE, binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Flatten, Dense, BatchNormalization
import tensorflow_probability as tfp
from utils import squared_dist, find_k_smallest


class Causal_Model(Model, ABC):
    def __init__(self, num_users, num_items, flags, user_embs, item_embs, item_popularity,
                 **kwargs):
        super(Causal_Model, self).__init__(**kwargs)
        self.item_popularity = tf.cast(tf.squeeze(item_popularity), tf.float32)
        self.estimator_layer_units = flags.estimator_layer_units
        self.click_layer_units = flags.click_layer_units
        self.emb_layer_units = flags.embedding_layer_units
        self.lambda_1 = flags.lambda_1
        self.dims = flags.dimension
        self.p_weight = flags.p_weight
        self.pro_weight = flags.pro_weight
        self.rele_weight = flags.rele_weight
                # _____________Initialize Embedding Layer____________________________________________
        self.norm_layer = tf.keras.constraints.non_neg()
        if user_embs is None:
            self.mf_user_embedding = Embedding(input_dim=num_users, output_dim=flags.dimension,
                                               name='mf_user_embedding', input_length=1, trainable=False,
                                               embeddings_regularizer="l2")
        else:
            self.mf_user_embedding = Embedding(input_dim=num_users, output_dim=flags.dimension,
                                               name='mf_user_embedding', input_length=1, weights=[user_embs],
                                               trainable=False, embeddings_regularizer="l2")
        if item_embs is None:
            self.mf_item_embedding = Embedding(input_dim=num_items, output_dim=flags.dimension,
                                               name='mf_item_embedding', input_length=1, trainable=False,
                                               embeddings_regularizer="l2")
        else:
            self.mf_item_embedding = Embedding(input_dim=num_items, output_dim=flags.dimension,
                                               name='mf_item_embedding', input_length=1, weights=[item_embs],
                                               trainable=False, embeddings_regularizer="l2")
        # _____________Initialize shared embedding Layers____________________________________________
        self.flatten_layers = Flatten()
        self.emb_layers = []
        for i, unit in enumerate(flags.embedding_layer_units):
            self.emb_layers.append(
                Dense(unit, activation=tf.keras.layers.LeakyReLU(), name="emb_{}".format(i), kernel_initializer='he_normal', trainable=True))

        # _____________Initialize propensity estimator and relevance estimator Layer____________________
        self.propensity_layers = []
        self.relevance_layers = []
        self.propensity_bn_layers = []
        self.relevance_bn_layers = []
        self.film_alpha_propensity = []
        self.film_beta_propensity = []
        self.film_alpha_relevance = []
        self.film_beta_relevance = []
        self.exp_weight = tf.Variable(1.0, trainable=True)
        for i, unit in enumerate(flags.estimator_layer_units):
            self.propensity_layers.append(
                Dense(unit, activation=tf.keras.layers.LeakyReLU(), name='propensity_{}'.format(i), kernel_regularizer="l2",
                      kernel_initializer="he_normal", trainable=True))
            self.relevance_layers.append(
                Dense(unit, activation=tf.keras.layers.LeakyReLU(), name='relevance_{}'.format(i), kernel_regularizer="l2",
                      kernel_initializer="he_normal", trainable=True))
        self.propensity_Prediction_layer = Dense(1, activation='sigmoid',
                                                 name="propensity_prediction", kernel_regularizer="l2",
                                                 kernel_initializer="he_normal", trainable=True)
        self.relevance_Prediction_layer = Dense(1, activation='sigmoid',
                                                name="relevance_prediction", kernel_regularizer="l2",
                                                kernel_initializer="he_normal", trainable=True)
        # Define optimizers
        self.kl = tf.keras.losses.KLDivergence()
        self.target_dist = tfp.distributions.Beta(0.2, 1.0)
        self.estimator_optimizer = keras.optimizers.SGD(keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=0.01, first_decay_steps=2000))
        self.click_loss_tracker = keras.metrics.Mean(name="click_loss")
        self.estimator_loss_tracker = keras.metrics.Mean(name="estimator_loss")
        self.reg_loss_tracker = keras.metrics.Mean(name="reg_loss")

    @tf.function()
    def call(self, inputs, training=None, **kwargs):
        user_input, item_input = inputs
        mf_user_latent = self.flatten_layers(self.norm_layer(self.mf_user_embedding(user_input)))
        mf_item_latent = self.flatten_layers(self.norm_layer(self.mf_item_embedding(item_input)))
        mf_vector = tf.concat((mf_user_latent, mf_item_latent), axis=1)
        for i, unit in enumerate(self.emb_layer_units):
            emb_layer = self.emb_layers[i]
            mf_vector = emb_layer(mf_vector)
        propensity_vector = mf_vector
        relevance_vector = mf_vector
        film_reg_loss = 0.0
        for i, unit in enumerate(self.estimator_layer_units):
            propensity_layer = self.propensity_layers[i]
            propensity_vector = propensity_layer(propensity_vector)
            relevance_layer = self.relevance_layers[i]
            relevance_vector = relevance_layer(relevance_vector)
        propensity = self.propensity_Prediction_layer(propensity_vector)
        relevance = self.relevance_Prediction_layer(relevance_vector)
        propensity = tf.reshape(propensity, [-1, 1])
        relevance = tf.reshape(relevance, [-1, 1])
        propensity = tf.clip_by_value(propensity, 0.0001, 0.9999)
        relevance = tf.clip_by_value(relevance, 0.0001, 0.9999)
        click = tf.multiply(propensity, relevance)
        return click, propensity, relevance, film_reg_loss

    @tf.function()
    def propensity_train(self, data):
        user, item_i, item_j, y_true = data
        y_true = tf.reshape(y_true, [-1, 1]) 
        user = tf.reshape(user, [-1, 1])
        item_i = tf.reshape(item_i, [-1, 1])
        item_j = tf.reshape(item_j, [-1, 1])
        with tf.GradientTape() as tape2:
            y_i, p_i, r_i, film_reg_loss_1 = self((user, item_i), training=True)
            y_j, p_j, r_j, film_reg_loss_2 = self((user, item_j), training=True)
            loss_click = tf.reduce_mean(binary_crossentropy(y_true=y_true, y_pred=y_i))
            pop_signs = tf.sign(tf.gather(
                self.item_popularity, tf.squeeze(item_i)) - tf.gather(
                self.item_popularity, tf.squeeze(item_j)))
            pop_signs = tf.reshape(pop_signs, [-1, 1])
            pop_signs.shape.assert_is_compatible_with(y_i.shape)
            p_diff = tf.multiply(pop_signs, (p_i - p_j))
            r_diff = tf.multiply(pop_signs, (r_j - r_i))
            y_diff = tf.multiply(pop_signs, (y_i - y_j))
            weights_loss = tf.exp( - self.exp_weight * tf.square(y_diff))
            weights_loss = weights_loss / tf.math.reduce_max(weights_loss)
            loss_pair =  tf.math.log(self.pro_weight * p_diff +  self.rele_weight * r_diff)
            weights_loss.shape.assert_is_compatible_with(loss_pair.shape)
            loss_pair = tf.multiply(weights_loss, loss_pair)
            loss_pair = - tf.reduce_mean(loss_pair)
            target_samples_i = tf.stop_gradient(self.target_dist.sample(tf.shape(p_i)))
            target_samples_j = tf.stop_gradient(self.target_dist.sample(tf.shape(p_j)))
            p_loss =  self.kl(tf.sort(target_samples_i, axis=0), tf.sort(p_i, axis=0)) + self.kl(tf.sort(target_samples_j, axis=0), tf.sort(p_j, axis=0))
            reg_loss = 0.0001 * (tf.add_n(self.losses) + film_reg_loss_1 + film_reg_loss_2) + self.p_weight * (p_loss)
            loss = self.lambda_1 * loss_pair + loss_click + reg_loss
        self.estimator_optimizer.minimize(loss, self.trainable_weights, tape=tape2) 
        self.estimator_loss_tracker.update_state(loss_pair)
        self.click_loss_tracker.update_state(loss_click)
        self.reg_loss_tracker.update_state(reg_loss)
        return {"click_loss": self.click_loss_tracker.result(),
                "estimator_loss": self.estimator_loss_tracker.result(),
                "reg_loss": self.reg_loss_tracker.result()}

 


if __name__ == "__main__":
    pass
