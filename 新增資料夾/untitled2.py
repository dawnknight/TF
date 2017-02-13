# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 01:37:15 2017

@author: Dawnknight
"""

import tensorflow as tf
import numpy as np
import os

import config
import utils


class DenoisingAutoencoder(object):
    """ Implementation of Denoising Autoencoders using TensorFlow.
    The interface of the class is sklearn-like.
    """

    def __getattr__(self, name):
        """
        Does absolutely nothing, only for pycharm to stop complaining about missing attributes
        This function is *only* called when python fail to find certain attribute, so it always raises exception
        """
        raise AttributeError("Attribute %s is not part of %s class" % (name, self.__class__.__name__))

    def __init__(self, **kwargs):
        """ Constructor.

        :type directory_name: string, default 'dae'
        :param directory_name: Optional directory name for store models, data and summaries
                (appended to models_dir, data_dir and summary_dir in the config)

        :type n_components: int
        :param n_components: number of hidden units.

        :type tied_weights: boolean, default True
        :param tied_weights: Whether to use tied weights

        :type enc_act_func: string, default 'tanh', ['sigmoid', 'tanh']
        :param enc_act_func: Activation function for the encoder.

        :type dec_act_func: string, default 'tanh', ['sigmoid', 'tanh', 'none']
        :param dec_act_func: Activation function for the decoder.

        :type loss_func: string, default 'mean_squared', ['cross_entropy', 'mean_squared']
        :param loss_func: Loss function.

        :type xavier_init: int, default 1
        :param xavier_init: Value of the constant for xavier weights initialization

        :type opt: string, default 'gradient_descent', ['gradient_descent', 'ada_grad', 'momentum']
        :param opt: Which tensorflow optimizer to use.

        :type learning_rate: float, default 0.01
        :param learning_rate: Initial learning rate.

        :type momentum: float, default 0.5
        :param momentum: 'Momentum parameter.

        :type corr_type: string, default 'none'
        :param corr_type: Type of input corruption. ["none", "masking", "salt_and_pepper"]

        :type corr_frac: float, default 0.0
        :param corr_frac: Fraction of the input to corrupt.

        :type verbose: int, default 0
        :param verbose: Level of verbosity. 0 - silent, 1 - print accuracy.

        :type n_iter: int, default 10
        :param n_iter: Number of epochs

        :type batch_size: int, default 10
        :param batch_size: Size of each mini-batch

        :type dataset: string, default 'mnist'
        :param dataset: Optional name for the dataset.

        :type seed: int, default -1
        :param seed: positive integer for seeding random generators. Ignored if < 0.

        :return: self.
        """

        prop_defaults = {
            'model_name': '',
            'n_components': 256,
            'directory_name': 'dae/',
            'enc_act_func': 'tanh',
            'dec_act_func': 'none',
            'loss_func': 'mean_squared',
            'n_iter': 10,
            'batch_size': 10,
            'dataset': 'mnist',
            'tied_weights': True,
            'xavier_init': 1,
            'opt': 'gradient_descent',
            'learning_rate': 0.01,
            'dropout': 0.5,
            'momentum': 0.5,
            'corr_type': 'none',
            'corr_frac': 0.,
            'verbose': 1,
            'seed': -1,
        }

        for (prop, default) in prop_defaults.iteritems():
            setattr(self, prop, kwargs.get(prop, default))

        if self.seed >= 0:
            np.random.seed(self.seed)
            tf.set_random_seed(self.seed)

        # Directories paths
        self.directory_name = self.directory_name + '/' if self.directory_name[-1] != '/' else self.directory_name

        self.models_dir = config.models_dir + self.directory_name
        self.data_dir = config.data_dir + self.directory_name
        self.summary_dir = config.summary_dir + self.directory_name

        for d in [self.models_dir, self.data_dir, self.summary_dir]:
            if not os.path.isdir(d):
                os.mkdir(d)

        if self.model_name == '':
            # Assign model complete name
            self.model_name = 'dae-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
                self.dataset, self.n_components, self.corr_type, self.corr_frac, self.n_iter,
                self.batch_size, self.learning_rate, self.tied_weights, self.loss_func, self.enc_act_func,
                self.dec_act_func, self.opt)

        # ########################### #
        # Computational graph nodes   #
        # ########################### #

        # Placeholders
        self.x = None  # model original input
        self.x_corr = None  # model corrupted input

        # Model parameters
        self.Wf_ = None
        self.Wg_ = None
        self.bh_ = None
        self.bv_ = None

        # Model values
        self.y = None  # encoding phase output
        self.z = None  # decoding phase output

        # Model traning and evaluation
        self.train_step = None
        self.cost = None

        # tensorflow objects
        self.sess = None
        self.saver = None

    def _create_graph(self, n_features):
        """ Creates the computational graph.

        :type n_features: int
        :param n_features: Number of features.

        :return: self
        """
        # ################################### #
        #   Computation Graph Specification   #
        # ################################### #

        # Symbolic variables
        self.x = tf.placeholder('float', [None, n_features], name='x-input')
        self.x_corr = tf.placeholder('float', [None, n_features], name='x-corr-input')

        self.keep_prob = tf.placeholder('float')

        # Biases
        self.bh_ = tf.Variable(tf.zeros([self.n_components]), name='hidden-bias')
        self.bv_ = tf.Variable(tf.zeros([n_features]), name='visible-bias')

        # Weights
        self.Wf_ = tf.Variable(utils.xavier_init(n_features, self.n_components, self.xavier_init), name='enc-w')

        if self.tied_weights:
            self.Wg_ = tf.transpose(self.Wf_)

        else:
            self.Wg_ = tf.Variable(utils.xavier_init(n_features, self.n_components, self.xavier_init), name='dec-w')

        # ############ #
        #   Encoding   #
        # ############ #
        with tf.name_scope("Wf_x_bh"):
            if self.enc_act_func == 'sigmoid':
                self.y = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(self.x_corr, self.Wf_) + self.bh_), self.keep_prob)

            elif self.enc_act_func == 'tanh':
                self.y = tf.nn.dropout(tf.nn.tanh(tf.matmul(self.x_corr, self.Wf_) + self.bh_), self.keep_prob)

            else:  # cannot be reached, just for completeness
                self.y = None

        # ############ #
        #   Decoding   #
        # ############ #
        with tf.name_scope("Wg_y_bv"):
            if self.dec_act_func == 'sigmoid':
                self.z = tf.nn.sigmoid(tf.matmul(self.y, self.Wg_) + self.bv_)

            elif self.dec_act_func == 'tanh':
                self.z = tf.nn.tanh(tf.matmul(self.y, self.Wg_) + self.bv_)

            elif self.dec_act_func == 'none':
                self.z = tf.matmul(self.y, self.Wg_) + self.bv_

            else:  # cannot be reached, just for completeness
                self.z = None

        # ############### #
        #   Summary Ops   #
        # ############### #
        # Add summary ops to collect data
        _ = tf.histogram_summary("enc_weights", self.Wf_)
        _ = tf.histogram_summary("hid_biases", self.bh_)
        _ = tf.histogram_summary("vis_biases", self.bv_)
        _ = tf.histogram_summary("y", self.y)
        _ = tf.histogram_summary("z", self.z)

        if not self.tied_weights:
            _ = tf.histogram_summary("dec_weights", self.Wg_)

        # ######## #
        #   Cost   #
        # ######## #
        with tf.name_scope("cost"):
            if self.loss_func == 'cross_entropy':
                self.cost = - tf.reduce_sum(self.x * tf.log(self.z))
                _ = tf.scalar_summary("cross_entropy", self.cost)

            elif self.loss_func == 'mean_squared':
                self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.x - self.z)))
                _ = tf.scalar_summary("mean_squared", self.cost)

            else:  # cannot be reached, just for completeness
                self.cost = None

        with tf.name_scope("train"):
            if self.opt == 'gradient_descent':
                self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

            elif self.opt == 'ada_grad':
                self.train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.cost)

            elif self.opt == 'momentum':
                self.train_step = tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(self.cost)

            else:  # cannot be reached, just for completeness
                self.train_step = None

    def fit(self, trX, vlX=None, restore_previous_model=False):
        """ Fit the model to the data.

        :type trX: array_like, shape (n_samples, n_features).
        :param trX: Training data.

        :type vlX: array_like, shape (n_validation_samples, n_features).
        :param vlX: optional, default None. Validation data.

        :return: self
        """
        n_features = trX.shape[1]

        self._create_graph(n_features)

        # Merge all the summaries
        merged = tf.merge_all_summaries()
        # Initialize variables
        init_op = tf.initialize_all_variables()
        # Add ops to save and restore all the variables
        self.saver = tf.train.Saver()

        with tf.Session() as self.sess:

            self.sess.run(init_op)

            if restore_previous_model:
                # Restore previous model
                self.saver.restore(self.sess, self.models_dir + self.model_name)
                # Change model name
                self.model_name += '-restored{}'.format(self.n_iter)

            # ################## #
            #   Training phase   #
            # ################## #

            v = np.round(self.corr_frac * n_features).astype(np.int)

            # Write the summaries to summary_dir
            writer = tf.train.SummaryWriter(self.summary_dir, self.sess.graph_def)

            for i in range(self.n_iter):

                # #################### #
                #   Input Corruption   #
                # #################### #

                if self.corr_type == 'masking':
                    x_corrupted = utils.masking_noise(trX, v)

                elif self.corr_type == 'salt_and_pepper':
                    x_corrupted = utils.salt_and_pepper_noise(trX, v)

                else:  # none, normal autoencoder
                    x_corrupted = trX

                # Randomly shuffle the input
                shuff = zip(trX, x_corrupted)
                np.random.shuffle(shuff)

                # # Divide dataset into mini-batches
                batches = [_ for _ in utils.gen_batches(shuff, self.batch_size)]

                # All the batches for each epoch
                for batch in batches:
                    x_batch, x_corr_batch = zip(*batch)
                    tr_feed = {self.x: x_batch, self.x_corr: x_corr_batch, self.keep_prob: self.dropout}
                    self.sess.run(self.train_step, feed_dict=tr_feed)

                # Record summary data
                if vlX is not None:
                    vl_feed = {self.x: vlX, self.x_corr: vlX, self.keep_prob: 1.}
                    result = self.sess.run([merged, self.cost], feed_dict=vl_feed)
                    summary_str = result[0]
                    err = result[1]

                    writer.add_summary(summary_str, i)

                    if self.verbose == 1:
                        print("Validation cost at step %s: %s" % (i, err))

            # Save trained model
            self.saver.save(self.sess, self.models_dir + self.model_name)

    def transform(self, data, name='train', save=False):
        """ Transform data according to the model.

        :type data: array_like
        :param data: Data to transform

        :type name: string, default 'train'
        :param name: Identifier for the data that is being encoded

        :type save: boolean, default 'False'
        :param save: If true, save data to disk

        :return: transformed data
        """

        with tf.Session() as self.sess:

            # Restore trained model
            self.saver.restore(self.sess, self.models_dir + self.model_name)

            # Return the output of the encoding layer
            encoded_data = self.y.eval({self.x_corr: data})

            if save:
                # Save transformation to output file
                np.save(self.data_dir + self.model_name + '-' + name, encoded_data)

            return encoded_data

    def load_model(self, shape, model_path):
        """ Restore a previously trained model from disk.

        :type shape: tuple
        :param shape: tuple(n_features, n_components)

        :type model_path: string
        :param model_path: path to the trained model

        :return: self, the trained model
        """
        self.n_components = shape[1]

        self._create_graph(shape[0])

        # Initialize variables
        init_op = tf.initialize_all_variables()

        # Add ops to save and restore all the variables
        self.saver = tf.train.Saver()

        with tf.Session() as self.sess:

            self.sess.run(init_op)

            # Restore previous model
            self.saver.restore(self.sess, model_path)

    def get_model_parameters(self):
        """ Return the model parameters in the form of numpy arrays.

        :return: model parameters
        """
        with tf.Session() as self.sess:

            # Restore trained model
            self.saver.restore(self.sess, self.models_dir + self.model_name)

            return {
                'enc_w': self.Wf_.eval(),
                'dec_w': self.Wg_.eval() if self.tied_weights else None,
                'enc_b': self.bh_.eval(),
                'dec_b': self.bv_.eval()
            }

    def get_weights_as_images(self, width, height, outdir='img/', max_images=10, model_path=None):
        """ Save the weights of this autoencoder as images, one image per hidden unit.
        Useful to visualize what the autoencoder has learned.

        :type width: int
        :param width: Width of the images

        :type height: int
        :param height: Height of the images

        :type outdir: string, default 'data/sdae/img'
        :param outdir: Output directory for the images. This path is appended to self.data_dir

        :type max_images: int, default 10
        :param max_images: Number of images to return.
        """
        assert max_images <= self.n_components

        outdir = self.data_dir + outdir

        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        with tf.Session() as self.sess:

            # Restore trained model
            if model_path is not None:
                self.saver.restore(self.sess, model_path)
            else:
                self.saver.restore(self.sess, self.models_dir + self.model_name)

            # Extract encoding weights as numpy array
            enc_weights = self.Wf_.eval()

            # Extract decoding weights as numpy arrays
            dec_weights = self.Wg_.eval() if not self.tied_weights else None

            perm = np.random.permutation(self.n_components)[:max_images]

            for p in perm:

                enc_w = np.array([i[p] for i in enc_weights])
                image_path = outdir + self.model_name + '-enc_weights_{}.png'.format(p)
                utils.gen_image(enc_w, width, height, image_path)

                if not self.tied_weights:
                    dec_w = np.array([i[p] for i in dec_weights])
                    image_path = outdir + self.model_name + '-dec_weights_{}.png'.format(p)
                    utils.gen_image(dec_w, width, height, image_path)