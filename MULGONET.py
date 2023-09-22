#构建模型
import keras
import numpy as np
from keras import regularizers
from keras.engine import Layer
# from keras import initializations
from keras.initializers import glorot_uniform, Initializer
from keras.layers import activations, initializers, constraints
# our layer will take input shape (nb_samples, 1)
from keras.regularizers import Regularizer
import tensorflow as tf

from keras.layers import Dense, Dropout, Activation, BatchNormalization, multiply
from keras.regularizers import l2
from keras import Input
from keras.engine import Model
from keras import backend as K
import random
random.seed(10290000)




class MUlGONET(Layer):
    def __init__(self, units, mapp=None, nonzero_ind=None, kernel_initializer='glorot_uniform', W_regularizer=None,
                 activation='tanh', use_bias=True, bias_initializer='zeros', bias_regularizer=None,
                 bias_constraint=None, **kwargs):

        self.units = units
        self.activation = activation
        self.mapp = mapp
        self.nonzero_ind = nonzero_ind
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(W_regularizer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activation_fn = activations.get(activation)
        super(MUlGONET, self).__init__(**kwargs)

    def build(self, input_shape):

        input_dim = input_shape[1]

        if not self.mapp is None:
            self.mapp = self.mapp.astype(np.float32)

        if self.nonzero_ind is None:
            nonzero_ind = np.array(np.nonzero(self.mapp)).T
            self.nonzero_ind = nonzero_ind

        self.kernel_shape = (input_dim, self.units)

        nonzero_count = self.nonzero_ind.shape[0]  # node and node  connection nunber

        self.kernel_vector = self.add_weight(name='kernel_vector',
                                             shape=(nonzero_count,),
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                             trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer
                                        )
        else:
            self.bias = None

        super(MUlGONET, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):


        tt = tf.scatter_nd(tf.constant(self.nonzero_ind, tf.int32), self.kernel_vector,
                           tf.constant(list(self.kernel_shape)))

        output = K.dot(inputs, tt)

        if self.use_bias:
            output = K.bias_add(output, self.bias)

        if self.activation_fn is not None:
            output = self.activation_fn(output)

        return output

    def get_config(self):
        config = {
            'units': self.units,
            'activation': self.activation,
            #             'kernel_shape': self.kernel_shape,
            'use_bias': self.use_bias,
            'nonzero_ind': np.array(self.nonzero_ind),

            'bias_initializer': initializers.serialize(self.bias_initializer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),

            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'W_regularizer': regularizers.serialize(self.kernel_regularizer),

        }
        base_config = super(MUlGONET, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):

        return (input_shape[0], self.units)





def create_model(inputss,gene_pathway_bp_dfss,Get_Node_relation, gene_pathway_mf_dfss,Get_Node_relation_mf,
                 bp_net=False, mf_net=False, optimizers='Adam'):

    # network input
    multi_input = Input(shape=(inputss.shape[1],), dtype='float32', name='input_multi')
    #     input_drop = keras.layers.Dropout(0.5)(multi_input)

    #  BP network
    if bp_net:

        bp_layer0 = MUlGONET(gene_pathway_bp_dfss.shape[0], mapp=gene_pathway_bp_dfss.values.T, W_regularizer=l2(0.0001),name='input_drop')(
            multi_input)
        bp_drop0 = keras.layers.Dropout(0.5)(bp_layer0)


        bp_layer1 = MUlGONET(Get_Node_relation[3].shape[1], mapp=Get_Node_relation[3].values, name='bp_layer1')(
            bp_drop0)
        bp_drop1 = keras.layers.Dropout(0.1)(bp_layer1)


        bp_layer2 = MUlGONET(Get_Node_relation[2].shape[1], mapp=Get_Node_relation[2].values, name='bp_layer2')(
            bp_drop1)
        bp_drop2 = keras.layers.Dropout(0.1)(bp_layer2)


        bp_layer3 = MUlGONET(Get_Node_relation[1].shape[1], mapp=Get_Node_relation[1].values, name='bp_layer3')(
            bp_drop2)
        bp_drop3 = keras.layers.Dropout(0.1)(bp_layer3)


        bp_layer4 = MUlGONET(Get_Node_relation[0].shape[1], mapp=Get_Node_relation[0].values, name='bp_layer4')(
            bp_drop3)

        finally_layer = keras.layers.Dense(1, activation='sigmoid')(bp_layer4)

    # MF network

    elif mf_net:

        mf_layer0 = MUlGONET(gene_pathway_mf_dfss.shape[0], mapp=gene_pathway_mf_dfss.values.T, W_regularizer=l2(0.0001),name='mf_layer0')(
            multi_input)
        mf_drop0 = keras.layers.Dropout(0.5)(mf_layer0)


        mf_layer1 = MUlGONET(Get_Node_relation_mf[3].shape[1], mapp=Get_Node_relation_mf[3].values, name='mf_layer1')(
            mf_drop0)
        mf_drop1 = keras.layers.Dropout(0.1)(mf_layer1)


        mf_layer2 = MUlGONET(Get_Node_relation_mf[2].shape[1], mapp=Get_Node_relation_mf[2].values, name='mf_layer2')(
            mf_drop1)
        mf_drop2 = keras.layers.Dropout(0.1)(mf_layer2)


        mf_layer3 = MUlGONET(Get_Node_relation_mf[1].shape[1], mapp=Get_Node_relation_mf[1].values, name='mf_layer3')(
            mf_drop2)
        mf_drop3 = keras.layers.Dropout(0.1)(mf_layer3)


        mf_layer4 = MUlGONET(Get_Node_relation_mf[0].shape[1], mapp=Get_Node_relation_mf[0].values, name='mf_layer4')(
            mf_drop3)

        finally_layer = keras.layers.Dense(1, activation='sigmoid')(mf_layer4)

    else:
        #  BP+MF

        # BP
        bp_layer0 = MUlGONET(gene_pathway_bp_dfss.shape[0], mapp=gene_pathway_bp_dfss.values.T,W_regularizer=l2(0.0001), name='h0_bp')(
            multi_input)
        bp_drop0 = keras.layers.Dropout(0.5)(bp_layer0)


        bp_layer1 = MUlGONET(Get_Node_relation[3].shape[1], mapp=Get_Node_relation[3].values, name='h1_bp')(bp_drop0)
        bp_drop1 = keras.layers.Dropout(0.1)(bp_layer1)


        bp_layer2 = MUlGONET(Get_Node_relation[2].shape[1], mapp=Get_Node_relation[2].values, name='h2_bp')(bp_drop1)
        bp_drop2 = keras.layers.Dropout(0.1)(bp_layer2)


        bp_layer3 = MUlGONET(Get_Node_relation[1].shape[1], mapp=Get_Node_relation[1].values, name='h3_bp')(bp_drop2)
        bp_drop3 = keras.layers.Dropout(0.1)(bp_layer3)


        bp_layer4 = MUlGONET(Get_Node_relation[0].shape[1], mapp=Get_Node_relation[0].values, name='h4_bp')(bp_drop3)

        #MF
        mf_layer0 = MUlGONET(gene_pathway_mf_dfss.shape[0], mapp=gene_pathway_mf_dfss.values.T, W_regularizer=l2(0.0001),name='h0_mf')(
            multi_input)
        mf_drop0 = keras.layers.Dropout(0.5)(mf_layer0)


        mf_layer1 = MUlGONET(Get_Node_relation_mf[3].shape[1], mapp=Get_Node_relation_mf[3].values, name='h1_mf')(
            mf_drop0)
        mf_drop1 = keras.layers.Dropout(0.1)(mf_layer1)


        mf_layer2 = MUlGONET(Get_Node_relation_mf[2].shape[1], mapp=Get_Node_relation_mf[2].values, name='h2_mf')(
            mf_drop1)
        mf_drop2 = keras.layers.Dropout(0.1)(mf_layer2)


        mf_layer3 = MUlGONET(Get_Node_relation_mf[1].shape[1], mapp=Get_Node_relation_mf[1].values, name='h3_mf')(
            mf_drop2)
        mf_drop3 = keras.layers.Dropout(0.1)(mf_layer3)


        mf_layer4 = MUlGONET(Get_Node_relation_mf[0].shape[1], mapp=Get_Node_relation_mf[0].values, name='h4_mf')(
            mf_drop3)



        concat_layer = keras.layers.concatenate([bp_layer4, mf_layer4])
        concat_layer1 = keras.layers.Dense(16, activation='tanh')(concat_layer)
        finally_layer = keras.layers.Dense(1, activation='sigmoid')(concat_layer1)

    models = keras.Model(
        inputs=multi_input,
        outputs=finally_layer)

    models.compile(optimizer=optimizers,
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

    models.summary()

    return models


