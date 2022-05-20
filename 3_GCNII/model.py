import math
import numpy as np
import tensorflow as tf 
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Layer

class GraphConvolution(Layer):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = np.zeros((self.in_features,self.out_features), float)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight = tf.random.uniform(self.weight.shape, -stdv, stdv)

    def call(self, inp, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = tf.cast(tf.sparse.to_dense(adj), float) @ inp
        if self.variant:
            support = np.concatenate([hi,h0], axis=1)
            r = (1-alpha)*hi + alpha*h0
        else:
            support = (1-alpha)*hi + alpha*h0
            r = support
        output = theta*(support @ self.weight) + (1-theta)*r
        if self.residual:
            output = output + inp
        return output

class GCNII(Model):
    def __init__(
      self, 
      nfeat, 
      nlayers,
      nhidden, 
      nclass, 
      dropout, 
      lamda, 
      alpha, 
      variant,
      training
    ):
        super().__init__()
        self.convs = []
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant))
        self.fc1 = Dense(nhidden, activation='relu', use_bias=True)
        self.fc2 = Dense(nclass, activation=None, use_bias=True)
        self.act_fn = tf.nn.relu
        self.dropout = Dropout(dropout)
        self.alpha = alpha
        self.lamda = lamda
        self.training = training

    def call(self, x):
        x, adj = x
        _layers = []
        x = self.dropout(x, training=self.training)
        layer_inner = self.fc1(x)
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = self.dropout(layer_inner, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = self.dropout(layer_inner, training=self.training)
        layer_inner = self.fc2(layer_inner)
        return -tf.nn.log_softmax(layer_inner, axis=1)






