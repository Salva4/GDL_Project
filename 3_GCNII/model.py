import math
import numpy as np
import tensorflow as tf 
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Layer

class GraphConvolution(Layer):
    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.in_features = in_features
        self.out_features = out_features
        self.residual = residual
        self.W = np.zeros((self.in_features,self.out_features), float)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.W = tf.random.uniform(self.W.shape, -stdv, stdv)

    def call(self, H, adj , H0 , lamda, alpha, l):
        W = self.W
        beta = math.log(lamda/l + 1)
        A_hat = tf.cast(tf.sparse.to_dense(adj), float) + tf.eye(adj.shape[0])
        D_hat_sqrtinv = tf.linalg.diag(tf.squeeze(1 / tf.sqrt(tf.reduce_sum(A_hat, 0))))
        P = D_hat_sqrtinv @ A_hat @ D_hat_sqrtinv
        
        # Initial residual: a portion alpha of the input to each conv.layer will be H0,
        #...which still preserves information on the graph structure
        init_res = (1-alpha)*P@H + alpha*H0

        # Identity mapping: as layers get deeper, the importance of the weights decreases 
        #...to avoid the high number of interactions that, as the number of layers tends to 
        #...infinity, undermine the preservation of the graph structure information
        id_map = (1-beta)*tf.eye(W.shape[0]) + beta*W
        
        # H^(l+1) = ReLU((1-alpha)PH^(l) + alpha*H0) * ((1-beta)I + beta*W^(l)))
        output = init_res @ id_map
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
        conv_layers = []
        x = self.dropout(x, training=self.training)
        H0 = self.fc1(x)
        conv_layers.append(H0)
        H = H0
        for i, conv in enumerate(self.convs):
            H = self.dropout(H, training=self.training)
            H = self.act_fn(
                conv(
                    H, 
                    adj,
                    conv_layers[0],
                    self.lamda,
                    self.alpha,
                    i+1
                )
            )
        H = self.dropout(H, training=self.training)
        output = self.fc2(H)
        return -tf.nn.log_softmax(output, axis=1)
































