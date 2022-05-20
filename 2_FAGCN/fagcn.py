from tensorflow.keras import backend as K
import tensorflow as tf

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.random import set_seed

from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.layers import GATConv
from spektral.transforms import LayerPreprocess

import numpy as np
import random
import scipy.sparse as sp
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score as acc


def high_dim_gaussian(mu, sigma):
    if mu.ndim > 1:
        d = len(mu)
        res = np.zeros(d)
        for i in range(d):
            res[i] = np.random.normal(mu[i], sigma[i])
    else:
        d = 1
        res = np.zeros(d)
        res = np.random.normal(mu, sigma)
    return res


def generate_uniform_theta(Y, c):
    theta = np.zeros(len(Y), dtype='float')
    for i in range(c):
        idx = np.where(Y == i)
        sample = np.random.uniform(low=0, high=1, size=len(idx[0]))
        sample_sum = np.sum(sample)
        for j in range(len(idx[0])):
            theta[idx[0][j]] = sample[j] * len(idx[0]) / sample_sum
    return theta


def generate_theta_dirichlet(Y, c):
    theta = np.zeros(len(Y), dtype='float')
    for i in range(c):
        idx = np.where(Y == i)
        temp = np.random.uniform(low=0, high=1, size=len(idx[0]))
        sample = np.random.dirichlet(temp, 1)
        sample_sum = np.sum(sample)
        for j in range(len(idx[0])):
            theta[idx[0][j]] = sample[0][j] * len(idx[0]) / sample_sum
    return theta
    
def SBM(sizes, probs, mus, sigmas, noise,
        radius, feats_type='gaussian', selfloops=True):
    # -----------------------------------------------
    #     step1: get c,d,n
    # -----------------------------------------------
    c = len(sizes)
    if mus.ndim > 1:
        d = mus.shape[1]
    else:
        d = 1
    n = sizes.sum()
    all_node_ids = [ids for ids in range(0, n)]
    # -----------------------------------------------
    #     step2: generate Y with sizes
    # -----------------------------------------------
    Y = np.zeros(n, dtype='int')
    for i in range(c):
        class_i_ids = random.sample(all_node_ids, sizes[i])
        Y[class_i_ids] = i
        for item in class_i_ids:
            all_node_ids.remove(item)
    # -----------------------------------------------
    #     step3: generate A with Y and probs
    # -----------------------------------------------
    if selfloops:
        A = np.diag(np.ones(n, dtype='int'))
    else:
        A = np.zeros((n, n), dtype='int')
    for i in range(n):
        for j in range(i + 1, n):
            prob_ = probs[Y[i]][Y[j]]
            rand_ = random.random()
            if rand_ <= prob_:
                A[i][j] = 1
                A[j][i] = 1
    # -----------------------------------------------
    #     step4: generate X with Y and mus, sigmas
    # -----------------------------------------------
    X = np.zeros((n, d), dtype='float')
    for i in range(n):
        mu = mus[Y[i]]
        sigma = sigmas[Y[i]]
        X[i] = high_dim_gaussian(mu, sigma)

    return A, X, Y


def generate(p, q, idx):
    A, X, Y = \
        SBM(sizes=np.array([100, 100]),
        probs=np.array([[p, q], [q, p]]),
        mus=np.array([[-0.5]*20, [0.5]*20]),
        sigmas=np.array([[2]*20, [2]*20]),
        noise=[],
        radius=[],
        selfloops=False)
        
    return A, X, Y
        
        
def calculate(A, X, Y):

    A = sp.coo_matrix(A)
    A = A + A.T.multiply(A.T > A) - A.multiply(A.T > A)
    rowsum = np.array(A.sum(1)).clip(min=1)
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    A = A.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

    low = 0.5 * sp.eye(A.shape[0]) + A
    high = 0.5 * sp.eye(A.shape[0]) - A
    low = low.todense()
    high = high.todense()

    low_signal = np.dot(np.dot(low, low), X)
    high_signal = np.dot(np.dot(high, high), X)

    low_MLP = MLPClassifier(hidden_layer_sizes=(16), activation='relu', max_iter=2000)
    low_MLP.fit(low_signal[:100, :], Y[:100])
    low_pred = low_MLP.predict(low_signal[100:, :])

    high_MLP = MLPClassifier(hidden_layer_sizes=(16), activation='relu', max_iter=2000)
    high_MLP.fit(high_signal[:100, :], Y[:100])
    high_pred = high_MLP.predict(high_signal[100:, :])

    return acc(Y[100:], low_pred), acc(Y[100:], high_pred)


low_record = []
high_record = []


for i in range(1, 11):
    q = i * 0.01
    p = 0.05
    low_rec = []
    high_rec = []
    mlp_rec = []
    print(i, p, q)

    for j in range(10):
        A, X, Y = generate(p, q, 0)
        low, high, = calculate(A, X, Y)
        low_rec.append(low)
        high_rec.append(high)
    low_record.append([np.max(low_rec), np.min(low_rec), np.mean(low_rec)])
    high_record.append([np.max(high_rec), np.min(high_rec), np.mean(high_rec)])

A,X,Y=generate(p, q,0)
A1,X1,Y1=generate(p, q,0)

from spektral.transforms.adj_to_sp_tensor import AdjToSpTensor

L = []
for i in range(1, 11):
    q = i * 0.01
    p = 0.05
    A,X,Y = generate(p, q,0)
    L.append([p,q,A,X,Y])

from tensorflow.keras.losses import SparseCategoricalCrossentropy

import spektral.data.graph as gg
from scipy import sparse

import random
import itertools
import math
from sklearn.model_selection import train_test_split

class DictionaryLookupDataset(object):
    def __init__(self, L):
        super().__init__()
        self.L = L
        
    def generate_data(self):
      indices = np.argwhere(self.L[2]==1)
      edge_index = sparse.csr_matrix((np.ones(len(indices.T[0])),(indices.T[0],indices.T[1])),shape=(200,200))

      nodes = self.L[3]
            
      C = np.zeros((len(self.L[-1]),2))
      for i,j in enumerate(self.L[-1]):
        if j==0:
          C[i] = np.array([1,0])
        else:
          C[i] = np.array([0,1]) 
      LL=[]
      LL.append(gg.Graph(x=nodes, a=edge_index, y=C))

      return LL

import random
import itertools
import math
from sklearn.model_selection import train_test_split

class DictionaryLookupDataset(object):
    def __init__(self, L):
        super().__init__()
        self.L = L
        
    def generate_data(self):
      indices = np.argwhere(self.L[2]==1)
      edge_index = sparse.csr_matrix((np.ones(len(indices.T[0])),(indices.T[0],indices.T[1])),shape=(200,200))

      nodes = self.L[3]
            
      C = np.zeros((len(self.L[-1]),2))
      for i,j in enumerate(self.L[-1]):
        if j==0:
          C[i] = np.array([1,0])
        else:
          C[i] = np.array([0,1]) 
      LL_t=[]
      LL_v=[]
      pool0 = np.argwhere(self.L[-1]==0)
      pool1 = np.argwhere(self.L[-1]==1)
      
      np.random.shuffle(pool0)
      np.random.shuffle(pool1)
      tr = np.concatenate([pool0[0:50],pool1[0:50]])
      va = np.concatenate([pool0[50:],pool1[50:]])
      np.random.shuffle(tr)
      np.random.shuffle(va)
      
      C_tr = C[tr.flatten()] 
      C_va = C[va.flatten()] 
      LL_t.append(gg.Graph(x=nodes, a=edge_index, y=C_tr))
      LL_v.append(gg.Graph(x=nodes, a=edge_index, y=C_va))

      return LL_t,LL_v,tr.flatten(),va.flatten()

id = -1
dictionary = DictionaryLookupDataset(L[id])

LL=dictionary.generate_data()

from spektral.data import Dataset

class MyDataset(Dataset):
    """
    A dataset of five random graphs.
    """
    def __init__(self, list_g, **kwargs):
        self.list_g = list_g

        super().__init__(**kwargs)
    
    def read(self):
      return self.list_g

train = MyDataset(LL[0])
val = MyDataset(LL[1])

from spektral.utils.convolution import degree_power,add_self_loops, normalized_adjacency
from spektral.transforms import gcn_filter

ad = np.copy(L[id][2])
final = normalized_adjacency(ad)
final=final.astype('float32')
final = tf.convert_to_tensor(final)

final

#VANILLA ATTENTION NETWORK  
from tensorflow.keras import constraints, initializers, regularizers

from spektral.layers import ops
from spektral.layers.convolutional.conv import Conv
from spektral.layers.ops import modes


class FAGCN(Conv):
    def __init__(
        self,
        channels,
        eps=0.3,
        L=5,
        out=2,
        deg=None,
        dropout_rate=0.5,
        add_self_loops=False,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.channels = channels
        self.eps = eps
        self.L = L
        self.deg = deg
        self.out = out
        self.dropout_rate = dropout_rate
        self.add_self_loops = add_self_loops
        
      
        self.output_dim = out

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        self.kernel1 = self.add_weight(
            name="kernel1",
            shape=[input_dim, self.channels],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.kernel2 = self.add_weight(
            name="kernel2",
            shape=[self.channels, self.channels],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.kernel3 = self.add_weight(
            name="kernel3",
            shape=[self.channels,self.out ],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )


        
        if self.use_bias:
            self.bias = self.add_weight(
                shape=[self.output_dim],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name="bias",
            )

        self.dropout = Dropout(self.dropout_rate, dtype=self.dtype)
        self.built = True

    def call(self, inputs, mask=None):
        x, a = inputs
  
       
        output = self._call_single(x, a)
        
        if self.use_bias:
            output += self.bias
        if mask is not None:
            output *= mask[0]
        output = self.activation(output)

        return output

    def _call_single(self, x, a):
        
        
        indices = a.indices
     

        N = tf.shape(x, out_type=indices.dtype)[-2]
        
        if self.add_self_loops:
            indices = ops.add_self_loops_indices(indices, N)
       
        targets, sources = indices[:, 1], indices[:, 0]
        
        x = K.dot(x, self.kernel1)
        x = tf.reshape(x, (-1, self.channels))
        raw = tf.keras.layers.ReLU()(x)
        raw = self.dropout(raw)
        h = raw
        for layers in range(0,self.L):
          gh = K.dot(h, self.kernel2)
          theta1 = tf.gather(gh, targets)
          theta2 = tf.gather(gh, sources)
          alpha = tf.math.tanh(theta1+theta2)
          #alpha = 1 * tf.math.abs(alpha) <---- low filter: uncomment
          #alpha = -1 * tf.math.abs(alpha) <---- high filter: uncomment

       
          indexes = tf.concat([tf.expand_dims(targets, axis=1),tf.expand_dims(sources, axis=1)],1)
          div = tf.gather_nd(indices=indexes,params=self.deg)
          div = div[...,None]
          alpha = alpha*div
          alpha = self.dropout(alpha)
          sums = tf.math.multiply(alpha, tf.gather(h, sources)) 
          sums = tf.math.unsorted_segment_sum(sums, targets,N) #len(tf.gather(raw, targets)))
          h = self.eps*raw +sums

        output = K.dot(h, self.kernel3)
        
        return output

 

    @property
    def config(self):
        return {
            "channels": self.channels,
            "attn_heads": self.attn_heads,
            "concat_heads": self.concat_heads,
            "dropout_rate": self.dropout_rate,
            "return_attn_coef": self.return_attn_coef,
            "attn_kernel_initializer": initializers.serialize(
                self.attn_kernel_initializer
            ),
            "attn_kernel_regularizer": regularizers.serialize(
                self.attn_kernel_regularizer
            ),
            "attn_kernel_constraint": constraints.serialize(
                self.attn_kernel_constraint
            ),
        }

N = 200  # Number of nodes in the graph
F = 20  # Original size of node features
n_out = 2  # Number of classes

# Parameters
channels = 20  # Number of channels in each head of the first GAT layer
dropout = 0.0  # Dropout rate for the features and adjacency matrix
learning_rate = 0.01  # Learning rate
epochs = 20000  # Number of training epochs
patience = 100  # Patience for early stopping

# Model definition
x_in = Input(shape=(F))
a_in = Input((N), sparse=True)

#do_1 = Dropout(dropout)(attr)
gc_1 = FAGCN(
    channels,
    eps=0.2,
    L=1,
    deg = final,
    dropout_rate=dropout,
    activation="softmax",
)([x_in, a_in])

model = Model(inputs=[x_in, a_in], outputs=gc_1)
optimizer = Adam(learning_rate=learning_rate)

model.summary()

loader_tr = SingleLoader(train)
loader_va = SingleLoader(val)

from tensorflow import keras
# Instantiate an optimizer.

# Instantiate a loss function.
loss_fn =  tf.keras.losses.BinaryCrossentropy(from_logits=False)

"""TRAIN"""

# Commented out IPython magic to ensure Python compatibility.
epochs = 300
acc=0
running_loss = 0
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step, g in enumerate(train):
        inputs, target = loader_tr.__next__()
        
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.

            logits = model(inputs, training=True)  # Logits for this minibatch
            logits = tf.gather(logits,LL[2])
            # Compute the loss value for this minibatch.
            loss_value = loss_fn(target, logits)
            acc = sum(tf.argmax(logits,1).numpy()==tf.argmax(target,1).numpy())/len(target)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        inputs, target = loader_va.__next__()
        val_logits = model(inputs, training=False)
        val_logits = tf.gather(val_logits,LL[3])
           
        loss_value_v = loss_fn(target, val_logits)
        acc_v = sum(tf.argmax(val_logits,1).numpy()==tf.argmax(target,1).numpy())/len(target)

        print(
            "Training loss (for one batch) at step %d: %.4f"
#             % (step, float(loss_value))
        )
        
        print(
            "train-Accuracy (for one batch) at step %d: %.4f"
#             % (step, float(acc))
        )
        print(
            "val loss (for one batch) at step %d: %.4f"
#             % (step, float(loss_value_v))
        )        
        print(
            "val-Accuracy (for one batch) at step %d: %.4f"
#             % (step, float(acc_v))
        )
 
        print("Seen so far: %s samples" % ((step + 1)))