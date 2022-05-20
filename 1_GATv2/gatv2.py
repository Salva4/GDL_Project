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

from tensorflow.keras.losses import SparseCategoricalCrossentropy

import spektral.data.graph as gg
from scipy import sparse

import random
import itertools
import math
from sklearn.model_selection import train_test_split

class DictionaryLookupDataset(object):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.edges, self.empty_id = self.init_edges()
      
    def init_edges(self):
        targets = range(0, self.size)
        sources = range(self.size, self.size * 2)
        next_unused_id = self.size
        all_pairs = itertools.product(sources, targets)
        edges = [list(i) for i in zip(*all_pairs)]

        return edges, next_unused_id

    def create_empty_graph(self, add_self_loops=False):
      edge_index = np.array(self.edges, dtype=np.long)
      return edge_index
    
    def get_combinations(self):
      # returns: an iterable of [permutation(size)]
      # number of combinations: size!
      max_examples = 32000 # starting to affect from size=8, because 8!==40320
      if math.factorial(self.size) > max_examples:
        permutations = [np.random.permutation(range(self.size)) for _ in range(max_examples)]
      else:
        permutations = itertools.permutations(range(self.size))
        
      return permutations
    
    def generate_data(self, train_fraction, unseen_combs):
      data_list = []
      for perm in self.get_combinations():
        edge_index = self.create_empty_graph(add_self_loops=False)
        edge_index = sparse.csr_matrix((np.ones(self.size*self.size),(edge_index[0],edge_index[1])),shape=(self.size*2,self.size*2))
        nodes = np.array(self.get_nodes_features(perm),dtype=np.long)
        target_mask =  np.array([True] * (self.size) + [False] * self.size, dtype=np.bool)
        labels = np.array(perm, dtype=np.long)
        
        data_list.append(gg.Graph(x=nodes, a=edge_index, target_mask=target_mask, y=labels))

      dim0, out_dim = self.get_dims()
      if unseen_combs:
        X_train, X_test = self.unseen_combs_train_test_split(data_list, train_fraction=train_fraction, shuffle=True)
      else:
        X_train, X_test = train_test_split(data_list, train_size=train_fraction, shuffle=True)

      return X_train, X_test, dim0, out_dim

    def get_nodes_features(self, perm):
      # perm: a list of indices
      #Node features is basically {[(A,_),(B,_),...(D,_)] , [(A,1),(B,2),...(D,4)]}.
      #Then what is nodes,5,6,7,8,9? These are node numberings. there exists 2k=10 nodes and each have features i.e. a 2-tuple.
      # The first row contains (key, empty_id)
      # The second row contains (key, value) where the order of values is according to perm
      nodes = [(key, self.empty_id) for key in range(self.size)]
      for key, val in zip(range(self.size), perm):
        nodes.append((key, val))

      return nodes

    def get_dims(self):
      # get input and output dims
      in_dim = self.size + 1
      out_dim = self.size
      return in_dim, out_dim

    def unseen_combs_train_test_split(self, data_list, train_fraction, shuffle=True):
      per_position_fraction = train_fraction ** (1 / self.size)
      num_training_pairs = int(per_position_fraction * (self.size ** 2))
      allowed_positions = set(random.sample(list(itertools.product(range(self.size), range(self.size))), num_training_pairs))
      train = []
      test = []
        
      for example in data_list:
        if all([(i, label.item()) in allowed_positions for i, label in enumerate(example.y)]):
          train.append(example)
        else:
          test.append(example)
        
        if shuffle:
            random.shuffle(train)
      return train, test

nodes_num = 5
dictionary = DictionaryLookupDataset(nodes_num)

dictionary.get_dims()

# For k=5
X_train, X_test, dim0, out_dim = dictionary.generate_data(0.75,False)
mean_divisor = 90
update = 30

# For k=4. Dont run if above has been chosen
'''
X_train, X_test, dim0, out_dim = dictionary.generate_data(0.85,False)
mean_divisor = 20
update = 5
'''

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

md1 = MyDataset(X_train)
md2 = MyDataset(X_test)

"""GAT"""

#GAT

from tensorflow.keras import constraints, initializers, regularizers
from spektral.layers import ops
from spektral.layers.convolutional.conv import Conv
from spektral.layers.ops import modes


class GAT(Conv):
    def __init__(
        self,
        channels,
        attn_heads=1,
        concat_heads=True,
        dropout_rate=0.5,
        return_attn_coef=True,
        add_self_loops=True,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        attn_kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        bias_regularizer=None,
        attn_kernel_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        attn_kernel_constraint=None,
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
        self.attn_heads = attn_heads
        self.concat_heads = concat_heads
        self.dropout_rate = dropout_rate
        self.return_attn_coef = return_attn_coef
        self.add_self_loops = add_self_loops
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)

        if concat_heads:
            self.output_dim = self.channels * self.attn_heads
        else:
            self.output_dim = self.channels

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
       
        self.kernel = self.add_weight(
            name="kernel",
            shape=[input_dim, self.attn_heads, self.channels],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.attn_kernel_self = self.add_weight(
            name="attn_kernel_self",
            shape=[self.channels, self.attn_heads, 1],
            initializer=self.attn_kernel_initializer,
            regularizer=self.attn_kernel_regularizer,
            constraint=self.attn_kernel_constraint,
        )
        self.attn_kernel_neighs = self.add_weight(
            name="attn_kernel_neigh",
            shape=[self.channels, self.attn_heads, 1],
            initializer=self.attn_kernel_initializer,
            regularizer=self.attn_kernel_regularizer,
            constraint=self.attn_kernel_constraint,
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

        mode = ops.autodetect_mode(x, a)
        if mode == modes.SINGLE and K.is_sparse(a):
            output, attn_coef = self._call_single(x, a)
        else:
            if K.is_sparse(a):
                a = tf.sparse.to_dense(a)
            output, attn_coef = self._call_dense(x, a)

        if self.concat_heads:
            shape = tf.concat(
                (tf.shape(output)[:-2], [self.attn_heads * self.channels]), axis=0
            )
            output = tf.reshape(output, shape)
        else:
            output = tf.reduce_mean(output, axis=-2)

        if self.use_bias:
            output += self.bias
        if mask is not None:
            output *= mask[0]
        output = self.activation(output)

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output

    def _call_single(self, x, a):
        # Reshape kernels for efficient message-passing
        kernel = tf.reshape(self.kernel, (-1, self.attn_heads * self.channels))
        attn_kernel_self = ops.transpose(self.attn_kernel_self, (2, 1, 0))
        attn_kernel_neighs = ops.transpose(self.attn_kernel_neighs, (2, 1, 0))

        # Prepare message-passing
        
        indices = a.indices

        N = tf.shape(x, out_type=indices.dtype)[-2]
        if self.add_self_loops:
            indices = ops.add_self_loops_indices(indices, N)
       
        targets, sources = indices[:, 1], indices[:, 0]
        # # Update node features
        x = K.dot(x, kernel)
        x = tf.reshape(x, (-1, self.attn_heads, self.channels))

        # Compute attention
        attn_for_self = tf.reduce_sum(x * attn_kernel_self, -1) #sums up "deep" hidden representation after attention kernel operation.
        attn_for_self = tf.gather(attn_for_self, targets) #targets recieve attention hence attention for self. e(h_i,h_j) -> edge j to i so source=j and target=i
        attn_for_neighs = tf.reduce_sum(x * attn_kernel_neighs, -1)
        attn_for_neighs = tf.gather(attn_for_neighs, sources) #sources give attention.
        attn_coef = attn_for_self + attn_for_neighs
        attn_coef = tf.nn.leaky_relu(attn_coef, alpha=0.2)
        attn_coef = ops.unsorted_segment_softmax(attn_coef, targets, N)

        attn_coef = self.dropout(attn_coef)
        attn_coef = attn_coef[..., None]
        # Update representation
        output = attn_coef * tf.gather(x, sources)
        output = tf.math.unsorted_segment_sum(output, targets, N)
        
        return output, attn_coef

    def _call_dense(self, x, a):
        shape = tf.shape(a)[:-1]
        if self.add_self_loops:
            a = tf.linalg.set_diag(a, tf.ones(shape, a.dtype))
        x = tf.einsum("...NI , IHO -> ...NHO", x, self.kernel)

        attn_for_self = tf.einsum("...NHI , IHO -> ...NHO", x, self.attn_kernel_self)

        attn_for_neighs = tf.einsum(
            "...NHI , IHO -> ...NHO", x, self.attn_kernel_neighs
        )
        attn_for_neighs = tf.einsum("...ABC -> ...CBA", attn_for_neighs)

        attn_coef = attn_for_self + attn_for_neighs
        attn_coef = tf.nn.leaky_relu(attn_coef, alpha=0.2)

        mask = tf.where(a == 0.0, -10e9, 0.0)
        mask = tf.cast(mask, dtype=attn_coef.dtype)
        attn_coef += mask[..., None, :]
        attn_coef = tf.nn.softmax(attn_coef, axis=-1)
        attn_coef_drop = self.dropout(attn_coef)

        output = tf.einsum("...NHM , ...MHI -> ...NHI", attn_coef_drop, x)
        return output, attn_coef

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

"""GATV2"""

#Gatv2-self-att
from tensorflow.keras import constraints, initializers, regularizers

from spektral.layers import ops
from spektral.layers.convolutional.conv import Conv
from spektral.layers.ops import modes


class GATConv2(Conv):
    def __init__(
        self,
        channels,
        attn_heads=1,
        concat_heads=True,#True for 8 heads
        dropout_rate=0.5,
        return_attn_coef=True,
        add_self_loops=True,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        attn_kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        bias_regularizer=None,
        attn_kernel_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        attn_kernel_constraint=None,
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
        self.attn_heads = attn_heads
        self.concat_heads = concat_heads
        self.dropout_rate = dropout_rate
        self.return_attn_coef = return_attn_coef
        self.add_self_loops = add_self_loops
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)

        if concat_heads:
            self.output_dim = self.channels * self.attn_heads
        else:
            self.output_dim = self.channels

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
       
        self.kernel = self.add_weight(
            name="kernel",
            shape=[input_dim, self.attn_heads, self.channels],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.attn_kernel_self = self.add_weight(
            name="attn_kernel_self",
            shape=[self.channels, self.attn_heads, 1],
            initializer=self.attn_kernel_initializer,
            regularizer=self.attn_kernel_regularizer,
            constraint=self.attn_kernel_constraint,
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

        mode = ops.autodetect_mode(x, a)
        if mode == modes.SINGLE and K.is_sparse(a):
            output, attn_coef = self._call_single(x, a)
        else:
            if K.is_sparse(a):
                a = tf.sparse.to_dense(a)
            output, attn_coef = self._call_dense(x, a)

        if self.concat_heads:
            shape = tf.concat(
                (tf.shape(output)[:-2], [self.attn_heads * self.channels]), axis=0
            )
            output = tf.reshape(output, shape)
        else:
            output = tf.reduce_mean(output, axis=-2)

        if self.use_bias:
            output += self.bias
        if mask is not None:
            output *= mask[0]
        output = self.activation(output)

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output

    def _call_single(self, x, a):
        # Reshape kernels for efficient message-passing
        kernel = tf.reshape(self.kernel, (-1, self.attn_heads * self.channels))
        attn_kernel_self = ops.transpose(self.attn_kernel_self, (2, 1, 0))
   

        # Prepare message-passing
        
        indices = a.indices

        N = tf.shape(x, out_type=indices.dtype)[-2]
        #print(N)
        if self.add_self_loops:
            indices = ops.add_self_loops_indices(indices, N)
       
        targets, sources = indices[:, 1], indices[:, 0]
        # # Update node features

        x = K.dot(x, kernel)
        x = tf.reshape(x, (-1, self.attn_heads, self.channels))
        xr = tf.nn.leaky_relu(x, alpha=0.2)
        attn_for_self = tf.reduce_sum(xr * attn_kernel_self, -1)
        attn_for_self = tf.gather(attn_for_self, targets)


        # Compute attention
        
        attn_coef = ops.unsorted_segment_softmax(attn_for_self, targets, N)

        attn_coef = self.dropout(attn_coef)
        attn_coef = attn_coef[..., None]
        output = attn_coef * tf.gather(x, sources)
        output = tf.math.unsorted_segment_sum(output, targets, N)
        
        return output, attn_coef


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

N = nodes_num*2  # Number of nodes in the graph
F = 2  # Original size of node features
n_out = nodes_num  # Number of classes

# Parameters
channels = 128  # Number of channels in each head of the first GAT layer
n_attn_heads = 1  # Number of attention heads in first GAT layer
dropout = 0.0  # Dropout rate for the features and adjacency matrix
l2_reg = 2.5e-4  # L2 regularization rate
learning_rate = 0.001#5e-3  # Learning rate
epochs = 20000  # Number of training epochs
patience = 100  # Patience for early stopping

# Model definition
x_in = Input(shape=(F))
a_in = Input((N), sparse=True)

keys = tf.keras.layers.Embedding(nodes_num+1, 128)(x_in[:,0])
values = tf.keras.layers.Embedding(nodes_num+1, 128)(x_in[:,1])
attr = keys + values
layer = tf.keras.layers.ReLU()
attr = layer(attr)

#GATConv2 for gat2
gc_1 = GAT(
    channels,
    attn_heads=n_attn_heads,
    concat_heads=False, #True When >=2heads else False
    dropout_rate=dropout,
    activation="relu",
)([attr, a_in])

#GATConv2 for gat2
gc_2 = GAT(
    n_out,
    attn_heads=1,
    concat_heads=False, #always False
    dropout_rate=dropout,
    activation="softmax",
)([gc_1[0], a_in])

model = Model(inputs=[x_in, a_in], outputs=gc_2)
optimizer = Adam(learning_rate=learning_rate)

model.summary()

l=[]
for step, g in enumerate(md1):
  loader_tr  = SingleLoader(MyDataset([g]))
  l.append(loader_tr.__next__())

from tensorflow import keras
# Instantiate an optimizer.

# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

"""Training"""

# Commented out IPython magic to ensure Python compatibility.
epochs = 4000
acc=0
running_loss = 0
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step, g in enumerate(md1):
        inputs, target = l[step]
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.

            logits, att = model(inputs, training=True)  # Logits for this minibatch
            logits = logits[0:n_out]
            # Compute the loss value for this minibatch.
            loss_value = loss_fn(target, logits)
            running_loss+=loss_value
            acc += (tf.argmax(logits,1)==tf.reshape(target,-1)).numpy().sum()==len(tf.reshape(target,-1))

    


        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        
        grads = tape.gradient(loss_value, model.trainable_weights)
        if step%update==0:
          old = [0]*len(grads)
          for i,j in enumerate(grads):
            if i >1:
              old[i] = j + old[i]
            else:
              old[i] = tf.IndexedSlices(j.values + old[i] , j.indices,  j.dense_shape) 
      
        else:
          for i,j in enumerate(grads):
            if i >1:
              old[i] = j + old[i] 
            else:
              old[i] = tf.IndexedSlices(j.values + old[i].values , j.indices,  j.dense_shape) 
        
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        if step % update == update-1:
    
          for i,j in enumerate(old):
            if i >1:
              old[i] = j/update 
            else:
              old[i] = tf.IndexedSlices(j.values/update , j.indices,  j.dense_shape) 
            
          optimizer.apply_gradients(zip(old, model.trainable_weights))
          old = [0]*len(grads)
        # Log every 200 batches.
        if step == 0 and epoch!=0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
#                 % (step, float(running_loss/mean_divisor))
            )
            
            print(
                "Accuracy (for one batch) at step %d: %.4f"
#                 % (step, float(acc/mean_divisor))
            )
            acc=0
            running_loss=0
            print("Seen so far: %s samples" % ((step + 1)))

"""Testing"""

l=[]
for step, g in enumerate(md2):
  loader_tr  = SingleLoader(MyDataset([g]))
  l.append(loader_tr.__next__())

acc=0
running_loss = 0

for step, g in enumerate(md2):
    inputs, target = l[step]
    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation.


    logits,att = model(inputs, training=False)  # Logits for this minibatch
    logits = logits[0:n_out]
    # Compute the loss value for this minibatch.
    loss_value = loss_fn(target, logits)
    running_loss+=loss_value
    acc += (tf.argmax(logits,1)==tf.reshape(target,-1)).numpy().sum()==len(tf.reshape(target,-1))

print("f-loss",running_loss/len(md2))
print("f-acc",acc/len(md2))