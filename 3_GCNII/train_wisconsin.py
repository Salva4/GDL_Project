import time
import random
import numpy as np
from utils import *
from model import *
import matplotlib.pyplot as plt
import scipy.sparse as sp

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping

from spektral.data import Dataset, Graph
from spektral.data.loaders import SingleLoader

################ Settings
SEED = 42
EPOCHS = 1500
LR = .1
LAYER = 64
HIDDEN = 64
DROPOUT = .6
PATIENCE = 100
NEW_DATA = 'wisconsin'
DEV = 0
ALPHA = .1
LAMBDA = .5
VARIANT = False
TEST = True
GOAL = {
  'cornell': .7649,
  'texas': .7784,
  'wisconsin': .8157
}
#########################

########## Create dataset
class DS(Dataset):
    """
    A dataset of five random graphs.
    """
    def __init__(
      self, 
      node_features, 
      adj, 
      labels, 
      **kwargs
    ):
        self.node_features = node_features
        self.adj = adj
        self.labels = labels

        super().__init__(**kwargs)

    def read(self):
        g = Graph(
            x=self.node_features, 
            a=self.adj, 
            y=self.labels
        )
        output = [g]
        return output
#########################

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# Load data
with open(f'new_data/{NEW_DATA}/out1_graph_edges.txt', 'r') as f:
  text = f.read()
rows = text.split('\n')[1:-1]   # remove header and empty row at the end
edges = [row.split('\t') for row in rows]
edges = np.array(edges, dtype=int)

with open(f'new_data/{NEW_DATA}/out1_node_feature_label.txt', 'r') as f:
  text = f.read()
rows = text.split('\n')[1:-1]   # remove header and empty row at the end
node_features = []
labels = []
for row in rows:
  index, nf, t = row.split('\t')
  node_features.append(nf.split(','))
  labels.append(t)
features = np.array(node_features, dtype=float)
labels = np.array(labels, dtype=int)
labels = tf.one_hot(labels, max(labels)+1)

n = features.shape[0]
adj = sp.csr_matrix(
  (
    np.ones((edges.shape[0],), int), 
    edges.T
  ), shape=(n, n)
)
adj = np.array(adj.todense(), int)

dataset = DS(
  node_features=features,
  adj=adj,
  labels=labels
)

model = GCNII(
  nfeat=features.shape[1],
  nlayers=LAYER,
  nhidden=HIDDEN,
  nclass=dataset[0].n_labels,
  dropout=DROPOUT,
  lamda = LAMBDA, 
  alpha=ALPHA,
  variant=VARIANT,
  training=True
)
 
model.compile(
  optimizer=Adam(LR),
  loss=CategoricalCrossentropy(reduction="sum"),
  weighted_metrics=["acc"],
)

t_total = time.time()

n = features.shape[0]
indices = list(range(n))
idx_tr = indices[:int(.6*n)]
idx_va = indices[int(.6*n):int(.8*n)]
idx_te = indices[int(.8*n):]
weights_tr = np.zeros((n,), float)
weights_va = np.zeros((n,), float)
weights_te = np.zeros((n,), float)
weights_tr[idx_tr] = 1. 
weights_va[idx_va] = 1. 
weights_te[idx_te] = 1. 
weights_tr /= np.sum(weights_tr)
weights_va /= np.sum(weights_va)
weights_te /= np.sum(weights_te)

loader_tr = SingleLoader(dataset, sample_weights=weights_tr)
loader_va = SingleLoader(dataset, sample_weights=weights_va)
loader_te = SingleLoader(dataset, sample_weights=weights_te)

history = model.fit(
    loader_tr.load(),
    epochs=EPOCHS,
    steps_per_epoch=loader_tr.steps_per_epoch,
    validation_data=loader_va.load(),
    validation_steps=loader_va.steps_per_epoch,
    callbacks=[EarlyStopping(patience=PATIENCE, restore_best_weights=True)],
)

print("Training cost: {:.4f}s".format(time.time() - t_total))

if TEST:
    print('\nEvaluating model:')
    model.evaluate(
      loader_te.load(),
      steps=loader_te.steps_per_epoch
    )

# Plot evolution of validation accuracy 
plt.plot(history.history['val_acc'])
plt.title('Validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.axhline(y=GOAL[NEW_DATA], color='k', linestyle='--')
plt.legend(['validation accuracy', 'goal accuracy'])
plt.ylim([0., 1.])
plt.yticks([i/100 for i in range(0, 105, 10)])
plt.grid(True)
plt.savefig('output/fig_wisconsin.png')





