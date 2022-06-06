import time
import random
import numpy as np
from utils import *
from model import *
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping

from spektral.data import Dataset, Graph
from spektral.data.loaders import SingleLoader

from tensorflow_addons.optimizers import AdamW

################ Settings
SEED = 42
EPOCHS = 1500
LR = .01
WEIGHT_DECAY = 5e-4
LAYER = 16
HIDDEN = 256
DROPOUT = .5
PATIENCE = 100
DATA = 'pubmed'
DEV = 0
ALPHA = .1
LAMBDA = .4
VARIANT = False
TEST = True
GOAL = {
  'pubmed': .803
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
adj, features, labels, idx_tr, idx_va, idx_te = load_citation(DATA)

features = np.array(features, float)
labels = np.array(labels, int)
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
  optimizer=AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY),#Adam(LR),
  loss=CategoricalCrossentropy(reduction="sum"),
  weighted_metrics=["acc"],
)

t_total = time.time()

n = features.shape[0]
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
plt.axhline(y=GOAL[DATA], color='k', linestyle='--')
plt.legend(['validation accuracy', 'goal accuracy'])
plt.ylim([0., 1.])
plt.yticks([i/100 for i in range(0, 105, 10)])
plt.grid(True)
plt.savefig('output/fig_pubmed.png')





