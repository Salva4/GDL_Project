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

################ Settings
SEED = 42
EPOCHS = 1500
LR = .01
LAYER = 64
HIDDEN = 64
DROPOUT = .6
PATIENCE = 100
DATA = 'citeseer'
DEV = 0
ALPHA = .1
LAMBDA = .5
VARIANT = False
TEST = True
GOAL = {
  'cora': .855,
  'citeseer': .734,
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
adj, features, labels, idx_train, idx_val, idx_test = load_citation(DATA)

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
  optimizer=Adam(LR),
  loss=CategoricalCrossentropy(reduction="sum"),
  weighted_metrics=["acc"],
)

t_total = time.time()

n = features.shape[0]
indices = list(range(n))
np.random.shuffle(indices)
weights_tr = np.mean(
  np.concatenate((
    np.ones(int(.6*n), float), 
    np.zeros(int(.8*n) - int(.6*n), float),
    np.zeros(n - int(.8*n), float)
  ))
)
weights_va = np.mean(
  np.concatenate((
    np.zeros(int(.6*n), float), 
    np.ones(int(.8*n) - int(.6*n), float),
    np.zeros(n - int(.8*n), float)
  ))
)
weights_te = np.mean(
  np.concatenate((
    np.zeros(int(.6*n), float), 
    np.zeros(int(.8*n) - int(.6*n), float),
    np.ones(n - int(.8*n), float)
  ))
)

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
plt.savefig('output/fig_citeseer.png')





