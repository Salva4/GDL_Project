import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from model import PathfinderDiscoveryNetwork
from utils import obtain_edgesnodestarget


EPOCHS = 200
LR = .01
EDGE_FILTERS = 32
NODE_FILTERS = 32


def main():
    # TRAINING
    print('Training...')
    edges, node_features, edge_features, target, classes = obtain_edgesnodestarget(True)

    model = PathfinderDiscoveryNetwork(
        classes,
        NODE_FILTERS,
        EDGE_FILTERS,
        edges,
        edge_features,
        True
    )

    model.compile(
        optimizer=Adam(LR),
        loss=CategoricalCrossentropy(reduction="sum"),
        weighted_metrics=["acc"],
    )

    xxx = node_features
    yyy = tf.one_hot(target, classes).numpy()
    history = model.fit(xxx, yyy, epochs=EPOCHS)

    print("End of training")
    print()

    # TESTING
    print("Model accuracy:")
    edges, node_features, edge_features, target, classes2 = obtain_edgesnodestarget(False)
    assert classes == classes2

    model.edges = edges
    model.edge_features = edge_features
    model.training = False

    xxx = node_features
    yyy = tf.one_hot(target, classes).numpy()
    model.evaluate(xxx, yyy)

    # Plot evolution of accuracy during training
    plt.plot(history.history['acc'])
    plt.title('training accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.ylim([0., 1.])
    plt.yticks([i/100 for i in range(0, 105, 10)])
    plt.grid(True)
    plt.savefig('output/fig_PDN.png')


if __name__ == "__main__":
    main()