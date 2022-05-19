import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from spektral.layers import GCNConv

class PathfinderDiscoveryNetwork(Model):
    def __init__(self, classes, node_filters, edge_filters, edges, edge_fs, training):
        super(PathfinderDiscoveryNetwork, self).__init__()
        self.dense_1 = Dense(edge_filters, activation='relu', use_bias=True)
        self.dense_2 = Dense(1, activation='sigmoid', use_bias=True)
        self.convolution_1 = GCNConv(node_filters, activation='relu')
        self.convolution_2 = GCNConv(classes)
        self.dropout = Dropout(0.5)
        self.training = training
        
        #Marc
        self.classes = classes
        self.edges = edges
        self.edge_features = edge_fs

    def call(self, x):
        edge_x = self.edge_features

        edge_x = self.dense_1(edge_x)
        edge_x = self.dense_2(edge_x)
        edge_x = tf.squeeze(edge_x)

        n = 32
        edge_x = tf.sparse.SparseTensor(self.edges.T, edge_x, (n, n))

        x = self.convolution_1((x, edge_x))
        
        x = self.dropout(x, training=self.training)
        x = self.convolution_2((x, edge_x))
        x = -tf.nn.log_softmax(x, axis=1)

        return x
