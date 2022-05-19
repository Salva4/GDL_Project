import numpy as np


def obtain_edgesnodestarget(training):
  nodes_name = 'tr' if training else 'te'
  path_ds = f'graph/nodes_{nodes_name}/'

  edges = np.loadtxt(path_ds + 'edges.dat')
  edges = np.array(edges, int) - int(not training)*32
  node_features = np.loadtxt(path_ds + 'node_features.dat')
  edge_features = np.loadtxt(path_ds + 'edge_features.dat')
  target = np.loadtxt(path_ds + 'target.dat')

  with open(path_ds + 'classes.txt', 'r') as f:
    classes = int(f.read())
  
  return edges, node_features, edge_features, target, classes