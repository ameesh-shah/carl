import networkx as nx
import scipy.sparse.csgraph
import numpy as np
import gym
import pickle
import torch

WALLS = {
    'Small':
        np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]),
    'Cross':
        np.array([[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]]),
    'FourRooms':
        np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]),
    'Spiral5x5':
        np.array([[0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1],
                  [0, 1, 0, 0, 1],
                  [0, 1, 1, 0, 1],
                  [0, 0, 0, 0, 1]]),
    'Spiral7x7':
        np.array([[1, 1, 1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 1, 1, 0],
                  [1, 0, 1, 0, 0, 1, 0],
                  [1, 0, 1, 1, 0, 1, 0],
                  [1, 0, 0, 0, 0, 1, 0],
                  [1, 1, 1, 1, 1, 1, 0]]),
    'Spiral9x9':
        np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1],
                  [0, 1, 0, 0, 0, 0, 0, 0, 1],
                  [0, 1, 0, 1, 1, 1, 1, 0, 1],
                  [0, 1, 0, 1, 0, 0, 1, 0, 1],
                  [0, 1, 0, 1, 1, 0, 1, 0, 1],
                  [0, 1, 0, 0, 0, 0, 1, 0, 1],
                  [0, 1, 1, 1, 1, 1, 1, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1]]),
    'Spiral11x11':
        np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                  [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
                  [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
                  [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
                  [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                  [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]),
    'Maze5x5':
        # np.array([[0, 0, 0],
        #           [1, 1, 0],
        #           [0, 0, 0]]),
        np.array([[0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 0],
                  [1, 1, 1, 1, 0],
                  [1, 1, 1, 1, 0],
                  [1, 1, 1, 1, 0]]),
    'Maze6x6':
        np.array([[0, 0, 1, 0, 0, 0],
                  [1, 0, 1, 0, 1, 0],
                  [0, 0, 1, 0, 1, 1],
                  [0, 1, 1, 0, 0, 1],
                  [0, 0, 1, 1, 0, 1],
                  [1, 0, 0, 0, 0, 1]]),
    'Maze11x11':
        np.array([[0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                  [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                  [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
    'Tunnel':
        np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                  [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                  [0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                  [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                  [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
                  [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]),
    'U':
        np.array([[0, 0, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [1, 1, 0],
                  [1, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 0, 0]]),
    'Tree':
        np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        ]),
    'UMulti':
        np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         ]),
    'FlyTrapSmall':
        np.array([
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
         ]),
    'FlyTrapBig':
        np.array([
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
         ]),
    'Galton':
        np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        ]),
}


ACT_DICT = [
    [0.,0.],
    [0., -1.],
    [0., 1.],
    [-1., 0.],
    [1., 0.]
]

def resize_walls(walls, factor):
  """Increase the environment by rescaling.
  
  Args:
    walls: 0/1 array indicating obstacle locations.
    factor: (int) factor by which to rescale the environment."""
  (height, width) = walls.shape
  row_indices = np.array([i for i in range(height) for _ in range(factor)])
  col_indices = np.array([i for i in range(width) for _ in range(factor)])
  walls = walls[row_indices]
  walls = walls[:, col_indices]
  assert walls.shape == (factor * height, factor * width)
  return walls



class PointmassEnv(gym.Env):
  """Abstract class for 2D navigation environments."""

  def __init__(self,
               difficulty=1,
               dense_reward=False,
               # action_noise=0.5
               action_noise=0
               ):
    """Initialize the point environment.

    Args:
      walls: (str) name of one of the maps defined above.
      resize_factor: (int) Scale the map by this factor.
      action_noise: (float) Standard deviation of noise to add to actions. Use 0
        to add no noise.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    self.plt = plt
    self.fig = self.plt.figure()
    
    self.action_dim = self.ac_dim = 2
    # Normal observation dim + random variable + catastrophe prob
    # Similar to cartpole
    self.observation_dim = self.obs_dim = 5 # x, y, action_noise, catastrophe_prob # TODO: vary this across envs
    self.env_name = 'pointmass'
    self.is_gym = True

    if difficulty == 0:
      walls = 'Maze5x5'
      resize_factor = 2
      self.fixed_start = np.array([0.5, 0.5]) * resize_factor
      self.fixed_goal = np.array([4.5, 4.5]) * resize_factor
      self.max_episode_steps = 50
    elif difficulty == 1:
      walls = 'Maze6x6'
      resize_factor = 1
      self.fixed_start = np.array([0.5, 0.5]) * resize_factor
      self.fixed_goal = np.array([1.5, 5.5]) * resize_factor
      self.max_episode_steps = 150
    elif difficulty == 2:
      walls = 'FourRooms'
      resize_factor = 2
      self.fixed_start = np.array([1.0, 1.0]) * resize_factor
      self.fixed_goal = np.array([10.0, 10.0]) * resize_factor
      self.max_episode_steps = 100
    elif difficulty == 3:
      walls = 'Maze11x11'
      resize_factor = 1
      self.fixed_start = np.array([0.5, 0.5]) * resize_factor
      self.fixed_goal = np.array([0.5, 10.5]) * resize_factor
      self.max_episode_steps = 200
    else:
      print('Invalid difficulty setting')
      return 1/0

    if resize_factor > 1:
      self._walls = resize_walls(WALLS[walls], resize_factor)
    else:
      self._walls = WALLS[walls]
    (height, width) = self._walls.shape
    self._apsp = self._compute_apsp(self._walls)

    self._height = height
    self._width = width
    self.action_space = gym.spaces.Discrete(5)

    # since this is a discrete action space, there is no high or low.
    # Instead we need to set the possible actions.
    self.action_space.low = -1 # Dummy
    self.action_space.high = -1 # Dummy
    self.possible_actions = np.asarray(ACT_DICT) 

    self.observation_space = gym.spaces.Box(
        low=np.array([0,0]),
        high=np.array([self._height, self._width]),
        dtype=np.float32)

    self.dense_reward = dense_reward
    self.num_actions = 5
    self.epsilon = resize_factor
    self.action_noise = action_noise
    self.test_action_noise = 0.2 # FIXME: make it passed in from outside.
    
    self.obs_vec = []
    self.replay_buffer = np.array([]).reshape((0, self.obs_dim))
    self.wall_hits = 0
    self.last_trajectory = None
    self.difficulty = difficulty

    self.num_runs = 0
    self.reset()

  def seed(self, seed):
    np.random.seed(seed)

  def reset(self, mode='train', seed=None):

      if seed: 
          self.seed(seed)

      if len(self.obs_vec) > 0:
          self.last_trajectory = self.plot_trajectory()

      if mode == 'train':
          # self.action_noise = np.random.uniform(low=0.0, high=1.0) 
          # print('Resetting the environment. action_noise: ' + str(self.action_noise))
          self.action_noise = 0.5 # TODO: vary this across envs
      elif mode == 'test':
          self.action_noise = self.test_action_noise
      else:
          raise ValueError('Unrecognized mode: ' + mode)

      self.plt.clf()
      self.timesteps_left = self.max_episode_steps

      if len(self.obs_vec) != 0:
        self.replay_buffer = np.concatenate([self.replay_buffer, self.obs_vec.copy()], axis=0)
      self.obs_vec = [np.concatenate([
          self._normalize_obs(self.fixed_start.copy()),
          self.goal,
          [0] # catastrophe
        ], axis=-1)]
      self.state = self.fixed_start.copy()
      self.num_runs += 1
      self.wall_hits = 0

      return np.concatenate([
          self._normalize_obs(self.state.copy()),
          self.goal,
          [0]], axis=-1)

  def reset_model(self, seed=None):
    if seed: self.seed(seed)
        
    if len(self.obs_vec) > 0:
      self.last_trajectory = self.plot_trajectory()
    
    self.plt.clf()
    self.timesteps_left = self.max_episode_steps
    
    self.replay_buffer = np.concatenate([self.replay_buffer, np.array(self.obs_vec.copy())], axis=0)
    self.obs_vec = [self._normalize_obs(self.fixed_start.copy())]
    self.state = self.fixed_start.copy()
    self.num_runs += 1
    self.wall_hits = 0

    return self._normalize_obs(self.state.copy())

  def set_logdir(self, path):
    self.traj_filepath = path + '/'
    
  def _get_distance(self, obs, goal):
    """Compute the shortest path distance.
    
    Note: This distance is *not* used for training."""
    (i1, j1) = self._discretize_state(obs.copy())
    (i2, j2) = self._discretize_state(goal.copy())
    return self._apsp[i1, j1, i2, j2]


  def simulate_step(self, state, action):
    num_substeps = 1
    dt = 1 / num_substeps
    num_axis = len(action)
    for _ in np.linspace(0, 1, num_substeps):
      for axis in range(num_axis):
        new_state = state.copy()
        new_state[axis] += dt * action[axis]

        if not self._is_blocked(new_state):
          state = new_state
    return state
    
  def _discretize_state(self, state, resolution=1.0):
    (i, j) = np.floor(resolution * state).astype(np.int)
    # Round down to the nearest cell if at the boundary.
    if i == self._height:
      i -= 1
    if j == self._width:
      j -= 1
    return (i, j)

  def _normalize_obs(self, obs):
    return np.array([
      obs[0] / float(self._height),
      obs[1] / float(self._width)
    ])

  def _unnormalize_obs(self, obs):
    return np.array([
      obs[0] * float(self._height),
      obs[1] * float(self._width)
    ])

  def _normalize_ac(self, ac):
    return ac / np.linalg.norm(ac)
  
  def _is_blocked(self, state):
    # Check if the state is out of bound
    if not self.observation_space.contains(state):
      return True
    # Check if the state overlaps with wall
    (i, j) = self._discretize_state(state)
    return (self._walls[i, j] == 1)

  def get_dist_and_reward(self, state):
    """
    Args:
        state: UNNORMALIZED state observation (in [0, 10] instead [0, 1]
    """
    if (isinstance(state, torch.Tensor)):
        state = state.detach().cpu().numpy()
    # print('State: ' + str(state))
    # print('Goal: ' + str(self.fixed_goal))
    dist = np.linalg.norm(state - self.fixed_goal, axis=(state.ndim-1))
    # print('Dist: ' + str(dist))

    # In CARL, we want sparse reward
    # dense_reward defaults to False
    if self.dense_reward:
        reward = -dist
    else:
        reward = (dist < self.epsilon).astype(int) - 1
    
    return dist, reward 

  def step(self, action):
    self.timesteps_left -= 1
    action = np.random.normal(action, self.action_noise)
    action = self._normalize_ac(action)
    next_state = self.simulate_step(self.state, action)

    # If simulate_step returns the same state as the input state,
    # then the action is blocked, because the state is out-of-bound
    # or overlaps with the wall. 
    # When the agent hits the boundary, the catastrophe flag is set to True.
    #
    # We also count the number of time the agent runs into the wall.
    # A trained agent should minimize the number of times hitting the wall.
    if np.array_equal(next_state, self.state):
        catastrophe = True
    else:
        catastrophe = False
    self.state = next_state

    # Compute distance between current state and goal state
    # dist = np.linalg.norm(self.state - self.fixed_goal)
    dist, reward = self.get_dist_and_reward(self.state)
    done = (dist < self.epsilon) or (self.timesteps_left == 0)

    # Normalized original obs
    normalized_obs = self._normalize_obs(self.state.copy())

    # Add random env variable and default catastrophe 0
    # Some potential choices for random env variable:
    # 1. action_noise (currently in use, see _get_obs())
    # 2. resize_factor of the walls
    extended_obs = np.concatenate([
            normalized_obs,
            self.goal,
            [int(catastrophe)]
        ], axis=-1)

    # catastrophe calculation: when agent hits wall, catastrophe.
    info = {}
    if catastrophe:
        info['Catastrophe'] = True
    else:
        info['Catastrophe'] = False

    # log the number of times the agent has hit the wall
    info['Wall_hits_so_far'] = self.wall_hits

    # obs_vec is used for plotting trajectories.
    self.obs_vec.append(extended_obs.copy())

    return extended_obs, reward, done, info

  @property
  def walls(self):
    return self._walls

  @property
  def goal(self):
    return self._normalize_obs(self.fixed_goal.copy())

  def _compute_apsp(self, walls):
    (height, width) = walls.shape
    g = nx.Graph()
    # Add all the nodes
    for i in range(height):
      for j in range(width):
        if walls[i, j] == 0:
          g.add_node((i, j))

    # Add all the edges
    for i in range(height):
      for j in range(width):
        for di in [-1, 0, 1]:
          for dj in [-1, 0, 1]:
            if di == dj == 0: continue  # Don't add self loops
            if i + di < 0 or i + di > height - 1: continue  # No cell here
            if j + dj < 0 or j + dj > width - 1: continue  # No cell here
            if walls[i, j] == 1: continue  # Don't add edges to walls
            if walls[i + di, j + dj] == 1: continue  # Don't add edges to walls
            g.add_edge((i, j), (i + di, j + dj))

    # dist[i, j, k, l] is path from (i, j) -> (k, l)
    dist = np.full((height, width, height, width), np.float('inf'))
    for ((i1, j1), dist_dict) in nx.shortest_path_length(g):
      for ((i2, j2), d) in dist_dict.items():
        dist[i1, j1, i2, j2] = d

    return dist

  def render(self, mode=None):
    self.plot_walls()

    # current and end
    self.plt.plot(self.fixed_goal[0], self.fixed_goal[1], 'go')
    self.plt.plot(self.state[0], self.state[1], 'ko')
    self.plt.pause(0.1)

    img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
    return img

  def plot_trajectory(self):
    self.plt.clf()
    self.plot_walls()

    obs_vec, goal = np.array(self.obs_vec), self.goal
    # hardcoded: original maze is 5x5 so 1 unit normalized is 1 / 5
    reached_goal = np.linalg.norm(obs_vec[-1, 0:1] - goal) < .2
    self.plt.plot(obs_vec[:, 0], obs_vec[:, 1], 'b-o', alpha=0.3)
    self.plt.scatter([obs_vec[0, 0]], [obs_vec[0, 1]], marker='+',
                color='red', s=200, label='start')
    self.plt.scatter([obs_vec[-1, 0]], [obs_vec[-1, 1]], marker='*' if reached_goal else '+',
                color='green', s=200, label='end')
    self.plt.scatter([goal[0]], [goal[1]], marker='*',
                color='green', s=200, label='goal')

    # Annotate rewards
    for i in range(len(obs_vec)):
        unnormalized_obs = self._unnormalize_obs(obs_vec[i, :2])
        _, rwd = self.get_dist_and_reward(unnormalized_obs)
        self.plt.annotate(rwd, obs_vec[i, :2])

    # Draw a rewarded states for sparse rewards
    # Draw catastrophe states
    catastrophic_states = []
    rewarding_states = []
    for e in obs_vec:
        if e[-1] != 0:
            catastrophic_states.append(e)
        if not self.dense_reward:
            unnormalized_obs = self._unnormalize_obs(e[:2])
            d, re = self.get_dist_and_reward(unnormalized_obs)
            if re >= 0:
                rewarding_states.append(e)
    
    catastrophic_states = np.array(catastrophic_states)
    rewarding_states = np.array(rewarding_states)

    print(rewarding_states)

    if catastrophic_states.shape[0] > 0:
        self.plt.scatter([catastrophic_states[:, 0]], [catastrophic_states[:, 1]], marker='x', color='red', s=200, label='catastrophe')

    if not self.dense_reward and rewarding_states.shape[0] > 0:
        self.plt.scatter([rewarding_states[:, 0]], [rewarding_states[:, 1]], marker='o', color='green', s=200)

    # Create empty plot with blank marker containing the extra label
    self.plt.plot([], [], ' ', label="# of wall hits: " + str(self.wall_hits))

    self.plt.legend()
    self.plt.savefig(self.traj_filepath + 'sampled_traj_' + str(self.num_runs) + '.png')

  def plot_density_graph(self):
    self.plt.clf()
    H, xedges, yedges = np.histogram2d(self.replay_buffer[:,0], self.replay_buffer[:,1], range=[[0., 1.], [0., 1.]], density=True)
    self.plt.imshow(np.rot90(H), interpolation='bicubic')
    self.plt.colorbar()
    self.plt.title('State Density')
    self.fig.savefig(self.traj_filepath + 'density' + '.png', bbox_inches='tight')

  def get_last_trajectory(self):
    return self.last_trajectory

  def plot_walls(self, walls=None):
    if walls is None:
      walls = self._walls.T
    (height, width) = walls.shape
    for (i, j) in zip(*np.where(walls)):
      x = np.array([j, j+1]) / float(width)
      y0 = np.array([i, i]) / float(height)
      y1 = np.array([i+1, i+1]) / float(height)
      self.plt.fill_between(x, y0, y1, color='grey')
    self.plt.xlim([0, 1])
    self.plt.ylim([0, 1])
    self.plt.xticks([])
    self.plt.yticks([])
  
  def _sample_normalized_empty_state(self):
    s = self._sample_empty_state()
    return self._normalize_obs(s)
  
  def _sample_empty_state(self):
    candidate_states = np.where(self._walls == 0)
    num_candidate_states = len(candidate_states[0])
    state_index = np.random.choice(num_candidate_states)
    state = np.array([candidate_states[0][state_index],
                      candidate_states[1][state_index]],
                     dtype=np.float)
    state += np.random.uniform(size=2)
    assert not self._is_blocked(state)
    return state

def refresh_path():
  path = dict()
  path['observations'] = []
  path['actions'] = []
  path['next_observations'] = []
  path['terminals'] = []
  path['rewards'] = []
  return path

