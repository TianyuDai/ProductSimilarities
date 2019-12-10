#!/usr/bin/python

# Spectral Clustering 

import os
import numpy as np
from sklearn.cluster import SpectralClustering

from specSimilarity import evalProductSpecs
from listspecs import readStat

def clusteringSimilar(spec, ncluster=4, fname=None):
    if fname is None:
        fname = 'similarity_'+spec+'.txt'
    if not os.path.isfile(fname):
        evalProductSpecs(spec)
    mat = np.loadtxt(fname)
    # mat = np.array([[1, 0.2, 0.1 ,0 ,0], [0.2, 1, 0.1, 0, 0], [0.1, 0.1, 1, 0, 0], [0,0,0,1,0.5], [0,0,0,0.5,1]]) # debug
    labels = SpectralClustering(ncluster).fit_predict(mat)
    names = readStat(spec)
    # distribute different clusters
    allvals = list()
    for r in range(ncluster):
        allvals.append(list())
    for a, b in zip(labels, names):
        allvals[a].append(b[0])
    for r in range(ncluster):
        print('Cluster %d has following spec values' % (r), end='')
        for a, b in enumerate(allvals[r]):
            if a%5==0:
                print()
            print('%s, ' % (b), end='')
        if r<ncluster-1:
            print('\n')


if __name__ == '__main__':
    clusteringSimilar('tags', 2)