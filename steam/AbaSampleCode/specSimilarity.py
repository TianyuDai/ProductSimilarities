# calculate similarity and importance for a given spec
import os, sys
import numpy as np
import pandas as pd

import listspecs
from parse import parse

def importance(spec, value1, value2):
    #
    pass
    
def similarity(spec, value1, value2):
    # e.g. spec='tags', value1='2D', value2='2.5D'
    pass
    
def evalProductSpecs(spec):
    """Evaluate the similarity of values for a multiple choice specification.
e.g. Action and Adventure in tags.
Write similarity_*.txt"""
    # first count specs
    fname = 'stat_'+spec+'.txt'
    if not os.path.isfile(fname):
        listspecs.listSpecs(spec)
    specDicts = {}
    specCounts = []
    idx = 0
    with open(fname, 'r') as fin:
        for line in fin:
            a, b = line.rstrip().split('\t')
            specDicts[a] = idx
            specCounts.append(int(b))
            idx += 1
    specCounts = np.array(specCounts)
    nValues = idx
    # then stat products
    datahandle = parse('../data/steam_games_lite.json.gz')
    dictItemSpecs = {}
    idx = 0
    for line in datahandle:
        if spec in line:
            specs = line[spec]
            vec = np.zeros(nValues)
            for v in specs:
                vec[specDicts[v]] = 1.0
            dictItemSpecs[idx] = vec
            idx += 1
    prodSpecVecs = np.array(pd.DataFrame(dictItemSpecs)) # item * spec array
    specNorms = np.sqrt(specCounts).reshape((1, -1))
    similarityMat = np.divide( np.dot(prodSpecVecs, prodSpecVecs.T), np.dot(specNorms.T, specNorms))
    np.savetxt('similarity_'+spec+'.txt', similarityMat, fmt='%.6f')
                    
def querySpecSimilarity(spec, val1=None, val2=None):
    """show example of spec similarities"""
    specDicts = {}
    specLists = []
    with open('stat_'+spec+'.txt', 'r') as fin:
        idx = 0
        for line in fin:
            a, b = line.rstrip().split('\t')
            specDicts[a] = idx
            specLists.append(a)
            idx += 1
    smlMat = np.loadtxt('similarity_'+spec+'.txt')
    if val1 is None:
        val1 = input('specification value 1 (required): ')
        val2 = input('specification value 2 (optional): ')
        if not val2:
            val2 = None
    raws = smlMat[specDicts[val1], :]
    if val2 is not None:
        smlvalue = raws[specDicts[val2]]
    else:
        # output five most similar and five least similar
        orders = raws.argsort()
        mostsimilars = orders[-2:-7:-1] # skip self
        print('---'+val1+'---')
        print('Most similar:')
        for m in mostsimilars:
            print(' %s\t%f' % (specLists[m], raws[m]))
        leastsimilars = orders[:5]
        print('Least similar:')
        for m in leastsimilars:
            print(' %s\t%f' % (specLists[m], raws[m]))
        
if __name__ == '__main__':
    # generates similarity_ file
    
    # evalProductSpecs('tags')
    querySpecSimilarity('tags', 'Historical')
