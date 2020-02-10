# calculate similarity and importance for a given spec
import os, sys
import numpy as np
import pandas as pd

import comparecluster

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
    idx = 0
    with open(fname, 'r') as fin:
        for line in fin:
            a, b = line.rstrip().split('\t')
            specDicts[a] = idx
            idx += 1
    specCounts = np.zeros(idx)
    nValues = idx
    # then stat products
    games = comparecluster.readGames()
    gameids = comparecluster.readGameIdData('train')
    dictItemSpecs = {}
    for gameid in gameids:
        if spec in games[gameid]:
            specs = games[gameid][spec]
            vec = np.zeros(nValues)
            for v in specs:
                vec[specDicts[v]] = 1.0
                specCounts[specDicts[v]] += 1.0
            dictItemSpecs[idx] = vec
            idx += 1
            #if idx%100==0: print(idx, len(gameids))
    prodSpecVecs = np.array(pd.DataFrame(dictItemSpecs)) # item * spec array
    specNorms = np.sqrt(specCounts).reshape((1, -1))
    similarityMat = np.nan_to_num(np.divide( np.dot(prodSpecVecs, prodSpecVecs.T), np.dot(specNorms.T, specNorms)))
    np.savetxt('similarity_'+spec+'.txt', similarityMat, fmt='%.6f')

def querySpecSimilarity(spec, prefix='similarity_', val1=None, val2=None):
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
    smlMat = np.loadtxt(prefix+spec+'.txt')
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
        print('---'+val1+'---')
        print('Most similar:')
        count = 0
        for m in orders[::-1]:
            if m == specDicts[val1]: continue
            print(' %s\t%f' % (specLists[m], raws[m]))
            count += 1
            if count >= 10: break
        print('Least similar:')
        count = 0
        for m in orders:
            if m == specDicts[val1]: continue
            print(' %s\t%f' % (specLists[m], raws[m]))
            count += 1
            if count >= 10: break
        
if __name__ == '__main__':
    # generates similarity_ file
    
    evalProductSpecs('tags')
    #querySpecSimilarity('tags', prefix='purchaseSimilarity_', val1='Nonlinear')
