#!/usr/bin/python

# Calculate the similarity based on user purchase data
# For two spec values q and q', the similarity is given by
# s_qq' = < q \cdot q' >_(|q||q'|)
# Thus removes the similarity caused by one product has two spec value at one time
# Will combine the similarity based on product specs exclusively

import numpy as np
import pandas as pd

from parse import parse
from listspecs import readStat
from filterUserItem import getItemObject

import comparecluster

class AllUsersProcessor(object):
    """Process the database"""
    def __init__(self, spec):
        self.spec = spec
        self.specNames = [a[0] for a in readStat(self.spec)]
        self.specDict = {name: idx for (idx, name) in enumerate(self.specNames)}
        # Load item map
        self.itemMap = comparecluster.readGames()
        self.traingames = comparecluster.readGameIdData('train')
        self.reducedItemMap = {}
        self.creatHandle()
        self.nspecval = len(self.specNames)
        # spec similarity matrix
        self.specSimSumMat = np.zeros((self.nspecval, self.nspecval))
        # count the sums
        self.specSimCountMat = np.zeros((self.nspecval, self.nspecval), dtype=int)
        
    def creatHandle(self):
        self.datahandle = parse('../../data/australian_users_items_lite.json.gz')
        
    def queryId(self, id):
        """Query by item id. return spec value indices"""
        if id not in self.reducedItemMap:
            if self.spec not in self.itemMap[id]:
                return None
            else:
                idxes = [self.specDict[val] for val in self.itemMap[id][self.spec] ]
            self.reducedItemMap[id] = idxes
        return self.reducedItemMap[id]
        
    def processUserItems(self, items):
        """Process one single user"""
        # count N_q
        valCountList = np.zeros(self.nspecval)
        # count N_{q q'}
        for oneitem in items:
            if oneitem not in self.traingames: continue
            specvals = self.queryId(oneitem)
            if specvals is not None:
                for idxp in specvals:
                    valCountList[idxp] += 1.0
        return valCountList
        
    def process(self):
        count = 0
        dictItemSpecs = {}
        for line in self.datahandle:
            items = [it['id'] for it in line['items']]
            vec = self.processUserItems(items)
            # if count > 1000: break # debug usage
            if count%100 == 0:
                print('%d\r'%(count), end='')
            dictItemSpecs[count] = vec
            count += 1
            #if count > 1000: break # debug
        prodSpecVecs = np.array(pd.DataFrame(dictItemSpecs)) # item * spec array
        print('prodSpecVecs shape: ', prodSpecVecs.shape)
        specNorms = np.sqrt(np.sum(np.multiply(prodSpecVecs, prodSpecVecs), axis=1)).reshape((1, -1))
        print('specNorms sum: ', np.sum(specNorms))
        self.similarityMat = np.nan_to_num(np.divide( np.dot(prodSpecVecs, prodSpecVecs.T), np.dot(specNorms.T, specNorms)))
        # Save to file
        np.savetxt('copurchaseSimilarity_'+self.spec+'.txt', self.similarityMat, fmt='%g')
    
if __name__ == '__main__':
    print('Processing products ...')
    processor = AllUsersProcessor('tags')
    print('Processing user records ...')
    processor.process()