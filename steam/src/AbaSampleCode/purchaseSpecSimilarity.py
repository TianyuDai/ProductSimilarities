#!/usr/bin/python

# Calculate the similarity based on user purchase data
# For two spec values q and q', the similarity is given by
# s_qq' = < 4 |Q| |Q' \ Q| / |Q \cup Q'| >_(|Q| \neq 0)
# Thus removes the similarity caused by one product has two spec value at one time
# Will combine the similarity based on product specs exclusively

import numpy as np

from parse import parse
from listspecs import readStat
from filterUserItem import getItemObject

class AllUsersProcessor(object):
    """Process the database"""
    def __init__(self, spec):
        self.spec = spec
        self.specNames = [a[0] for a in readStat(self.spec)]
        self.specDict = {name: idx for (idx, name) in enumerate(self.specNames)}
        # Load item map
        self.itemMap = getItemObject()
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
        valCountList = np.zeros((self.nspecval,))
        # count N_{q q'}
        valCountMat = np.zeros((self.nspecval, self.nspecval))
        allspecvals = set()
        for oneitem in items:
            specvals = self.queryId(oneitem)
            if specvals is not None:
                for idxp, rp in enumerate(specvals):
                    allspecvals.add(rp)
                    valCountList[rp] += 1
                    for rq in specvals[idxp+1:]:
                        valCountMat[rp, rq] += 1
                        valCountMat[rq, rp] += 1
        # now computing single user contribution
        for rp in allspecvals:
            for rq in range(self.nspecval):
                if rp == rq: continue
                if rq in allspecvals:
                    unionpq = valCountList[rp] + valCountList[rq] - valCountMat[rp, rq]
                    siab = 4.0*(valCountList[rp])*(valCountList[rq] - valCountMat[rp, rq])/(unionpq*unionpq)
                    self.specSimSumMat[rp, rq] += siab
                self.specSimCountMat[rp, rq] += 1
        
    def process(self):
        count = 0
        for line in self.datahandle:
            items = [it['id'] for it in line['items']]
            self.processUserItems(items)
            # if count > 1000: break # debug usage
            if count%100 == 0:
                print('%d\r'%(count), end='')
            count += 1
        # Take average
        for rp in range(self.nspecval):
            for rq in range(self.nspecval):
                if self.specSimCountMat[rp, rq] > 0:
                    self.specSimSumMat[rp, rq] /= self.specSimCountMat[rp, rq]
        # Save to file
        np.savetxt('purchaseSimilarity_'+self.spec+'.txt', self.specSimSumMat, fmt='%f')
    
if __name__ == '__main__':
    print('Processing products ...')
    processor = AllUsersProcessor('tags')
    print('Processing user records ...')
    processor.process()