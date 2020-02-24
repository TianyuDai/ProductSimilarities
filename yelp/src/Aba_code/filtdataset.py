#!/usr/bin/python

import json, gzip
from parse import loader

"""filter dataset"""


def filterBusiness(businessfname, category):
    datahandle = loader(businessfname)
    categories = 'categories'
    count = 0
    countall = 0
    with gzip.open('business_'+category+'.json.gz', 'wb') as fout:
        for line in datahandle:
            record = json.loads(line)
            if categories in record and record[categories] is not None:
                specs = [s.strip() for s in record[categories].split(',')]
                if category in specs:
                    fout.write(line)
                    count += 1
            countall += 1
            if countall%10000==0: print(f'{count}:{countall}')
    print(f'{count} records in total')

if __name__ == '__main__':
    filterBusiness('../../data/business.json.gz', 'Restaurants')