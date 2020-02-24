#!/usr/bin/python

import os, sys
from parse import parse

"""Stat categories of business"""

def listSpecs(businessfname, filters={}):
    datahandle = parse(businessfname)
    categories = 'categories' # need split
    speclists = ['city', 'state', 'is_open', 'review_count']
    result = {i:{} for i in speclists}
    result[categories] = {}
    hasresult = {i:0 for i in result}
    nrecord = 0
    nanalyze = 0
    for line in datahandle:
        nrecord += 1
        filterflag = True
        for fkey in filters:
            if fkey not in line or line[fkey] != filters[fkey]:
                filterflag = False
                break
        if not filterflag: continue
        if categories in line and line[categories] is not None:
            hasresult[categories] += 1
            specs = [s.strip() for s in line[categories].split(',')]
            for spec in specs:
                if spec in result[categories]:
                    result[categories][spec] += 1
                else:
                    result[categories][spec] = 1
        for onespec in speclists:
            if onespec in line and line[onespec] is not None:
                hasresult[onespec] += 1
                spec = line[onespec]
                if onespec == 'city':
                    # fix same city name in different state
                    spec = spec+', '+line['state']
                elif onespec == 'review_count':
                    if spec >= 1000:
                        spec = (spec//1000)*1000

                if spec in result[onespec]:
                    result[onespec][spec] += 1
                else:
                    result[onespec][spec] = 1
        nanalyze += 1
        if nanalyze%10000==0: print(nanalyze)
        #if nanalyze>100000: break
    if not os.path.exists('stat'):
        os.mkdir('stat')
    with open('stat/stat_business.txt', 'w') as fout:
        fout.write(f'Total business: {nrecord}\n')
        if filters:
            fout.write(f'Business analyzed: {nrecord}\n')
        for akey in sorted(hasresult.keys()):
            fout.write(f'{akey}\t{hasresult[akey]}\n')
    for onespec in result:
        with open('stat/stat_business_'+onespec+'.txt', 'w') as fout:
            for akey in sorted(result[onespec].keys()):
                fout.write(f'{akey}\t{result[onespec][akey]}\n')

if __name__ == '__main__':
    listSpecs('../../data/business.json.gz')