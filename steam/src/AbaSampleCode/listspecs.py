# List specification of given property

from parse import parse

def listSpecs(property):
    """List specification values and count frequencies given a key.
Write stat_*.txt"""
    datahandle = parse('../data/steam_games_lite.json.gz')
    result = {}
    for line in datahandle:
        if property in line:
            specs = line[property]
            if isinstance(specs, list):
                for spec in specs:
                    if spec in result:
                        result[spec] += 1
                    else:
                        result[spec] = 1
            else:
                if specs in result:
                    result[specs] += 1
                else:
                    result[specs] = 1
    with open('stat_'+property+'.txt', 'w') as fout:
        for k in sorted(result):
            fout.write('%s\t%d\n' % (k, result[k]))

def readStat(spec):
    fname = 'stat_'+spec+'.txt'
    specCounts = []
    with open(fname, 'r') as fin:
        for line in fin:
            a, b = line.rstrip().split('\t')
            specCounts.append((a, b))
    return specCounts
    
if __name__ =='__main__':
    # listSpecs('specs')
    listSpecs('tags')
    # listSpecs('genres')
    # listSpecs('publisher')