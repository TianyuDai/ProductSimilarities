#!/usr/bin/python

# One time data analyses
import numpy as np

testchoice = 1

if testchoice is 1:
    # List most similar specs by cosine similarity / purchase similarity
    from listspecs import readStat as readStat
    spec = 'tags'
    specnames = [a[0] for a in readStat(spec)] # spec value names
    datafname = 'purchaseSimilarity_'+spec+'.txt' # similarity_ or purchaseSimilarity_
    mat = np.loadtxt(datafname) # similar matrix
    similarities = []
    for rp in range(mat.shape[0]):
        for rq in range(rp+1, mat.shape[1]):
            val = mat[rp, rq]
            if val > 0.:
                similarities.append( (val, rp, rq) )
    similarities.sort(reverse = True)
    for rp in range(50):
        val, i, j = similarities[rp]
        print('%s\t%s\t%f' % (specnames[i], specnames[j], val))
elif testchoice is 2:
    pass
elif testchoice is 3:
    pass
elif testchoice is 4:
    pass