#!/usr/bin/python

# compare results of spectral clustering 

import numpy as np
import gzip, pickle, os

def readGameIdData(option=None):
    """Read game id data"""
    if option is None or option == 'full':
        fname_gameid = '../../data/complete_game_id_list.gz'
    elif option == 'train':
        fname_gameid = '../../data/train_game_id_list.gz'
    elif option == 'test':
        fname_gameid = '../../data/test_game_id_list.gz'
    with gzip.open(fname_gameid, 'r') as gfin:
        x = pickle.load(gfin)
    return x

def readGameCluLabels():
    """Read game cluster labels"""
    fname_gameclu = '../../data/complete_cluster_labels.npz'
    gamemat = np.load(fname_gameclu)
    return gamemat['clustering_labels_array']

def readSimData():
    """Read game similarity data"""
    fname_gamemat = '../../data/game_weighted_similarity_matrix.npz'
    # game similarities
    gamemat = np.load(fname_gamemat)
    return gamemat['game_similarity_matrix']

def gameClustering(mat, ncluster=10):
    from sklearn.cluster import SpectralClustering
    labels = SpectralClustering(ncluster).fit_predict(mat)
    return labels

def readGames():
    fname_games = '../../data/steam_games_lite.json.gz'
    fname_games_pkl = '../../data/steam_games_lite.pkl.gz'
    result = {}
    if not os.path.exists(fname_games_pkl):
        g = gzip.open(fname_games, 'r')
        for l in g:
            line = eval(l)
            try:
                idkey = line[u"id"] #.encode("utf-8"))
            except KeyError:
                # lack id
                continue
            line.pop(u"id", None)
            result[idkey] = line
        # save object for further use
        with gzip.open(fname_games_pkl, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        with gzip.open(fname_games_pkl, 'rb') as f:
            result = pickle.load(f)
    return result

def getTagSets():
    fname_tags = 'stat_tags.txt'
    tags = dict()
    with open(fname_tags, 'r') as fin:
        rp = 0
        for l in fin:
            tags[l.split('\t')[0]] = rp
            rp += 1
    return tags

def getGameSpecVec(specs, tagset):
    """get spec vector of a game"""
    result = np.zeros(len(tagset))
    for spec in specs:
        result[tagset[spec]] = 1.0
    return result

def getAllGameSpecVector():
    fname_allgamevec_pkl = '../../data/allgame_vecs.pkl.gz'
    result = {}
    if not os.path.exists(fname_allgamevec_pkl):  
        tagset = getTagSets()
        games = readGames()
        ngames = len(games)
        rp = 0
        for id in games:
            if 'tags' in games[id]:
                vecs = getGameSpecVec(games[id]['tags'], tagset)
                result[id] = vecs
                rp += 1
        with gzip.open(fname_allgamevec_pkl, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        with gzip.open(fname_allgamevec_pkl, 'rb') as f:
            result = pickle.load(f)
    return result

def testSimilarities(gameids, gamevec, clusterlabels, nsample=10):
    # input: list(str), dict(str: ndassay), list(int), int
    # return: [ ( [ cosine similarity in], [cosine similarity out], ...]
    nlabels = np.max(clusterlabels)
    result = [([], []) for i in range(nlabels+1)]
    ngames = len(gameids)
    for idxa in range(ngames):
        # randomly pick up two games
        ida = gameids[idxa]
        if ida not in gamevec: continue
        clua = clusterlabels[idxa]
        rq = 0
        while rq < nsample:
            idxb = np.random.randint(ngames)
            idb = gameids[idxb]
            if idb not in gamevec: continue
            club = clusterlabels[idxb]
            simrst = np.dot(gamevec[ida], gamevec[idb])
            if clua == club:
                result[clua][0].append(simrst)
            else:
                result[clua][1].append(simrst)
                result[club][1].append(simrst)
            rq += 1
    return result

if __name__ == '__main__':
    gameids = readGameIdData()
    gameclusters = readGameCluLabels()
    gamemat = readSimData()
    gameclusters_b = gameClustering(gamemat, 8)
    #print(len(gameids))
    #print(gamemat.shape)