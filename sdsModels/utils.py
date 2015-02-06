import numpy as np
from scipy.cluster.vq import whiten, kmeans, vq


def printDictProbDist(dist):
    for sample in dist.samples():
        print(str(sample) + ": " + str(dist.prob(sample)))

def printCondDictProbDist(dist):
    for c in dist.conditions():
        print("probs for " + str(c))
        printDictProbDist(dist[c])


def dfToSequences(data, params, rated=False):
    data = data.groupby(['label'])
    seqs = list()

    for name, group in data:
        vals = list()
        for p in params:
            vals.append(group[p].values.tolist())

        features = map(None, *vals)

        if rated:
            ratings = group['rating'].values.tolist()
            features = map(None, *[features, ratings])

        seqs.append(features)

    return seqs


def squareDict(keylist, fill=0):
    return rectDict(keylist, keylist, fill)


def rectDict(keylista, keylistb, fill=0):
    return {a: {b: fill for b in keylistb} for a in keylista}


def squareDicts(num, keyList, fill=0):
    return [squareDict(keyList, fill) for _ in range(num)]


def rectDicts(num, keylista, keylistb, fill=0):
    return [rectDict(keylista, keylistb, fill) for _ in range(num)]


def dicts(num, keylist, fill=0):
    return [{a: fill for a in keylist} for _ in range(num)]


def matrix(lists):
    return [x.values() for x in lists.values()]


def cluster(data, features, k, iterations=100):
    # select only requested parameters for features
    obs = whiten(np.asarray(data[features]))

    # get centroids
    centroids, _ = kmeans(obs, k, iterations)

    # save'em
    # if isinstance(p, basestring):
    # self._centroids[p] = centroids
    # else:
    #	self._centroids['cluster'] = centroids

    # assign features to clusters
    index, _ = vq(obs, centroids)

    return index
