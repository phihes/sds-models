import numpy as np
from scipy.cluster.vq import whiten, kmeans, vq


def dfToSequences(data, params, rated=False):
    data = data.groupby(['label'])
    seqs = list()

    for name, group in data:
        vals = list()
        for p in params:
            vals.append(group[p].values.tolist())

        features = map(None, *vals)

        if (rated):
            ratings = group['rating'].values.tolist()
            features = map(None, *[features, ratings])

        seqs.append(features)

    return seqs


def squareDict(keyList, fill=0):
    return rectDict(keyList, keyList, fill)


def rectDict(keyListA, keyListB, fill=0):
    return {a: {b: fill for b in keyListB} for a in keyListA}


def squareDicts(num, keyList, fill=0):
    return [squareDict(keyList, fill) for _ in range(num)]


def rectDicts(num, keyListA, keyListB, fill=0):
    return [rectDict(keyListA, keyListB, fill) for _ in range(num)]


def dicts(num, keyList, fill=0):
    return [{a: fill for a in keyList} for _ in range(num)]


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
