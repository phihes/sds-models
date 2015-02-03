"""
Determine R^2 score of hidden Markov model using clustered features. Parameters are
different feature-sets (looks at all combinations of given features) and ks for k-means clustering.
Clustering is done per each feature in feature-set. Shows result in a scatter-plot grid.
"""

import pandas as pd
import itertools

import seaborn as sns
import matplotlib.pyplot as plt

import sdsModels as sdsm


data = pd.read_csv('../data/complete_april_2014.csv')

values = []
all_features = [
    "asr-conf",
    "words-user",
    "barge-in",
    "words-system",
    "start",
    "attempt",
    "SSA-ids"]

for num_features in xrange(1, len(all_features)):

    for features in itertools.combinations(all_features, num_features):

        f = list(features)
        print(f)

        # k-means clustering for k=4..12
        for k in xrange(4, 13):

            print("\tk=" + str(k))

            # cluster features individually
            clustered_names = list()
            for feat in f:
                clustered_feat_name = 'clustered-' + feat
                data[clustered_feat_name] = sdsm.cluster(data, feat, k)
                clustered_names.append(clustered_feat_name)
            exp = sdsm.Experiment(data=data)
            exp.clear()
            exp.addModel(sdsm.Hmm({'states': [1, 2, 3, 4, 5]}))
            exp.generateResults(clustered_names, cvMethod='kfolds', k=2)
            for result in exp.results:
                r = result.getResults()
                row = [k, r['r2'], ', '.join(f), str(num_features) + " features"]
                values.append(row)


# plot results

df = pd.DataFrame(values, columns=['k', 'r2', 'features', '# features'])
print(df.sort(['r2']))

grid = sns.FacetGrid(df, row="# features", col="features", margin_titles=True)
grid.map(plt.scatter, "k", "r2")
grid.fig.tight_layout(w_pad=1)

plt.show()
