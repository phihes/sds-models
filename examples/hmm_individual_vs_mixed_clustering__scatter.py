"""
Determine R^2 score of hidden Markov model using clustered features. Parameters are
different feature-sets, whether to cluster feature-sets as whole or each feature in set individually,
and ks for k-means clustering. Shows result in a scatter-plot grid.
"""

import pandas as pd
import itertools

import seaborn as sns
import matplotlib.pyplot as plt

import sdsModels as sdsm


data = pd.read_csv('../data/complete_april_2014.csv')
data = data[pd.notnull(data['rating'])]

values = []
all_features = ["asr-conf", "words-user", "barge-in", "SSA-ids"]
# number of features to use for each feature-set
num_features = 2

for features in itertools.combinations(all_features, num_features):

    f = list(features)
    print(f)

    # k-means clustering for k=2..4
    for k in xrange(3, 9):

        print("\tk=" + str(k))

        # mix features in clustering:
        data['cluster'] = sdsm.cluster(data, f, k)
        exp = sdsm.Experiment(data=data)
        exp.clear()
        exp.addModel(sdsm.Hmm({'states': [1, 2, 3, 4, 5]}))
        exp.generateResults(['cluster'], cvMethod='kfolds', k=2)
        for result in exp.results:
            r = result.getResults()
            row = [k, float(r['r2']), float(r['MAE']), float(r['accuracy']),
                   ', '.join(f), 'mixed']
            values.append(row)

        # cluster features individually
        clustered_names = list()
        for feat in f:
            clustered_feat_name = 'clustered-' + feat
            data[clustered_feat_name] = sdsm.cluster(data, feat, k)
            clustered_names.append(clustered_feat_name)
        exp = sdsm.Experiment(data=data)
        exp.clear()
        exp.addModel(sdsm.Hmm({'states': [1, 2, 3, 4, 5]}))
        exp.generateResults(clustered_names, cvMethod='kfolds', k=3)
        for result in exp.results:
            r = result.getResults()
            row = [k, float(r['r2']), float(r['MAE']), float(r['accuracy']),
                   ', '.join(f), 'individual']
            values.append(row)


# plot results

pd.set_option('display.precision', 4)
pd.set_option('display.width', 1024)
pd.set_option('display.max_rows', 512)

df = pd.DataFrame(values, columns=['k', 'r2', 'MAE', 'accuracy', 'features', 'clusters'])
print(df.sort(['r2'], ascending=False))

grid = sns.FacetGrid(df, row="clusters", col="features", margin_titles=True)
grid.map(plt.scatter, "k", "r2")
grid.fig.tight_layout(w_pad=1)

plt.show()
