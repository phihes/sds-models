"""
Determine R^2 score of hidden Markov model and ordinary least squares using clustered features. Parameters are
different feature-sets (looks at all combinations of given features) and ks for k-means clustering.
Clustering is done per each feature in feature-set. Shows results in a scatter-plot grid.
"""

import pandas as pd
import itertools
from scipy.misc import comb

import sdsModels as sdsm


fname = '../data/complete_april_2014.csv'

data = pd.read_csv(fname)
data = data[pd.notnull(data['rating'])]	

values = []
all_features = ["asr-conf", "words-user", "barge-in", "SSA-ids"]
num_features = 3
k_min = 2
k_max = 10
k_range = xrange(k_min, k_max + 1)
folds = 3

print("experiment with " + str(int(comb(len(all_features), num_features)))
      + " feature-combinations, individual clustering with k="
      + str(k_min) + ".." + str(k_max) + ", and " + str(folds) + "-folds CV")

for features in itertools.combinations(all_features, num_features):
    f = list(features)

    # k-means clustering for k=4..6
    for k in k_range:
        # cluster features individually
        clustered_names = list()
        for feat in f:
            clustered_feat_name = 'clustered-' + feat
            data[clustered_feat_name] = sdsm.cluster(data, feat, k)
            clustered_names.append(clustered_feat_name)
        exp = sdsm.Experiment(data=data)
        exp.clear()
        exp.addModel(sdsm.Hmm({'states': [1, 2, 3, 4, 5]}))
        exp.addModel(sdsm.Ols({}))
        exp.generateResults(clustered_names, cvMethod='kfolds', k=folds)
        for result in exp.results:
            r = result.getResults()
            row = [k, float(r['r2']), float(r['MAE']), float(r['accuracy']), ', '.join(f), r['model']]
            values.append(row)

# plot results

print('using ' + fname)

pd.set_option('display.precision', 4)
pd.set_option('display.width', 1024)
pd.set_option('display.max_rows', 512)

df = pd.DataFrame(values, columns=['k', 'r2', 'MAE', 'accuracy', 'features', 'model'])
print(df.sort(['r2'], ascending=False))

# grid = sns.FacetGrid(df, row="model", col="features", margin_titles=True)
#grid.map(plt.scatter, "k", "r2")
#grid.fig.tight_layout(w_pad=1)

#plt.show()
