"""
Compare cross-validation performance of Gaussian mixture model (GMM) and
hidden Markov model with Gaussian mixture emissions (GMM-HMM), using principal components
as features. Parameters are number of principal components and number of Gaussians to use
in mixtures. The CV-scores for both models are calculated for a range of the two parameters,
the results are shown in a scatter-plot grid:
Number of principal components vs. number of Gaussians in mixtures vs. model accuracy.
"""

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import sdsModels as sdsm


exp = sdsm.Experiment(pathToData='../data/complete_april_2014.csv')

values = []

#
pc_range = xrange(1, 9)
mix_range = xrange(1, 16)

for num_pc in pc_range:
    for num_mix in mix_range:
        exp.clear()
        features = ["pc-" + str(num_pc) + "-" + str(i) for i in xrange(1, num_pc + 1)]
        exp.addModel(sdsm.Gmm({
            'num_mixc': num_mix,
            'cov_type': 'diag'
        }))

        exp.addModel(sdsm.Gmmhmm({
            'num_mixc': num_mix,
            'cov_type': 'diag',
            'states': [1, 2, 3, 4, 5]
        }))

        exp.generateResults(
            features,
            cvMethod='kfolds',
            k=2
        )

        for result in exp.results:
            r = result.getResults()
            row = [r['model'], r['accuracy'], num_pc, num_mix]
            values.append(row)

df = pd.DataFrame(values, columns=['model', 'accuracy', 'components', 'mixtures'])
print(df)

grid = sns.FacetGrid(df, row="components", col="model", margin_titles=True)
grid.map(plt.scatter, "mixtures", "accuracy")
grid.fig.tight_layout(w_pad=1)

plt.show()
