"""
Calculates k-fold CV scores of OLS using principal component features.
"""
import pandas as pd

import sdsModels as sdsm


data = pd.read_csv('../data/complete_april_2014_ratings-latest.csv')
data = data[pd.notnull(data['rating'])]

values = []

num_pc = 2
all_features = ["pc-" + str(num_pc) + "-" + str(i) for i in xrange(1, num_pc + 1)]

exp = sdsm.Experiment(data=data)
exp.addModel(sdsm.Ols({}))
exp.generateResults(all_features, cvMethod='kfolds', k=10)
for result in exp.results:
    r = result.getResults()
    row = [r['r2'], r['MAE'], ', '.join(all_features), r['model']]
    values.append(row)


# plot results

df = pd.DataFrame(values, columns=['r2', 'MAE', 'features', 'model'])
print(df.sort(['r2']))
