"""
Determine R^2 score, MAE and accuracy of single hidden Markov model.
"""
import pandas as pd

import sdsModels as sdsm


data = pd.read_csv('../data/complete_april_2014.csv')
data['clustered-words-user'] = sdsm.cluster(data, ['words-user'], 4)

features = ['clustered-words-user', 'barge-in', 'attempt', 'SSA-ids']

values = []

exp = sdsm.Experiment(data=data)
exp.addModel(sdsm.Hmm({'states': [1, 2, 3, 4, 5]}))
exp.generateResults(features, cvMethod='kfolds', k=10)
for result in exp.results:
    r = result.getResults()
    row = [r['r2'], r['MAE'], r['accuracy']]
    values.append(row)


# plot results

df = pd.DataFrame(values, columns=['r2', 'MAE', 'accuracy'])
print(df.sort(['r2']))
