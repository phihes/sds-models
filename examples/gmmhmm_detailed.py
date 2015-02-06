import pandas as pd

import matplotlib.pyplot as plt

import sdsModels as sdsm


data = pd.read_csv('../data/complete_april_2014.csv')
data = data[pd.notnull(data['rating'])]

exp = sdsm.Experiment(data=data)

values = []

#
pc_range = xrange(1, 2)
mix_range = xrange(1, 2)

for num_pc in pc_range:
    for num_mix in mix_range:
        exp.clear()

        features = ["pc-" + str(num_pc) + "-" + str(i) for i in xrange(1, num_pc + 1)]

        exp.addModel(sdsm.Gmmhmm({
            'num_mixc': num_mix,
            'cov_type': 'diag',
            'states': [1, 2, 3, 4, 5]
        }, verbose=True))

        exp.generateResults(
            features,
            cvMethod='kfolds',
            k=10
        )

        for result in exp.results:
            r = result.getResults()
        row = [float(r['r2']), float(r['accuracy']), num_pc, num_mix]
        values.append(row)

pd.set_option('display.precision', 4)
pd.set_option('display.width', 1024)
pd.set_option('display.max_rows', 512)

df = pd.DataFrame(values, columns=['r2', 'accuracy', 'components', 'mixtures'])
print(df.sort(['r2'], ascending=False))

plt.show()