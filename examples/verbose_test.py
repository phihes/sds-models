import pandas as pd

import matplotlib.pyplot as plt

import sdsModels as sdsm


data = pd.read_csv('../data/complete_april_2014_ratings-latest-in-turn-0-1.csv')
data = data[pd.notnull(data['rating'])]

testGMMHMM = True
testHMM = False
testGNB = False
testMNB = False

states = [1, 2]
num_pc = 1
num_mix = 1
cv_k = 10
HMM_features = ["asr-conf", "words-user", "barge-in", "SSA-ids"]
HMM_k = 5




exp = sdsm.Experiment(data=data)
values = []


def addResults(v, r):
    for result in r:
        r = result.getResults()
        row = [r['model'], float(r['r2']), float(r['accuracy'])]
        v.append(row)
    return v


if testGMMHMM:
    exp.clear()
    features = ["pc-" + str(num_pc) + "-" + str(i) for i in xrange(1, num_pc + 1)]
    exp.addModel(sdsm.Gmmhmm({
        'num_mixc': num_mix,
        'cov_type': 'diag',
        'states': states
    }, verbose=True))
    exp.generateResults(
        features,
        cvMethod='kfolds',
        k=cv_k
    )
    values = addResults(values, exp.results)

if testHMM:
    exp.clear()
    data["cluster"] = sdsm.cluster(data, HMM_features, HMM_k)
    exp.addModel(sdsm.Hmm({'states': states}, verbose=True))
    exp.generateResults(
        ["cluster"],
        cvMethod='kfolds',
        k=cv_k
    )
    values = addResults(values, exp.results)

if testGNB:
    exp.clear()
    features = ["pc-" + str(num_pc) + "-" + str(i) for i in xrange(1, num_pc + 1)]
    exp.addModel(sdsm.GaussianNaiveBayes({}, verbose=True))
    exp.generateResults(
        features,
        cvMethod='kfolds',
        k=cv_k
    )
    values = addResults(values, exp.results)

if testMNB:
    exp.clear()
    data["cluster"] = sdsm.cluster(data, HMM_features, HMM_k)
    exp.addModel(sdsm.MultinomialNaiveBayes({'alpha': 0.5, 'fit_prior': False}, verbose=True))
    exp.generateResults(
        ["cluster"],
        cvMethod='kfolds',
        k=cv_k
    )
    values = addResults(values, exp.results)

pd.set_option('display.precision', 4)
pd.set_option('display.width', 1024)
pd.set_option('display.max_rows', 512)

df = pd.DataFrame(values, columns=['model', 'r2', 'accuracy'])
print(df.sort(['r2'], ascending=False))

plt.show()