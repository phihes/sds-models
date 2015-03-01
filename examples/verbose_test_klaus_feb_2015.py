import pandas as pd
from sklearn.decomposition import PCA

import sdsModels as sdsm


data = pd.read_csv('../data/data_klaus_feb_2015.csv')
#data = data[pd.notnull(data['rating'])]
data['rating'] = data['rating'].fillna(4)
data['label'] = data['user'].astype(int).astype('str') + data['task'].astype(int).astype('str') + data['attempt'].astype(int).astype('str')
data.index = range(0, len(data))


testGMMHMM = True
testLogit = True
testHMM = True
testGNB = True
testMNB = True

fillNa = True
fillNaValue = -1

states = [1, 2, 3, 4, 5]
num_pc = 3
PCA_features = ["asr_confidence", "asr_time", "barge-in", "n_sys_words", "n_user_words", "sysTurnDuration", "userTurnDuration", "WER", "num_turns", "num_sorry", "SSA_sorry", "SSA_info", "SSA_confirm"]
num_mix = 1
cv_k = 10
HMM_features = ["asr_confidence", "userTurnDuration", "WER"]
PCA_features = HMM_features
HMM_k = 5

# PCA
for f in PCA_features:
    data[f] = data[f].fillna(fillNaValue)

pca = PCA(n_components=num_pc, whiten=True)
pca.fit(data[PCA_features])
pc = pca.transform(data[PCA_features])
pc_df = pd.DataFrame(pc, columns=["pc-" + str(num_pc) + "-" + str(i) for i in xrange(1, num_pc + 1)])
data = data.join(pc_df)

if fillNa:
    for f in HMM_features:
        data[f] = data[f].fillna(fillNaValue)


"""
Run experiment
"""

exp = sdsm.Experiment(data=data)
values = []


def addResults(v, r):
    for result in r:
        r = result.getResults()
        row = [r['model'], float(r['r2']), float(r['accuracy']), float(r['MAE'])]
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

if testLogit:
    exp.clear()
    features = ["pc-" + str(num_pc) + "-" + str(i) for i in xrange(1, num_pc + 1)]
    exp.addModel(sdsm.LogisticRegression({'class_weight': 'auto'}, verbose=True))
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

df = pd.DataFrame(values, columns=['model', 'r2', 'accuracy', 'MAE'])
print(df.sort(['r2'], ascending=False))