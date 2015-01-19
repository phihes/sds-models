"""
Gaussian mixture model vs. hidden Markov model with Gaussian mixture emissions vs.
hidden Markov model.
"""

import sdsModels as sdsm

exp = sdsm.Experiment(pathToData='../data/complete_april_2014_ratings-latest.csv')
exp2 = sdsm.Experiment(pathToData='../data/complete_april_2014_ratings-latest.csv')

gmmhmm = sdsm.Gmmhmm({
    'num_mixc': 2,
    'cov_type': 'diag',
    'states': [1, 2, 3, 4, 5]
})
exp.addModel(gmmhmm)

gmm = sdsm.Gmm({
    'num_mixc': 2,
    'cov_type': 'diag'
})
exp.addModel(gmm)

# create hidden Markov model

hmm = sdsm.Hmm({
    'states': [1, 2, 3, 4, 5]
})
exp2.addModel(hmm)

# exp.addModel(sdsm.Dummy({'strategy':'constant', 'constant': 3.0}))


# run experiment
num_pc = 3
features = ["pc-" + str(num_pc) + "-" + str(i) for i in xrange(1, num_pc + 1)]
exp.generateResults(features, cvMethod='kfolds', k=10)
features2 = ['words-system', 'asr-conf']
exp2.generateResults(features2, cvMethod='kfolds', k=10)

# output results
exp.printResults(['model', 'accuracy', 'r2'])
exp2.printResults(['model', 'accuracy', 'r2'])
