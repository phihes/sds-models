import sdsModels as sdsm
import pandas as pd

d_simple_1 = [
	(1, 1, 1),
	(1, 1, 1),
	(2, 2, 2),
	(2, 2, 2),
	(3, 3, 3),
	(3, 3, 3)
]

data = pd.DataFrame(d_simple_1, columns = ["label","rating","feature1"])
exp = sdsm.Experiment(data=data)
exp.addModel(sdsm.Hmm({'states': [1,2,3]}))
exp.generateResults(["feature1"], cvMethod="loo")

exp.printResults(['model', 'accuracy', 'r2'])

# bug:
# leave one label out -> emission "1" is completely ignored and does
# not show up in emission alphabet during maximum likelihood estimation. This
# means that the probability of emission is undefined.
# 
# solution: 
# calculate emission alphabet in a different way (on a "higher level"), so that
# not only training data is considered, but complete data.
