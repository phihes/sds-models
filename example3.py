import sdsModels as sdsm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import itertools

data = pd.read_csv('complete_april_2014_ratings-latest.csv')
data = data[pd.notnull(data['rating'])]	

values = []
all_features = ["asr-conf", "words-user", "barge-in", "SSA-ids"]
num_features = 3

for features in itertools.combinations(all_features, num_features):

	f = list(features)
	print(f)

	# k-means clustering for k=2..4
	for k in xrange(3,9):
	
		print("\t" + str(k))

		# mix features in clustering:
		data['cluster'] = sdsm.cluster(data, f, k)
		exp = sdsm.Experiment(data=data)
		exp.clear()
		exp.addModel(sdsm.Hmm({'states':[1,2,3,4,5]}))
		exp.generateResults(['cluster'], cvMethod='kfolds', k=2)
		for result in exp.results:
			r = result.getResults()
			row = [k, r['r2'], ', '.join(f), 'mixed']
			values.append(row)
			
		# cluster features individually
		clustered_names = list()
		for feat in f:
			clustered_feat_name = 'clustered-'+feat
			data[clustered_feat_name] = sdsm.cluster(data, feat, k)
			clustered_names.append(clustered_feat_name)
		exp = sdsm.Experiment(data=data)
		exp.clear()
		exp.addModel(sdsm.Hmm({'states':[1,2,3,4,5]}))
		exp.generateResults(clustered_names, cvMethod='kfolds', k=2)
		for result in exp.results:
			r = result.getResults()
			row = [k, r['r2'], ', '.join(f), 'individual']
			values.append(row)			
		
		
# plot results

df = pd.DataFrame(values, columns=['k','r2', 'features', 'clusters'])
print(df.sort(['r2']))

grid = sns.FacetGrid(df, row="clusters", col="features", margin_titles=True)
grid.map(plt.scatter, "k", "r2")
grid.fig.tight_layout(w_pad=1)

plt.show()
