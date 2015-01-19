import pandas as pd


class Experiment:
    data = False
    models = []
    results = []

    def __init__(self, pathToData=None, data=None):
        self.clear()
        if isinstance(pathToData, basestring):
            self.data = pd.read_csv(pathToData)
            # clear of unrated turns
            self.data = self.data[pd.notnull(self.data['rating'])]
        elif isinstance(data, pd.DataFrame):
            self.data = data
        else:
            raise Exception("No data has been specified for the experiment.")

    def addModel(self, model):
        self.models.append(model)

    def generateResults(self, features, cvMethod, k=10):
        self.results = []
        for model in self.models:
            model.setFeatures(features)
            if cvMethod == 'loo':
                print "running leave-one-out cross-validation"
                result = model.loocv(self.getLabeledData(features))
            else:
                print "running " + str(k) + "-fold cross-validation"
                result = model.kfoldscv(self.getLabeledData(features), k)
            self.results.append(result)

    def getLabeledData(self, features):
        return self.data[features + ["rating", "label"]]

    def getUnlabeledData(self, features):
        return self.data[features]

    def clear(self):
        self.models = []
        self.results = []

    def printResults(self, metrics, extraMetrics=None):
        df = self.resultsAsDf(metrics, extraMetrics)
        print(df)

    def resultsAsDf(self, metrics, extraMetrics=None):
        values = list()
        for result in self.results:
            r = result.getResults()
            row = list()
            for metric in metrics:
                row.append(r[metric])
            values.append(row)

        if extraMetrics is None:
            df = pd.DataFrame(values, columns=metrics)
        else:
            for i in xrange(0, len(values)):
                for cols in extraMetrics.values():
                    values[i].append(cols[i])
            df = pd.DataFrame(values, columns=metrics + extraMetrics.keys())

        return df