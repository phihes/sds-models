import numpy as np
import math
from sklearn.metrics import *


class TestResults:
    """ Class in which performance metrics are defined and which can be used to calculate performance of true vs predicted series of judgments. Example usage:

        results = TestResults(model.name)
        for test,true in data:
            predicted = model.predict(test)
            results.compare(true,predicted)
        metrics = results.getResults()
        print('MAE of ' + model.name + ':')
        print(metrics['MAE'])
    """

    _indicators = {'MAE': 'mean', 'MSE': 'mean'}
    _finalInd = {'model', 'accuracy', 'r2'}  # ,'classification'}
    # 'scores', 'scores_f_1', 'scores_f_2', 'scores_f_3', 'scores_f_4', 'scores_f_5', 'scores_pr_1', 'scores_pr_2', 'scores_pr_3', 'scores_pr_4', 'scores_pr_5', 'scores_rec_1', 'scores_rec_2', 'scores_rec_3', 'scores_rec_4', 'scores_rec_5'
    _results = {}
    _true = list()
    _pred = list()
    _name = False

    def __init__(self, name, indicators=False):
        self._name = name
        self._true = list()
        self._pred = list()
        self._results = dict()
        if (indicators):
            self._indicators = indicators
        for i in self._indicators.keys():
            self._results[i] = list()
            self._results[i + '-last'] = list()

    def compare(self, true, pred):
        """ Add a series of true and predicted
        """
        self._true.append(true)
        self._pred.append(pred)
        for ind in self._indicators.keys():
            i = getattr(self, '_' + ind)
            self._results[ind].append(i(true, pred))
        self._compare_last(true, pred)

    def asLists(self):
        true = list()
        pred = list()
        for t in self._true:
            true = true + t

        for p in self._pred:
            pred = pred + p

        return true, pred


    def _compare_last(self, true, pred):
        last = len(true) - 1
        true = [true[last]]
        pred = [pred[last]]
        for ind in self._indicators.keys():
            i = getattr(self, '_' + ind)
            self._results[ind + '-last'].append(i(true, pred))

    def _MAE(self, true, pred):
        dist = 0.0
        for i in range(0, len(true)):
            dist += abs(float(true[i]) - float(pred[i]))
        return dist / (float(len(true)))


    def _MSE(self, true, pred):
        dist = 0.0
        for i in range(0, len(true)):
            dist += math.pow((float(true[i]) - float(pred[i])), 2)
        return dist / (float(len(true)))

    def _hits(self, true, pred):
        hits = 0
        count = 0
        for i in range(0, len(true)):
            if (true[i] == pred[i]):
                hits += 1
            count += 1
        return hits, count

    def _mean(self, values):
        return np.mean(values)

    def _hitRatio(self, values):
        h = 0
        c = 0
        for hits, count in values:
            h += hits
            c += count
        if (c > 0):
            return float(h) / float(c)
        else:
            return 0

    def _r2(self):
        true, pred = self.asLists()
        # print(self._name)
        # print(true)
        # print(pred)
        return str(r2_score(true, pred))

    def _accuracy(self):
        true, pred = self.asLists()
        return str(accuracy_score(true, pred))

    def _scores(self):
        true, pred = self.asLists()
        return map(list, precision_recall_fscore_support(true, pred,
                                                         labels=[1, 2, 3, 4, 5], average=None))

    def _scores_f_1(self):
        s = self._scores()
        return s[2][0]

    def _scores_f_2(self):
        s = self._scores()
        return s[2][1]

    def _scores_f_3(self):
        s = self._scores()
        return s[2][2]

    def _scores_f_4(self):
        s = self._scores()
        return s[2][3]

    def _scores_f_5(self):
        s = self._scores()
        return s[2][4]

    def _scores_pr_1(self):
        s = self._scores()
        return s[0][0]

    def _scores_pr_2(self):
        s = self._scores()
        return s[0][1]

    def _scores_pr_3(self):
        s = self._scores()
        return s[0][2]

    def _scores_pr_4(self):
        s = self._scores()
        return s[0][3]

    def _scores_pr_5(self):
        s = self._scores()
        return s[0][4]

    def _scores_rec_1(self):
        s = self._scores()
        return s[1][0]

    def _scores_rec_2(self):
        s = self._scores()
        return s[1][1]

    def _scores_rec_3(self):
        s = self._scores()
        return s[1][2]

    def _scores_rec_4(self):
        s = self._scores()
        return s[1][3]

    def _scores_rec_5(self):
        s = self._scores()
        return s[1][4]

    def _classification(self):
        true, pred = self.asLists()
        return "\"" + classification_report(true, pred) + "\""

    def _model(self):
        return self._name


    def __str__(self):
        s = "results for " + self._name + ":\n"
        for ind, aggr in self._results.items():
            aggr = self._indicators[ind.split('-')[0]]
            val = getattr(self, '_' + aggr)(self._results[ind])
            s += "\t" + ind + "\t" + str(val) + "\n"
        return s

    def getResults(self, getHeader=False):
        iMap = dict()
        for ind, aggr in self._results.items():
            aggr = self._indicators[ind.split('-')[0]]
            iMap[ind] = str(getattr(self, '_' + aggr)(self._results[ind]))

        for i in self._finalInd:
            iMap[i] = str(getattr(self, '_' + i)())

        return iMap

    def __repr__(self):
        return str(len(self._true))
