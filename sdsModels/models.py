import sklearn.cross_validation as cv
import sklearn.dummy as dummy
from sklearn.mixture import GMM
from sklearn.hmm import GMMHMM
from sklearn import linear_model
import collections
import itertools
import pandas as pd

from testResults import TestResults
from counters import *
import utils as utils


class Model():
    __metaclass__ = ABCMeta

    params = {}
    isSklearn = True

    def __init__(self, params, verbose=False):
        self.params = params
        self.verbose = verbose

    def printv(self, arg, title=None):
        if self.verbose:
            if title is not None:
                print title
            print arg

    @property
    def name(self):
        return self._name

    @abstractmethod
    def _train(self, data):
        """Returns a trained model."""
        pass

    def _test(self, model, testData, resultObj):
        """Compares predictions made by specified model against test data.
        Returns a TestResults object.
        """

        # restrict test data to principal component features
        features = self.params['features']
        test = np.array(testData[features])

        # predict a dialog sequence using test data
        # sklearn counts from 0 so add 1...
        if self.isSklearn:
            pred = [int(r) + 1 for r in list(model.predict(test))]
        else:
            pred = [int(r) for r in list(model.predict(test))]

        # extract true ratings from test data
        true = [int(rating) for rating in testData['rating'].values.tolist()]

        resultObj.compare(true, pred)

        return resultObj

    def loocv(self, data):
        """Leave-one-out cross validation using given data.
        Returns a TestResults objects, where results are averages from the
        cross validation steps.
        """
        mask = cv.LeaveOneLabelOut(data['label'].values)
        results = TestResults(self.name, verbose=self.verbose)

        for trainMask, testMask in mask:

            # training
            trainingData = data.loc[trainMask]
            self.printv(trainingData, "training data:")
            model = self._train(trainingData)

            # testing
            testData = data.loc[testMask]
            self.printv(testData, "test data:")
            # leave p labels out
            for label, testGroup in testData.groupby("label"):
                results = self._test(model, testGroup, results)

        return results

    def kfoldscv(self, data, folds):
        """K-folds cross validation using given data and number of folds.
        Returns a TestResults objects, where results are averages from the
        cross validation steps.
        """
        results = TestResults(self.name, verbose=self.verbose)
        labels = list(np.unique(data['label'].values))

        for tr, te in cv.KFold(len(labels), n_folds=folds):
            trainD = data[data['label'].isin([labels[i] for i in tr])]
            testD = data[data['label'].isin([labels[i] for i in te])]
            self.printv(trainD, "training data:")
            self.printv(testD, "test data:")

            model = self._train(trainD)

            for label, testGroup in testD.groupby("label"):
                results = self._test(model, testGroup, results)

        return results

    def setFeatures(self, features):
        self.params['features'] = features


class Dummy(Model):
    _name = "dummy"

    def _train(self, data):
        if 'constant' in self.params.keys():
            model = dummy.DummyClassifier(strategy=self.params['strategy'],
                                          constant=self.params['constant'])
        else:
            model = dummy.DummyClassifier(strategy=self.params['strategy'])
        d = np.array(zip(*[data[f].values for f in self.params['features']]))
        y = np.array(data['rating'].values)
        model.fit(d, y)

        return model


class Gmm(Model):
    """A Gaussian mixture model.
    Parameters are number of mixture components (num_mixc) and
    covariance type (cov_type). Example:
    model = Gmm(params = {num_mixc: 3,
                          cov_type:'diag'})
    """

    _name = "GMM"

    def _train(self, data):
        """Trains a Gaussian mixture model, using the sklearn implementation."""

        # parameters
        features = self.params['features']
        num_mixc = self.params['num_mixc']
        cov_type = self.params['cov_type']

        # prepare data shape
        d = np.array(zip(*[data[f].values for f in features]))

        # choose high number of EM-iterations to get constant results
        gmm = GMM(num_mixc, cov_type, n_iter=300)
        gmm.fit(d)

        return gmm


class Gmmhmm(Model):
    """A hidden Markov model with Gaussian mixture emissions.
    Parameters are number of mixture components (num_mixc), covariance type
    (cov_type) and states (states). One Gaussian mixture model is created for
    each state. Example:
    model = Gmmhmm(params = {'num_mixc': 3,
                             'cov_type': 'diag',
                             'states': [1,2,3,4,5]})
    """

    _name = "GMM-HMM"

    def _train(self, data):
        """Trains a GMMHMM model, using the sklearn implementation and maximum-
        likelihood estimates as HMM parameters (Hmm.mle(...)).
        """

        # parameters
        features = self.params['features']
        num_mixc = self.params['num_mixc']
        cov_type = self.params['cov_type']
        states = self.params['states']

        # train one GMM for each state
        mixes = list()
        for state in states:
            # select data with current state label
            d = data[data.rating == state]
            # prepare data shape
            d = np.array(zip(*[d[f].values for f in features]))

            # init GMM
            gmm = GMM(num_mixc, cov_type)
            # train
            gmm.fit(d)
            mixes.append(gmm)

        # train HMM with init, trans, GMMs=mixes
        mle = Hmm.mle(MatrixCounterNoEmissions, data, states)

        model = GMMHMM(n_components=len(states), init_params='', gmms=mixes)
        model.transmat_ = mle.transition
        model.startprob_ = mle.initial

        return model


class Ols(Model):
    """ Ordinary least squares regression """

    _name = "OLS"
    isSklearn = True

    def _train(self, data):
        features = self.params['features']
        X = np.array(zip(*[data[f].values for f in features]))
        y = np.array(data['rating'])
        model = linear_model.LinearRegression()
        model.fit(X, y)

        return model


class Hmm(Model):
    """A hidden Markov model, using the Nltk implementation and maximum-
    likelihood parameter estimates.
    """

    _name = "HMM"
    isSklearn = False
    Parameters = collections.namedtuple(
        'Parameters', 'initial transition emission emissionAlph')

    class NltkWrapper():

        def __init__(self, states, mle):
            self.model = nltk.HiddenMarkovModelTagger(mle.emissionAlph,
                                                      states,
                                                      mle.transition,
                                                      mle.emission,
                                                      mle.initial)

        def predict(self, obs):
            tagged = self.model.tag([tuple(o) for o in obs])
            return [val[1] for val in tagged]

    def _train(self, data):
        features = self.params['features']
        states = self.params['states']

        # calculate maximum-likelihood parameter estimates
        mle = Hmm.mle_multipleFeatures(NltkCounter, data, states, features)

        # create nltk HMM
        model = Hmm.NltkWrapper(states, mle)

        return model

    @staticmethod
    def mle(counterClass, data, stateAlphabet, feature=False):
        """ Calculate maximum likelihood estimates for the HMM parameters
        transitions probabilites, emission probabilites, and initial state
        probabilites.
        """

        f = feature is not False
        states = utils.dfToSequences(data, ['rating'])

        if (f):
            emissionAlphabet = pd.unique(data[feature].values.ravel())
            emissions = utils.dfToSequences(data, [feature])
        else:
            emissionAlphabet = None

        counter = counterClass(stateAlphabet, emissionAlphabet, states)

        # count for each state sequence
        for k, seq in enumerate(states):
            if (f): emi = emissions[k]
            # for each state transition
            for i, current in enumerate(seq):
                # count(current, next, first, emission)
                if (f):
                    emission = emi[i]
                else:
                    emission = False
                next = seq[i + 1] if i < len(seq) - 1 else False
                counter.count(i, current, next, emission)

        return Hmm.Parameters(
            initial=counter.getInitialProb(),
            transition=counter.getTransitionProb(),
            emission=counter.getEmissionProb(),
            emissionAlph=emissionAlphabet
        )

    @staticmethod
    def mle_multipleFeatures(counterClass, data, stateAlphabet, features):
        """ Calculate maximum likelihood estimates of HMM parameters.
        Parameters are transition probabilites, emission probabilites and
        initial sta<te probabilites.
        This method allows specifing multiple features and combines multiple
        emission features assuming conditional independence:
        P(feat1=a & feat2=b|state) = P(feat1=a|state) * P(feat2=b|state)
        """

        p = lambda feat: Hmm.mle(DictCounter, data, stateAlphabet, feat)
        counter = counterClass(stateAlphabet, [], False)

        # calculate conditional probabilites for each feature & corresponding
        # emission alphabet entry..
        # P(feat_i=emm_ij|state_k) forall: I features, J_i emissions, K states
        # ps = {feature:emmission distribution}
        emission_probs = [p(f).emission for f in features]

        # calculate inital state probabilites, transition probabilites using
        # first/any feature
        mle_single = Hmm.mle(counterClass, data, stateAlphabet, features[0])
        initial_probs = mle_single.initial
        transition_probs = mle_single.transition

        # combine the emission alphabets of all given features
        emissionAlphabet = list()
        for f in features:
            emissionAlphabet.append(pd.unique(data[f].values.ravel()))

        # calculate all emission combinations
        # and according probabilities per state
        for comb in list(itertools.product(*emissionAlphabet)):
            counter.addEmissionCombination(tuple(comb))
            for state in stateAlphabet:
                # for each individual prob of each feature
                for emission, featNum in zip(comb, xrange(0, len(emission_probs))):
                    prob = emission_probs[featNum][state][emission]
                    counter.addCombinedEmissionProb(state, tuple(comb), prob)

        return Hmm.Parameters(
            initial=initial_probs,
            transition=transition_probs,
            emission=counter.getCombinedEmissionProb(),
            emissionAlph=counter.getEmissionCombinations()
        )
