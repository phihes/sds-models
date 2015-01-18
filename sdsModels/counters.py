from abc import ABCMeta, abstractmethod
import nltk

from utils import *


class Counter():
    __metaclass__ = ABCMeta

    def __init__(self, stateAlph, emissionAlph, states):
        # emission probabilites for each state and combination of emission symbols
        # e.g. s1:(f1=x,f2=x)->p1, s1:(f1=x,f2=y)->p2, s2:(f1=x,f2=x)->p3, ...
        self.stateAlph = stateAlph
        self.states = states
        self.emissionAlph = emissionAlph
        self.emissionPs = {state: dict() for state in stateAlph}

    @abstractmethod
    def count(self, current, next, position, emission):
        pass

    @abstractmethod
    def getInitialProb(self):
        pass

    @abstractmethod
    def getTransitionProb(self):
        pass

    @abstractmethod
    def getEmissionProb(self):
        pass

    @abstractmethod
    def getEmissionProb(self, state, emission):
        pass

    def checkState(self, state):
        if not (isinstance(state, bool) or state in self.stateAlph):
            raise Exception("Encountered unknown state in data: " + str(state))

    def addEmissionCombination(self, combination):
        self.emissionAlph.append(combination)

    def getEmissionCombinations(self):
        return self.emissionAlph

    def addCombinedEmissionProb(self, state, combination, p):
        if combination in self.emissionPs[state]:
            self.emissionPs[state][combination] *= p
        else:
            self.emissionPs[state][combination] = p

    @abstractmethod
    def getCombinedEmissionProb(self):
        pass


class MatrixCounterNoEmissions(Counter):
    def __init__(self, stateAlph, emissionAlph, states):
        super(MatrixCounterNoEmissions, self).__init__(
            stateAlph, emissionAlph, states)
        self.init, self.states_count, self.trans_abs = dicts(3, stateAlph)
        self.trans_ind, self.transitions = squareDicts(2, stateAlph)

    def count(self, position, current, next, emission):
        self.checkState(current)
        self.checkState(next)
        # absolute
        self.states_count[current] += 1
        # inits
        if position == 0:
            self.init[current] += 1
        # transitions
        if (next is not False):
            self.trans_abs[current] += 1
            self.trans_ind[current][next] += 1

    def getInitialProb(self):
        pr_init = {
            state: float(self.init[state]) / float(len(self.states))
            for state in self.init
        }

        return pr_init.values()

    def getTransitionProb(self):
        pr_trans = squareDict(self.stateAlph)
        for state in self.stateAlph:
            for otherState in self.stateAlph:
                pr_trans[state][otherState] = (
                    float(self.trans_ind[state][otherState]) /
                    float(self.trans_abs[state])
                )
        return matrix(pr_trans)

    def getEmissionProb(self):
        return None

    def getCombinedEmissionProb(self):
        return None

    def addEmissionCombination(self, combination):
        pass

    def getEmissionCombinations(self):
        return None

    def addCombinedEmissionProb(self, state, combination, p):
        pass


class MatrixCounter(Counter):
    def __init__(self, stateAlph, emissionAlph, states):
        super(MatrixCounter, self).__init__(
            stateAlph, emissionAlph, states)
        self.init, self.states_count, self.trans_abs = dicts(3, stateAlph)
        self.trans_ind, self.transitions = squareDicts(2, stateAlph)
        self.emissions_count = rectDict(stateAlph, emissionAlph)

    def count(self, position, current, next, emission):
        self.checkState(current)
        self.checkState(next)
        # absolute
        self.states_count[current] += 1
        # inits
        if position == 0:
            self.init[current] += 1
        # transitions
        if next is not False:
            self.trans_abs[current] += 1
            self.trans_ind[current][next] += 1
        # emissions
        if emission is not False:
            self.emissions_count[current][emission] += 1

    def getInitialProb(self):
        pr_init = {
            state: float(self.init[state]) / float(len(self.states))
            for state in self.init
        }

        return pr_init.values()

    def getTransitionProb(self):
        pr_trans = squareDict(self.stateAlph)
        for state in self.stateAlph:
            for otherState in self.stateAlph:
                pr_trans[state][otherState] = (
                    float(self.trans_ind[state][otherState]) /
                    float(self.trans_abs[state])
                ) if self.trans_abs[state] > 0 else 0
        return matrix(pr_trans)

    def getEmissionProb(self):
        pr_emissions = rectDict(self.stateAlph, self.emissionAlph)
        for state in self.stateAlph:
            for emission in self.emissionAlph:
                p = float(self.emissions_count[state][emission]) / float(self.states_count[state])
                pr_emissions[state][emission] = p

        return matrix(pr_emissions)

    def getCombinedEmissionProb(self):
        raise NotImplementedError


class DictCounter(MatrixCounter):
    def getEmissionProb(self):
        pr_emissions = rectDict(self.stateAlph, self.emissionAlph)
        for state in self.stateAlph:
            for emission in self.emissionAlph:
                if self.states_count[state] > 0:
                    p = float(self.emissions_count[state][emission]) / float(self.states_count[state])
                else:
                    p = 0
                pr_emissions[state][emission] = p

        return pr_emissions


class NltkCounter(Counter):
    roundnum = 100000000.0

    def __init__(self, stateAlph, emissionAlph, states):
        super(NltkCounter, self).__init__(
            stateAlph, emissionAlph, states)
        self.prior = nltk.FreqDist()
        self.transitions = nltk.ConditionalFreqDist()
        self.emissions = nltk.ConditionalFreqDist()

    def count(self, position, current, next, emission):
        self.checkState(current)
        self.checkState(next)
        self.emissions[current].inc(emission)
        if (position == 0): self.prior.inc(current)
        if (next is not False): self.transitions[current].inc(next)

    def getEmissionProb(self, state, emission):
        return float(self.emissions[state][emission]) / self.roundnum

    def getInitialProb(self):
        return nltk.LaplaceProbDist(self.prior)

    def getTransitionProb(self):
        return nltk.ConditionalProbDist(self.transitions, nltk.LaplaceProbDist)

    def getEmissionProb(self):
        return nltk.ConditionalProbDist(self.emissions, nltk.LaplaceProbDist)

    def getCombinedEmissionProb(self):
        allEmissions = nltk.ConditionalFreqDist()
        for state in self.emissionPs.keys():
            print state
            for combination in self.emissionPs[state].keys():
                print combination
                count = round(self.emissionPs[state][combination] * self.roundnum)
                print count
                allEmissions[state].inc(combination, count)

        print allEmissions

        return nltk.ConditionalProbDist(allEmissions, nltk.LaplaceProbDist)
