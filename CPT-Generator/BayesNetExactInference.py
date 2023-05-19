#############################################################################
# BayesNetExactInference.py
#
# This program implements the algorithm "Inference by Enumeration", which
# makes use of BayesNetsReader to facilitate reading data of a Bayes net via
# the object self.bn created by the inherited class (BayesNetReader). It also
# makes use of miscellaneous methods implemented in BayesNetUtil.
# Its purpose is to answer probabilistic queries such as P(Y|X=true,Z=false).
# This implementation is agnostic of the data and provides a general
# implementation that can ne used across datasets by providing a config file.
#
# WARNING: This code has not been thoroughly tested.
#
# Version: 1.0, Date: 06 October 2022, first version
# Version: 1.2, Date: 21 October 2022, revised version (more query compatible)
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import sys
import BayesNetUtil as bnu
from BayesNetReader import BayesNetReader


class BayesNeExactInference(BayesNetReader):
    query = {}
    prob_dist = {}

    def __init__(self):
        if len(sys.argv) != 3:
            print("USAGE: BayesNetInference.py [your_config_file.txt] [query]")
            print("EXAMPLE> BayesNetInference.py config-alarm.txt \"P(B|J=true,M=true)\"")
        else:
            file_name = sys.argv[1]
            prob_query = sys.argv[2]
            super().__init__(file_name)
            self.query = bnu.tokenise_query(prob_query)
            self.prob_dist = self.enumeration_ask()
            normalised_dist = bnu.normalise(self.prob_dist)
            print("unnormalised probability_distribution="+str(self.prob_dist))
            print("normalised probability_distribution="+str(normalised_dist))

    def enumeration_ask(self):
        print("\nSTARTING Inference by Enumeration...")
        Q = {}
        for value in self.bn["rv_key_values"][self.query["query_var"]]:
            value = value.split('|')[0]
            Q[value] = 0

        for value, probability in Q.items():
            value = value.split('|')[0]
            variables = self.bn["random_variables"].copy()
            evidence = self.query["evidence"].copy()
            evidence[self.query["query_var"]] = value
            probability = self.enumerate_all(variables, evidence)
            Q[value] = probability
            print("\tQ="+str(Q))

        return Q

    def enumerate_all(self, variables, evidence):
        print("\nCALL to enumerate_all(): V=%s E=%s" % (variables, evidence))
        if len(variables) == 0:
            return 1.0

        V = variables[0]

        if V in evidence:
            v = evidence[V].split('|')[0]
            p = bnu.get_probability_given_parents(V, v, evidence, self.bn)
            variables.pop(0)
            return p*self.enumerate_all(variables, evidence)

        else:
            sum = 0
            evidence_copy = evidence.copy()
            for v in bnu.get_domain_values(V, self.bn):
                evidence[V] = v
                p = bnu.get_probability_given_parents(V, v, evidence, self.bn)
                rest_variables = variables.copy()
                rest_variables.pop(0)
                sum += p*self.enumerate_all(rest_variables, evidence)
                evidence = evidence_copy

            return sum


BayesNeExactInference()
