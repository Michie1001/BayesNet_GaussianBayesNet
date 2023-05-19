#############################################################################
# NB_Classifier.py
#
# Implements the Naive Bayes classifier for simple probabilistic inference.
# It assumes the existance of data in CSV format, where the first line contains
# the names of random variables -- the last being the variable to predict.
# This implementation aims to be agnostic of the data (no hardcoded vars/probs)
#
# WARNING: This code has not been thoroughly tested.
#
# Version: 1.0, Date: 03 October 2022, basic functionality
# Version: 1.5, Date: 15 October 2022, extended with performance metrics
# Version: 2.0, Date: 18 October 2022, extended with LL and BIC functions
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import sys
import math
import time
import numpy as np
from sklearn import metrics


class NB_Classifier:
    rand_vars = []
    rv_key_values = {}
    rv_all_values = []
    predictor_variable = None
    num_data_instances = 0
    default_missing_count = 0.000001
    probabilities = {}
    predictions = []
    training_time = None
    inference_time = None
    log_probabilities = False
    verbose = False

    def __init__(self, file_name, fitted_model=None):
        if file_name is None:
            return
        else:
            self.read_data(file_name)

        if fitted_model is None:
            self.training_time = time.time()
            self.estimate_probabilities()
            self.calculate_scoring_functions()
            self.training_time = time.time() - self.training_time

        else:
            self.rv_key_values = fitted_model.rv_key_values
            self.probabilities = fitted_model.probabilities
            self.training_time = fitted_model.training_time
            self.test_learnt_probabilities(file_name)
            self.compute_performance()

    def read_data(self, data_file):
        print("\nREADING data file %s..." % (data_file))
        print("---------------------------------------")

        self.rand_vars = []
        self.rv_key_values = {}
        self.rv_all_values = []

        with open(data_file) as csv_file:
            for line in csv_file:
                line = line.strip()
                if len(self.rand_vars) == 0:
                    self.rand_vars = line.split(',')
                    for variable in self.rand_vars:
                        self.rv_key_values[variable] = []
                else:
                    values = line.split(',')
                    self.rv_all_values.append(values)
                    self.update_variable_key_values(values)
                    self.num_data_instances += 1

        self.predictor_variable = self.rand_vars[len(self.rand_vars)-1]

        print("RANDOM VARIABLES=%s" % (self.rand_vars))
        print("VARIABLE KEY VALUES=%s" % (self.rv_key_values))
        print("VARIABLE VALUES (first 10)=%s" % (self.rv_all_values[:10]))
        print("PREDICTOR VARIABLE=%s" % (self.predictor_variable))
        print("|data instances|=%d" % (self.num_data_instances))

    def update_variable_key_values(self, values):
        for i in range(0, len(self.rand_vars)):
            variable = self.rand_vars[i]
            key_values = self.rv_key_values[variable]
            value_in_focus = values[i]
            if value_in_focus not in key_values:
                self.rv_key_values[variable].append(value_in_focus)

    def estimate_probabilities(self):
        countings = self.estimate_countings()
        prior_counts = countings[self.predictor_variable]

        print("\nESTIMATING probabilities...")
        for variable, counts in countings.items():
            prob_distribution = {}
            for key, val in counts.items():
                variables = key.split('|')

                if len(variables) == 1:
                    # prior probability
                    probability = float(val/self.num_data_instances)
                else:
                    # conditional probability
                    probability = float(val/prior_counts[variables[1]])

                if self.log_probabilities is False:
                    prob_distribution[key] = probability
                else:
                    # convert probability to log probability
                    prob_distribution[key] = math.log(probability)

            self.probabilities[variable] = prob_distribution

        for variable, prob_dist in self.probabilities.items():
            prob_mass = 0
            for value, prob in prob_dist.items():
                prob_mass += prob
            print("P(%s)=>%s\tSUM=%f" % (variable, prob_dist, prob_mass))

    def estimate_countings(self):
        print("\nESTIMATING countings...")

        countings = {}
        for variable_index in range(0, len(self.rand_vars)):
            variable = self.rand_vars[variable_index]

            if variable_index == len(self.rand_vars)-1:
                # prior counts
                countings[variable] = self.get_counts(None)
            else:
                # conditional counts
                countings[variable] = self.get_counts(variable_index)

        print("countings="+str(countings))
        return countings

    def get_counts(self, variable_index):
        counts = {}
        predictor_index = len(self.rand_vars)-1

        # accumulate countings
        for values in self.rv_all_values:
            if variable_index is None:
                # case: prior probability
                value = values[predictor_index]
            else:
                # case: conditional probability
                value = values[variable_index]+"|"+values[predictor_index]

            try:
                counts[value] += 1
            except Exception:
                counts[value] = 1

        # verify countings by checking missing prior/conditional counts
        if variable_index is None:
            counts = self.check_missing_prior_counts(counts)
        else:
            counts = self.check_missing_conditional_counts(counts, variable_index)

        return counts

    def check_missing_prior_counts(self, counts):
        for var_val in self.rv_key_values[self.predictor_variable]:
            if var_val not in counts:
                print("WARNING: missing count for variable=" % (var_val))
                counts[var_val] = self.default_missing_count

        return counts

    def check_missing_conditional_counts(self, counts, variable_index):
        variable = self.rand_vars[variable_index]
        for var_val in self.rv_key_values[variable]:
            for pred_val in self.rv_key_values[self.predictor_variable]:
                pair = var_val+"|"+pred_val
                if pair not in counts:
                    print("WARNING: missing count for variables=%s" % (pair))
                    counts[pair] = self.default_missing_count

        return counts

    def test_learnt_probabilities(self, file_name):
        print("\nEVALUATING on "+str(file_name))
        self.inference_time = time.time()

        # iterate over all instances in the test data
        for instance in self.rv_all_values:
            distribution = {}
            if self.verbose:
                print("Input vector=%s" % (instance))

            # iterate over all values in the predictor variable
            for predictor_value in self.rv_key_values[self.predictor_variable]:
                prob_dist = self.probabilities[self.predictor_variable]
                prob = prob_dist[predictor_value]

                # iterate over all instance values except the predictor var.
                for value_index in range(0, len(instance)-1):
                    variable = self.rand_vars[value_index]
                    value = instance[value_index]
                    prob_dist = self.probabilities[variable]
                    cond_prob = value+"|"+predictor_value

                    if self.log_probabilities is False:
                        prob *= prob_dist[cond_prob]
                    else:
                        prob += prob_dist[cond_prob]

                distribution[predictor_value] = prob

            normalised_dist = self.get_normalised_distribution(distribution)
            self.predictions.append(normalised_dist)
            if self.verbose:
                print("UNNORMALISED DISTRIBUTION=%s" % (distribution))
                print("NORMALISED DISTRIBUTION=%s" % (normalised_dist))
                print("---")

        self.inference_time = time.time() - self.inference_time

    def get_normalised_distribution(self, distribution):
        normalised_dist = {}
        prob_mass = 0
        for var_val, prob in distribution.items():
            prob = math.exp(prob) if self.log_probabilities is True else prob
            prob_mass += prob

        for var_val, prob in distribution.items():
            prob = math.exp(prob) if self.log_probabilities is True else prob
            normalised_prob = prob/prob_mass
            normalised_dist[var_val] = normalised_prob

        return normalised_dist

    def compute_performance(self):
        Y_true = []
        Y_pred = []
        Y_prob = []

        # obtain vectors of categorical and probabilistic predictions
        for i in range(0, len(self.rv_all_values)):
            target_value = self.rv_all_values[i][len(self.rand_vars)-1]
            if target_value == 'yes': Y_true.append(1)
            elif target_value == 'no': Y_true.append(0)
            elif target_value == '1': Y_true.append(1)
            elif target_value == '0': Y_true.append(0)

            predicted_output = self.predictions[i][target_value]
            Y_prob.append(predicted_output)

            best_key = max(self.predictions[i], key=self.predictions[i].get)
            if best_key == 'yes': Y_pred.append(1)
            elif best_key == 'no': Y_pred.append(0)
            elif best_key == '1': Y_pred.append(1)
            elif best_key == '0': Y_pred.append(0)

        P = np.asarray(Y_true)+0.00001 # constant to avoid NAN in KL divergence
        Q = np.asarray(Y_prob)+0.00001 # constant to avoid NAN in KL divergence

        # calculate metrics: accuracy, auc, brief, kl, training/inference times
        acc = metrics.balanced_accuracy_score(Y_true, Y_pred)
        fpr, tpr, _ = metrics.roc_curve(Y_true, Y_prob, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        brier = metrics.brier_score_loss(Y_true, Y_prob)
        kl_div = np.sum(P*np.log(P/Q))
        print("PERFORMANCE:")
        print("Balanced Accuracy="+str(acc))
        print("Area Under Curve="+str(auc))
        print("Brier Score="+str(brier))
        print("KL Divergence="+str(kl_div))
        print("Training Time="+str(self.training_time)+" secs.")
        print("Inference Time="+str(self.inference_time)+" secs.")

    def calculate_scoring_functions(self):
        print("\nCALCULATING LL and BIC on training data...")
        LL = self.calculate_log_lilelihood()
        BIC = self.calculate_bayesian_information_criterion(LL)
        print("LL score="+str(LL))
        print("BIC score="+str(BIC))

    def calculate_log_lilelihood(self):
        LL = 0
 
        # iterate over all instances in the training data
        for instance in self.rv_all_values:
            predictor_value = instance[len(instance)-1]

            # iterate over all random variables except the predictor var.
            for value_index in range(0, len(instance)-1):
                variable = self.rand_vars[value_index]
                value = instance[value_index]
                prob_dist = self.probabilities[variable]
                prob = prob_dist[value+"|"+predictor_value]
                LL += math.log(prob)

            # accumulate the log prob of the predictor variable
            prob_dist = self.probabilities[self.predictor_variable]
            prob = prob_dist[predictor_value]
            LL += math.log(prob)

            if self.verbose is True:
                print("LL: %s -> %f" % (instance, LL))

        return LL

    def calculate_bayesian_information_criterion(self, LL):
        penalty = 0

        for variable in self.rand_vars:
            num_params = len(self.probabilities[variable])
            local_penalty = (math.log(self.num_data_instances)*num_params)/2
            penalty += local_penalty

        BIC = LL - penalty
        return BIC


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: NB_Classifier.py [train_file.csv] [test_file.csv]")
        exit(0)
    else:
        file_name_train = sys.argv[1]
        file_name_test = sys.argv[2]
        nb_fitted = NB_Classifier(file_name_train)
        nb_tester = NB_Classifier(file_name_test, nb_fitted)
