#############################################################################
# PDF_Generator.py
#
# This program generates Conditional Probability Density Functions (PDFs) 
# into a config file in order to be useful for probabilistic inference. 
# Similarly to CPT_Generator, it does that by rewriting a given config file 
# without PDFs. The new PDFs are derived from the given data file (CSV file).
#
# WARNING: This code has not been thoroughly tested.
#
# Version: 1.0, Date: 11 October 2022, first version
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import sys
import numpy as np
from BayesNetReader import BayesNetReader
from NB_Classifier_v3 import NB_Classifier
from sklearn import linear_model


class PDF_Generator(BayesNetReader, NB_Classifier):
    configfile_name = None
    bn = None
    nbc = None
    means = {}
    stdevs = {}
    coefficients = {}
    intercepts = {}
    equations = {}
    lm = linear_model.Ridge()

    def __init__(self, configfile_name, datafile_name):
        self.configfile_name = configfile_name
        self.bn = BayesNetReader(configfile_name)
        self.nbc = NB_Classifier(None)
        self.nbc.read_data(datafile_name)
        self.estimate_linear_gaussians()
        self.write_PDFs_to_configuration_file()

    def estimate_linear_gaussians(self):
        print("\nESTIMATING linear Gaussians...")
        print("---------------------------------------------------")

        for pd in self.bn.bn["structure"]:
            print(str(pd))
            p = pd.replace('(', ' ')
            p = p.replace(')', ' ')
            tokens = p.split("|")

            # generate univariate distributions
            variable = tokens[0].split(' ')[1]
            parents = None			
            mean, stdev = self.get_mean_and_standard_deviation(variable)
            self.means[variable] = mean
            self.stdevs[variable] = stdev
            print("mean=%s stdev=%s" % (mean, stdev))

            # generate multivariate distributions, if needed (via linear Gaussians)
            if len(tokens) == 2:
                variable = tokens[0].split(' ')[1]
                parents = tokens[1].strip().split(',')
                inputs, outputs = self.get_feature_vectors(parents, variable)
                self.lm.fit(inputs, outputs)
                self.coefficients[variable] = self.lm.coef_
                self.intercepts[variable] = self.lm.intercept_
                print("coefficients=%s intercept=%s" % (self.lm.coef_, self.lm.intercept_))

            equation = self.get_equation_from_linear_models(variable, parents)
            self.equations[pd] = equation
            print("equation=%s" % (equation))
            print()

    def get_mean_and_standard_deviation(self, variable):
        feature_vector = self.get_feature_vector(variable)
        mean = np.mean(feature_vector)
        stdev = np.std(feature_vector)
        return mean, stdev

    def get_feature_vector(self, variable):
        variable_index = self.get_variable_index(variable)
        feature_vector = []
        for instance in self.nbc.rv_all_values:
            value = instance[variable_index]
            feature_vector.append(value)
        return feature_vector

    def get_variable_index(self, variable):
        for i in range(0, len(self.nbc.rand_vars)):
            if variable == self.nbc.rand_vars[i]:
                return i
        print("WARNING: couldn't find index of variables=%s" % (variable))
        return None

    def get_feature_vectors(self, parents, variable):
        input_features = []
        for parent in parents:
            feature_vector = self.get_feature_vector(parent)
            if len(input_features) == 0:
                for f in range(0, len(feature_vector)):
                    input_features.append([feature_vector[f]])
            else:
                for f in range(0, len(feature_vector)):
                    tmp_vector = input_features[f]
                    tmp_vector.append(feature_vector[f])
                    input_features[f] = tmp_vector

        output_features = self.get_feature_vector(variable)
        return input_features, output_features

    def get_equation_from_linear_models(self, variable, parents):
        mean = self.means[variable]
        stdev = self.stdevs[variable]
        equation = ""
        if variable not in self.coefficients:
            equation = str(mean) +" ; "+ str(stdev)
        else:
            coefficient_vector = self.coefficients[variable]
            intercept = self.intercepts[variable]
            for p in range(0, len(parents)):
                coefficient = coefficient_vector[p]
                parent = parents [p]
                equation += " + " if len(equation) > 0 else ""
                equation += str(coefficient) +"*"+ parent
            equation += " + "+ str(intercept)
            equation += " ; "+ str(stdev)
        return equation

    def write_PDFs_to_configuration_file(self):
        print("\nWRITING config file with linear Gaussians...")
        print("See rewritten file "+str(self.configfile_name))
        print("---------------------------------------------------")
        name = self.bn.bn["name"]

        rand_vars = self.bn.bn["random_variables_raw"]
        rand_vars = str(rand_vars).replace('[', '').replace(']', '')
        rand_vars = str(rand_vars).replace('\'', '').replace(', ', ';')

        structure = self.bn.bn["structure"]
        structure = str(structure).replace('[', '').replace(']', '')
        structure = str(structure).replace('\'', '').replace(', ', ';')

        with open(self.configfile_name, 'w') as cfg_file:
            cfg_file.write("name:"+str(name))
            cfg_file.write('\n')
            cfg_file.write('\n')
            cfg_file.write("random_variables:"+str(rand_vars))
            cfg_file.write('\n')
            cfg_file.write('\n')
            cfg_file.write("structure:"+str(structure))
            cfg_file.write('\n')
            cfg_file.write('\n')
            for key, equation in self.equations.items():
                cpt_header = key.replace("P(", "PDF(")
                cfg_file.write(str(cpt_header)+":")
                cfg_file.write('\n')
                cfg_file.write(str(equation))
                cfg_file.write('\n')
                cfg_file.write('\n')


if len(sys.argv) != 3:
    print("USAGE: PDF_Generator.py [your_config_file.txt] [training_file.csv]")
    print("EXAMPLE> PDF_Generator.py config-playtennis.txt play_tennis-train.csv")
    exit(0)

else:
    configfile_name = sys.argv[1]
    datafile_name = sys.argv[2]
    PDF_Generator(configfile_name, datafile_name)
