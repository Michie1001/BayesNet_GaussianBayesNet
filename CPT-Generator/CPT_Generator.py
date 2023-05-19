#############################################################################
# CPT_Generator.py
#
# This program generates Conditional Probability Tables (CPTs) into a config
# file in order to be useful for probabilistic inference. It does that by
# rewriting a given config file without CPTS. The new CPTs are derived from
# the given data file (in CSV format) -- compatible with the config file.
# This program assumes Laplacian smoothing (with l=1 to avoid 0 probabilities)
#
# Depending on the dataset and Bayes net structure, this program may take some
# minutes instead of seconds to generate CPTs. So be patient when running it.
#
# WARNING: This code has not been thoroughly tested.
#
# Version: 1.0, Date: 08 October 2022, first version
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import sys
from BayesNetReader import BayesNetReader
from NB_Classifier_v2 import NB_Classifier


class CPT_Generator(BayesNetReader, NB_Classifier):
    configfile_name = None
    bn = None
    nbc = None
    countings = {}
    CPTs = {}
    constant_l = 1  # to avoid zero probabilities

    def __init__(self, configfile_name, datafile_name):
        self.configfile_name = configfile_name
        self.bn = BayesNetReader(configfile_name)
        self.nbc = NB_Classifier(None)
        self.nbc.read_data(datafile_name)
        self.generate_prior_and_conditional_countings()
        self.generate_probabilities_from_countings()
        self.write_CPTs_to_configuration_file()

    def generate_prior_and_conditional_countings(self):
        print("\nGENERATING countings for prior/conditional distributions...")
        print("-------------------------------------------------------------")

        for pd in self.bn.bn["structure"]:
            print(str(pd))
            p = pd.replace('(', ' ')
            p = p.replace(')', ' ')
            tokens = p.split("|")

            # generate countings for prior probabilities
            if len(tokens) == 1:
                variable = tokens[0].split(' ')[1]
                variable_index = self.get_variable_index(variable)
                counts = self.initialise_counts(variable)
                self.get_counts(variable_index, None, counts)

            # generate countings for conditional probabilities
            if len(tokens) == 2:
                variable = tokens[0].split(' ')[1]
                variable_index = self.get_variable_index(variable)
                parents = tokens[1].strip().split(',')
                parent_indexes = self.get_parent_indexes(parents)
                counts = self.initialise_counts(variable, parents)
                self.get_counts(variable_index, parent_indexes, counts)

            self.countings[pd] = counts
            print("counts="+str(counts))
            print()

    def generate_probabilities_from_countings(self):
        print("\nGENERATING prior and conditional probabilities...")
        print("---------------------------------------------------")

        for pd, counts in self.countings.items():
            print(str(pd))
            tokens = pd.split("|")
            variable = tokens[0].replace("P(", "")
            cpt = {}

            # generate prior probabilities
            if len(tokens) == 1:
                _sum = 0
                for key, count in counts.items():
                    _sum += count

                Jl = len(counts)*self.constant_l
                for key, count in counts.items():
                    cpt[key] = (count+self.constant_l)/(_sum+Jl)

            # generate conditional probabilities
            if len(tokens) == 2:
                parents_values = self.get_parent_values(counts)
                for parents_value in parents_values:
                    _sum = 0
                    for key, count in counts.items():
                        if key.endswith("|"+parents_value):
                            _sum += count

                    J = len(self.nbc.rv_key_values[variable])
                    Jl = J*self.constant_l
                    for key, count in counts.items():
                        if key.endswith("|"+parents_value):
                            cpt[key] = (count+self.constant_l)/(_sum+Jl)

            self.CPTs[pd] = cpt
            print("CPT="+str(cpt))
            print()

    def get_variable_index(self, variable):
        for i in range(0, len(self.nbc.rand_vars)):
            if variable == self.nbc.rand_vars[i]:
                return i
        print("WARNING: couldn't find index of variables=%s" % (variable))
        return None

    def get_parent_indexes(self, parents):
        indexes = []
        for parent in parents:
            index = self.get_variable_index(parent)
            indexes.append(index)
        return indexes

    def get_parent_values(self, counts):
        values = []
        for key, count in counts.items():
            value = key.split('|')[1]
            if value not in values:
                values.append(value)
        return values

    def initialise_counts(self, variable, parents=None):
        counts = {}

        if parents is None:
            # initialise counts of variables without parents
            for var_val in self.nbc.rv_key_values[variable]:
                if var_val not in counts:
                    counts[var_val] = 0

        else:
            # enumerate all sequence values of parent variables
            parents_values = []
            last_parents_values = []
            for i in range(0, len(parents)):
                parent = parents[i]
                for var_val in self.nbc.rv_key_values[parent]:
                    if i == 0:
                        parents_values.append(var_val)
                    else:
                        for last_val in last_parents_values:
                            parents_values.append(last_val+','+var_val)

                last_parents_values = parents_values.copy()
                parents_values = []

            # initialise counts of variables with parents
            for var_val in self.nbc.rv_key_values[variable]:
                for par_val in last_parents_values:
                    counts[var_val+'|'+par_val] = 0

        return counts

    def get_counts(self, variable_index, parent_indexes, counts):
        # accumulate countings
        for values in self.nbc.rv_all_values:
            if parent_indexes is None:
                # case: prior probability
                value = values[variable_index]
            else:
                # case: conditional probability
                parents_values = ""
                for parent_index in parent_indexes:
                    value = values[parent_index]
                    if len(parents_values) == 0:
                        parents_values = value
                    else:
                        parents_values += ','+value
                value = values[variable_index]+'|'+parents_values
            counts[value] += 1

    def write_CPTs_to_configuration_file(self):
        print("\nWRITING config file with CPT tables...")
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
            for key, cpt in self.CPTs.items():
                cpt_header = key.replace("P(", "CPT(")
                cfg_file.write(str(cpt_header)+":")
                cfg_file.write('\n')
                num_written_probs = 0
                for domain_vals, probability in cpt.items():
                    num_written_probs += 1
                    line = str(domain_vals)+"="+str(probability)
                    line = line+";" if num_written_probs < len(cpt) else line
                    cfg_file.write(line)
                    cfg_file.write('\n')
                cfg_file.write('\n')


if len(sys.argv) != 3:
    print("USAGE: CPT_Generator.py [your_config_file.txt] [training_file.csv]")
    print("EXAMPLE> CPT_Generator.py config-playtennis.txt play_tennis-train.csv")
    exit(0)

else:
    configfile_name = sys.argv[1]
    datafile_name = sys.argv[2]
    CPT_Generator(configfile_name, datafile_name)
