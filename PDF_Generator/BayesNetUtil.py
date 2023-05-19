#############################################################################
# BayesNetUtil.py
#
# Implements functions to simplify the implementation of algorithms for
# probabilistic inference with Bayesian networks.
#
# Version: 1.0, 06 October 2022
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import numpy as np


def tokenise_query(prob_query):
    print("\nTOKENISING probabilistic query="+str(prob_query))

    query = {}
    prob_query = prob_query[2:]
    prob_query = prob_query[:len(prob_query)-1]
    query["query_var"] = prob_query.split("|")[0]

    try:
        evidence = {}
        query["evidence"] = prob_query.split("|")[1]
        if query["evidence"].find(','):
            for pair in query["evidence"].split(','):
                tokens = pair.split('=')
                evidence[tokens[0]] = tokens[1]
            query["evidence"] = evidence

    except Exception:
        query["evidence"] = {}

    print("query="+str(query))
    return query


def get_parents(child, bn):
    for conditional in bn["structure"]:
        if conditional.startswith("P("+child+")"):
            return None
        elif conditional.startswith("P("+child+"|"):
            parents = conditional.split("|")[1]
            parents = parents[:len(parents)-1]
            return parents

    print("ERROR: Couldn't find parent(s) of variable "+str(child))
    exit(0)


def get_probability_given_parents(V, v, evidence, bn):
    parents = get_parents(V, bn)
    is_gaussian = is_gaussian_distribution(V, parents, bn)
    probability = 0

    if parents is None and is_gaussian == False:
        cpt = bn["CPT("+V+")"]
        probability = cpt[v]

    elif parents is not None and is_gaussian == False:
        cpt = bn["CPT("+V+"|"+parents+")"]
        values = v
        for parent in parents.split(","):
            separator = "|" if values == v else ","
            values = values + separator + evidence[parent]
        probability = cpt[values]

    elif parents is None and is_gaussian == True:
        mean, std = get_gaussian_params(bn["PDF("+V+")"], evidence)
        probability = get_gaussian_probability(float(v), mean, std)
        #probability = np.exp(probability)/(1+np.exp(probability))
        #print("V=%s v=%s mean=%s std=%d p=%s" % (V, v, mean, std, probability))

    elif parents is not None and is_gaussian == True:
        mean, std = get_gaussian_params(bn["PDF("+V+"|"+parents+")"], evidence)
        probability = get_gaussian_probability(float(v), mean, std)
        #probability = np.exp(probability)/(1+np.exp(probability))
        #print("V=%s v=%s mean=%s std=%d p=%s" % (V, v, mean, std, probability))

    else:
        print("ERROR: Don't know how to get probability for V="+str(V))
        exit(0)

    return probability


def get_domain_values(V, bn):
    domain_values = []

    for key, cpt in bn.items():
        if key == "CPT("+V+")":
            domain_values = list(cpt.keys())

        elif key.startswith("CPT("+V+"|"):
            for entry, prob in cpt.items():
                value = entry.split("|")[0]
                if value not in domain_values:
                    domain_values.append(value)

    if len(domain_values) == 0:
        print("ERROR: Couldn't find values of variable "+str(V))
        exit(0)

    return domain_values


def get_index_of_variable(V, bn):
    for i in range(0, len(bn["random_variables"])):
        variable = bn["random_variables"][i]
        if V == variable:
            return i

    print("ERROR: Couldn't find index of variable "+str(V))
    exit(0)


def normalise(counts):
    _sum = 0
    for value, count in counts.items():
        _sum += count

    distribution = {}
    for value, count in counts.items():
        p = float(count/_sum)
        distribution[value] = p

    return distribution


def is_gaussian_distribution(V, parents, bn):
    for key, pd in bn.items():
        if key.startswith("PDF("+V):
            return True
    return False


def get_gaussian_params(pdf, evidence):
    if pdf[0].find('*') > 0:
        mean = 0
        sum_tokens = pdf[0].split('+')
        for sum_token in sum_tokens:
            mul_tokens = sum_token.split('*')
            if len(mul_tokens) == 2:
                variable = mul_tokens[1].strip()
                evidence_val = float(evidence[variable])
                mul_res = float(mul_tokens[0]) * evidence_val
                mean += mul_res
            else:
                mean += float(mul_tokens[0])
    else:
        mean = float(pdf[0])

    std = float(pdf[1])
    return mean, std


def get_gaussian_probability(x, mean, stdev):
    e_val = -0.5*np.power((x-mean)/stdev, 2)
    probability = (1/(stdev*np.sqrt(2*np.pi))) * np.exp(e_val)
    return probability
