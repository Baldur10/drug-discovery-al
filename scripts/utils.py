from ast import expr_context
from dataclasses import replace
import numpy as np
from abc import ABC, abstractclassmethod
from modAL.utils.combination import make_linear_combination, make_product
from modAL.uncertainty import classifier_uncertainty, classifier_margin, classifier_entropy
from modAL.utils.selection import multi_argmax

class ActiveLearnerDecay(ABC):
    def __init__(self) -> None:
        self.constant_value=100

    @abstractclassmethod
    def calculate(self):
        ...

class StepWiseDecay(ActiveLearnerDecay):
    def __init__(self,percent_lists:list) -> None:
        super().__init__()
        self.percent_lists=percent_lists

    def calculate(self,num_samples):
        if len(self.percent_lists)!=7:
            print('Input only takes in 5 values')
        else:
            if num_samples<50:
                return self.percent_lists[0]
            elif num_samples<100:
                return self.percent_lists[1]
            elif num_samples<250:
                return self.percent_lists[2]
            elif num_samples<500:
                return self.percent_lists[3]
            elif num_samples<1000:
                return self.percent_lists[4]
            elif num_samples<10000:
                return self.percent_lists[5]
            else:
                return self.percent_lists[6]

class ExponentialHyperbolicDecay(ActiveLearnerDecay):
    def __init__(self, const_value=100) -> None:
        super().__init__()
        self.constant_value=const_value

    def calculate(self,train_set_size,exponent):
        result = (self.constant_value - (((train_set_size)*result)*exponent))/(self.constant_value/100)
        return result

# # Array-Wise Exploration
# assay_length = []
# for assay_id in df["assay_id"].unique():
#     length = df.loc[df["assay_id"]==assay_id].loc[df["Clustering"]=="TRN"].shape[0]
#     assay_length.append(length)

# print(max(assay_length))
# print(min(assay_length))

# list_50 = 0
# list_100 = 0
# list_250 = 0
# list_500 = 0
# list_1000 = 0
# list_10000 = 0
# list_rest = 0

# for num_samples in assay_length:
#     if num_samples<50:
#         list_50+=1
#     elif num_samples<100:
#         list_100+= 1
#     elif num_samples<250:
#         list_250+= 1
#     elif num_samples<500:
#         list_500+= 1
#     elif num_samples<1000:
#         list_1000+= 1
#     elif num_samples<10000:
#         list_10000+= 1
#     else:
#         list_rest+= 1

# print(list_50)
# print(list_100)
# print(list_250)
# print(list_500)
# print(list_1000)
# print(list_10000)
# print(list_rest)

linear_combination_1 = make_linear_combination(
    classifier_uncertainty, classifier_margin, classifier_entropy,
    weights= [1.0,1.0,1.0]
)
linear_combination_2 = make_linear_combination(
    classifier_uncertainty, classifier_margin, classifier_entropy,
    weights = [0.1,0.8,0.8]
)
linear_combination_3 = make_linear_combination(
    classifier_uncertainty, classifier_margin, classifier_entropy,
    weights = [0.8,0.8,0.1]
)
product_combination = make_product(
    classifier_uncertainty, classifier_margin, classifier_entropy,
    exponents = [0.75,0.2,0.75]
)

def random_sampling(classifier, X, n_instances=1):
    try:
        num_samples = X.shape[0]
    except:
        num_samples = len(X)
    query_idx = np.random.choice(range(num_samples), size = n_instances,replace=False)
    return query_idx, X[query_idx]

def equivalent_sampling(classifier, X, n_instances=1):
    utility = linear_combination_1(classifier, X)
    query_idx = multi_argmax(utility, n_instances=n_instances)
    return query_idx, X[query_idx]

def margin_entropy_sampling(classifier, X, n_instances=1):
    utility = linear_combination_2(classifier, X)
    query_idx = multi_argmax(utility, n_instances=n_instances)
    return query_idx, X[query_idx]

def uncertainty_margin_sampling(classifier, X, n_instances=1):
    utility = linear_combination_3(classifier, X)
    query_idx = multi_argmax(utility, n_instances=n_instances)
    return query_idx, X[query_idx]

def product_sampling(classifier, X, n_instances=1):
    utility = product_combination(classifier, X)
    query_idx = multi_argmax(utility, n_instances=n_instances)
    return query_idx, X[query_idx]