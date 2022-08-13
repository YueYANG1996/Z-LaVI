import sys
import pickle
import numpy as np
from scipy.special import softmax
from dataloader import load_test_data
from evaluate import eval

def wsd(model1, model2, weight):
    word2senses, _, coarse_wsd, _ = load_test_data("coarse_wsd")
    word2scores = {}
    for word in word2senses:
        test_data = coarse_wsd[word]["test"]
        test_instances = [data[0] for data in test_data]
        instance2scores = {}
        for test_instance in test_instances:
            model1_score = softmax(model1[word][test_instance])
            model2_score = softmax(model2[word][test_instance])
            combined_score = weight * model1_score + (1 - weight) * model2_score
            instance2scores[test_instance] = combined_score
        word2scores[word] = instance2scores
    return word2scores

def general_ensemble(model1, model2, weight):
    ind2scores = {}
    for ind in model1:
        model1_score = softmax(model1[ind])
        model2_score = softmax(model2[ind])
        combined_score = weight * model1_score + (1 - weight) * model2_score
        ind2scores[ind] = combined_score
    return ind2scores

def vicomte_ensemble(word2scores1, word2scores2, weight):
    word2combine_scores = {}
    for word, scores1 in word2scores1.items():
        scores2 = word2scores2[word]
        combine_scores = weight * softmax(np.mean(scores1, axis=0)) + (1 - weight) * softmax(np.mean(scores2, axis=0))
        word2combine_scores[word] = combine_scores
    return word2combine_scores

def get_ensemble_scores(dataset, model1, model2, weight):
    
    if dataset == "coarse_wsd":
        ensemble_scores = wsd(model1, model2, weight)
    
    elif "vicomte" in dataset:
        ensemble_scores = vicomte_ensemble(model1, model2, weight)

    else:
        ensemble_scores = general_ensemble(model1, model2, weight)

    return ensemble_scores

if __name__ == "__main__":
    dataset, model1_name, model2_name, weight = sys.argv[1:]
    model1 = pickle.load(open("../output/{}/{}.p".format(dataset, model1_name), "rb"))
    model2 = pickle.load(open("../output/{}/{}.p".format(dataset, model2_name), "rb"))
    ensemble_scores = get_ensemble_scores(dataset, model1, model2, float(weight))

    print("Dataset: {}".format(dataset))

    print("Model-1: {}".format(model1_name))
    eval(dataset, model1)

    print("Model-2: {}".format(model2_name))
    eval(dataset, model2)

    print("Ensemble weight = {}:".format(weight))
    eval(dataset, ensemble_scores)