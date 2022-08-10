import sys
import pickle
import numpy as np
from scipy import stats
from dataloader import load_test_data
from sklearn.metrics import f1_score, accuracy_score

def eval(dataset, scores):
    test_data = load_test_data(dataset)
    if "arc" in dataset or dataset == "qasc":
        correct = 0
        for i, data in enumerate(test_data):
            labels = [choice['label'] for choice in data['question']['choices']]
            prediction = labels[np.argmax(scores[i])]
            if prediction == data["answerKey"]:
                correct += 1
        print("Accuracy: ", round(100 * (correct / len(test_data)), 1))
    
    elif dataset == "sciq":
        correct = 0
        for i, data in enumerate(test_data):
            prediction = np.argmax(scores[i])
            if prediction == 3:
                correct += 1
        print("Accuracy: ", round(100 * (correct / len(test_data)), 1))
    
    elif dataset == "ag_news":
        correct = 0
        labels = ["1", "2", "3", "4"]
        for i in test_data:
            prediction = labels[np.argmax(scores[i])]
            if prediction == test_data[i]['label']:
                correct += 1
        print("Accuracy: ", round(100 * (correct / len(test_data)), 1))
    
    elif dataset == "situation":
        correct = 0
        for i in test_data:
            prediction = np.argmax(scores[i])
            if str(prediction) in test_data[i]['label']:
                correct += 1
        print("Accuracy: ", round(100 * (correct / len(test_data)), 1))
    
    elif dataset == "coarse_wsd":
        word2senses, _, coarse_wsd, sense2def = test_data
        f1 = []
        accuracy = []
        for word in word2senses:
            instance2scores = scores[word]
            predictions = [np.argmax(s) for i, s in instance2scores.items()]
            labels = [data[1] for data in coarse_wsd[word]["test"]]
            f1.append(round(f1_score(labels, predictions, average='macro') * 100, 1))
            accuracy.append(round(accuracy_score(labels, predictions) * 100, 1))
            print(word, "F1:", f1[-1], accuracy[-1])

        print("Avg. F1: ", round(np.mean(f1), 1))
        print("Avg. Accuracy: ", round(np.mean(accuracy), 1))

    elif "vicomte" in dataset:
        test_data, dist_data, _, _, candidates = test_data
        spears = []
        correct = 0
        for data in test_data:
            true_scores = dist_data[data[0]]
            mean_scores = np.mean(scores[data[0]], axis=0)
            pred = candidates[np.argmax(mean_scores)]
            spears.append(stats.spearmanr(true_scores, mean_scores)[0])

            if data[1] == pred:
                correct += 1

        print("Average Prompt:")
        print("Accuracy@1:", round((correct / len(test_data)) * 100, 1))
        print("Avg Spears:", round(np.mean(spears) * 100, 1), "std:", round(np.std(spears) * 100, 1))

if __name__ == "__main__":
    dataset, model_name = sys.argv[1:]

    try:
        scores = pickle.load(open("../output/{}/{}.p".format(dataset, model_name), "rb"))
        print("Dataset: {}\nModel: {}\nPerformance:".format(dataset, model_name))
        eval(dataset, scores)
    except:
        print("Output file not exist!")