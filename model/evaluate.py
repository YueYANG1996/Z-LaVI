import numpy as np
from dataloader import load_test_data

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