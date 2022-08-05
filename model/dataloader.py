import json
import pickle

def load_test_data(dataset):
    if dataset == "arc_easy":
        test_data = []
        with open("../data/arc/ARC-Easy-Dev.jsonl", "r") as f:
            for line in f.readlines():
                test_data.append(json.loads(line))
    
    elif dataset == "arc_challenge":
        test_data = []
        with open("../data/arc/ARC-Challenge-Dev.jsonl", "r") as f:
            for line in f.readlines():
                test_data.append(json.loads(line))
    
    elif dataset == "qasc":
        test_data = []
        with open("../data/qasc/QASC_Dataset/dev.jsonl") as f:
            for line in f:
                test_data.append(json.loads(line))
    
    elif dataset == "sciq":
        test_data = json.load(open("../data/sciq/valid.json", "rb"))
    
    elif dataset == "ag_news":
        test_data = pickle.load(open("../data/ag_news/test.p", "rb"))
    
    elif dataset == "situation":
        test_data = pickle.load(open("../data/situation/test.p", "rb"))
    
    return test_data