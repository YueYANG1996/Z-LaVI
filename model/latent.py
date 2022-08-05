import sys
import pickle
from tqdm import tqdm
from dataloader import load_test_data
from evaluate import eval
import os
os.environ['TRANSFORMERS_CACHE'] = '/data1/yueyang/packages/huggingface/' # change to your own cache directory
from sentence_transformers import SentenceTransformer, util
from simcse import SimCSE

def init_model(model_name):
    if model_name == "simcse":
        model = SimCSE("princeton-nlp/unsup-simcse-roberta-large")
    elif model_name == "sbert":
        model = SentenceTransformer('all-mpnet-base-v2', device = "cuda:0", cache_folder = "/data1/yueyang/packages/sentence_transformer/")
    else:
        print("Error: model not supported!!!\nSupported models: [simcse, sbert]")
    return model

def arc_qasc(model, test_data):
    '''ARC-Easy & ARC-Challenge & QASC'''
    id2scores = {}
    for i, data in tqdm(enumerate(test_data)):
        question = data['question']['stem']
        choices = [choice['text'] for choice in data['question']['choices']]
        question_embedding = model.encode(question)
        option_embedding = model.encode(choices)
        scores = util.cos_sim(question_embedding, option_embedding)[0].tolist()
        id2scores[i] = scores
    return id2scores

def sciq(model, test_data):
    '''SciQ'''
    id2scores = {}
    for i, data in tqdm(enumerate(test_data)):
        question = data['question']
        choices = [data[choice] for choice in ['distractor1', 'distractor2', 'distractor3', 'correct_answer']]
        question_embedding = model.encode(question)
        option_embedding = model.encode(choices)
        scores = util.cos_sim(question_embedding, option_embedding)[0].tolist()
        id2scores[i] = scores
    return id2scores

def ag_news(model, test_data):
    '''AG News'''
    prompt = "This example is {}."
    news_types = ["political news", "sports news", "business news", "technology news"]
    choices = [prompt.format(news_type) for news_type in news_types]
    option_embedding = model.encode(choices)
    id2scores = {}

    for ind in tqdm(test_data):
        title = test_data[ind]['title']
        sentence = test_data[ind]['sentence']
        question = title + sentence
        question_embedding = model.encode(question)
        scores = util.cos_sim(question_embedding, option_embedding)[0].tolist()
        id2scores[ind] = scores
    return id2scores

def situation(model, test_data):
    situation_types = []
    with open("../data/situation/label_names.txt", "r") as f:
        for line in f.readlines():
            situation_types.append(line[:-1])

    prompt = "This example is {}."
    choices = [prompt.format(situation_type) for situation_type in situation_types]
    option_embedding = model.encode(choices)

    id2scores = {}
    for ind in tqdm(test_data):
        question = test_data[ind]['sent']
        question_embedding = model.encode(question)
        scores = util.cos_sim(question_embedding, option_embedding)[0].tolist()
        id2scores[ind] = scores
    return id2scores

def get_prediction_scores(dataset, model_name):
    test_data = load_test_data(dataset)
    model = init_model(model_name)

    if "arc" in dataset or dataset == "qasc":
        scores = arc_qasc(model, test_data)
    
    elif dataset == "sciq":
        scores = sciq(model, test_data)
    
    elif dataset == "ag_news":
        scores = ag_news(model, test_data)

    elif dataset == "situation":
        scores = situation(model, test_data)

    return scores


if __name__ == "__main__":
    # run model
    dataset, model_name = sys.argv[1:]
    prediction_scores = get_prediction_scores(dataset, model_name)

    # save scores for ensemble
    pickle.dump(prediction_scores, open("../output/{}/{}.p".format(dataset, model_name), "wb"))

    # evaluate
    print("Dataset: {}\nModel: {}\nPerformance:".format(dataset, model_name))
    eval(dataset, prediction_scores)