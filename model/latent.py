cache_folder = '/data1/yueyang/packages/huggingface/' # change to your own cache directory

import tqdm
def nop(it, *a, **k):
    return it
tqdm.tqdm = nop

import sys
import pickle
from dataloader import load_test_data
from evaluate import eval
import os
os.environ['TRANSFORMERS_CACHE'] = cache_folder
from sentence_transformers import SentenceTransformer, util
from simcse import SimCSE
from progress.bar import Bar

def init_model(model_name):
    if model_name == "simcse":
        model = SimCSE("princeton-nlp/unsup-simcse-roberta-large")
    elif model_name == "sbert":
        model = SentenceTransformer('all-mpnet-base-v2', device = "cuda:0", cache_folder = cache_folder)
    else:
        print("Error: model not supported!!!\nSupported models: [simcse, sbert]")
    return model

def arc_qasc(model, test_data):
    '''ARC-Easy & ARC-Challenge & QASC'''
    id2scores = {}
    for i, data in Bar('Running').iter(enumerate(test_data)):
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
    for i, data in Bar('Running').iter(enumerate(test_data)):
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

    for ind in Bar('Running').iter(test_data):
        title = test_data[ind]['title']
        sentence = test_data[ind]['sentence']
        question = title + sentence
        question_embedding = model.encode(question)
        scores = util.cos_sim(question_embedding, option_embedding)[0].tolist()
        id2scores[ind] = scores
    return id2scores

def situation(model, test_data):
    '''Situation'''
    situation_types = []
    with open("../data/situation/label_names.txt", "r") as f:
        for line in f.readlines():
            situation_types.append(line[:-1])

    prompt = "This example is {}."
    choices = [prompt.format(situation_type) for situation_type in situation_types]
    option_embedding = model.encode(choices)

    id2scores = {}
    for ind in Bar('Running').iter(test_data):
        question = test_data[ind]['sent']
        question_embedding = model.encode(question)
        scores = util.cos_sim(question_embedding, option_embedding)[0].tolist()
        id2scores[ind] = scores
    return id2scores

def wsd(model, test_data):
    '''CoarseWSD-20'''
    word2senses, _, coarse_wsd, sense2def = test_data
    word2scores = {}
    for word in word2senses:
        ind2sense = word2senses[word]
        test_data = coarse_wsd[word]["test"]
        test_instances = [data[0] for data in test_data]
        definitions = [sense2def[sense] for _, sense in ind2sense.items()]
        definition_embeddings = model.encode(definitions)
        instance2scores = {}
        for test_instance in Bar('Running').iter(test_instances):
            instance_embedding = model.encode([test_instance])
            cosine_scores = util.cos_sim(instance_embedding, definition_embeddings)[0].tolist()
            instance2scores[test_instance] = cosine_scores
        word2scores[word] = instance2scores
    return word2scores

def vicomte(model, test_data):
    '''ViComTe'''
    test_data, _, prompts, _, candidates = test_data
    word2scores = {data[0]: [] for data in test_data}

    for prompt in prompts:
        for data in Bar('Running').iter(test_data):
            candidate_embeddings = model.encode([prompt.format(candidate).replace("object", data[0]) for candidate in candidates])
            target_embedding = model.encode("The object is {} .".format(data[0]))
            pred_scores = util.cos_sim(target_embedding, candidate_embeddings)[0].tolist()
            word2scores[data[0]].append(pred_scores)

    return word2scores

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
    
    elif dataset == "coarse_wsd":
        scores = wsd(model, test_data)
    
    elif "vicomte" in dataset:
        scores = vicomte(model, test_data)
    
    else:
        print("Error: dataset not supported!!!\nSupported datasets: [arc_easy, arc_challenge, qasc, sciq, ag_news, situation]")

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