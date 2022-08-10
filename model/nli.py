cache_folder = '/data1/yueyang/packages/huggingface/' # change to your own cache directory
import sys
import pickle
from tqdm import tqdm
from dataloader import load_test_data
from evaluate import eval
import os
os.environ['TRANSFORMERS_CACHE'] = cache_folder
from transformers import pipeline

def init_model(model_name):
    if model_name == "roberta":
        model = pipeline("zero-shot-classification", model="roberta-large-mnli", cache_dir=cache_folder, device=0)

    elif model_name == "bart":
        model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", cache_dir=cache_folder, device=0)
    
    else:
        print("Error: model not supported!!!\nSupported NLI models: [roberta, bart]")

    return model

def arc_qasc(model, test_data):
    '''ARC-Easy & ARC-Challenge & QASC'''
    id2scores = {}
    for ind, data in tqdm(enumerate(test_data)):
        question = "Question: " + data['question']['stem']
        candidate_options = []
        option2label = {}
        for choice in data['question']['choices']:
            option = "The answer is {}.".format(choice['text'])
            label = choice['label']
            candidate_options.append(option)
            option2label[option] = label

        model_output = model(question, candidate_options)
        labels = model_output['labels']
        scores = model_output['scores']
        option2scores = {labels[i]: scores[i] for i in range(len(candidate_options))}
        real_scores = [option2scores[option] for option in candidate_options]
        id2scores[ind] = real_scores
    return id2scores

def sciq(model, test_data):
    id2scores = {}
    for ind, data in tqdm(enumerate(test_data)):
        question = "Question: " + data['question']
        candidate_options = []
        option2label = {}
        for i, choice in enumerate(['distractor1', 'distractor2', 'distractor3', 'correct_answer']):
            option = "The answer is {}.".format(data[choice])
            candidate_options.append(option)
            option2label[option] = i
        model_output = model(question, candidate_options)

        labels = model_output['labels']
        scores = model_output['scores']
        option2scores = {labels[i]: scores[i] for i in range(len(candidate_options))}
        real_scores = [option2scores[option] for option in candidate_options]
        id2scores[ind] = real_scores
    return id2scores

def ag_news(model, test_data):
    news_types = ["political news", "sports news", "business news", "technology news"]
    candidate_options = ["{}".format(news_type) for news_type in news_types]
    id2scores = {}
    for ind in tqdm(list(test_data.keys())):
        title = test_data[ind]['title']
        sentence = test_data[ind]['sentence']
        question = title + sentence

        model_output = model(question, candidate_options)
        labels = model_output['labels']
        scores = model_output['scores']
        option2scores = {labels[i]: scores[i] for i in range(len(candidate_options))}
        real_scores = [option2scores[option] for option in candidate_options]
        id2scores[ind] = real_scores
    return id2scores

def situation(model, test_data):
    situation_types = []
    with open("/cephfs/user/yueyang/visual_grounding/situation/data/label_names.txt", "r") as f:
        for line in f.readlines():
            situation_types.append(line[:-1])
    choices = ["{}".format(situation_type) for situation_type in situation_types]
    id2scores = {}
    for ind in tqdm(test_data):
        question = test_data[ind]['sent']
        model_output = model(question, choices)

        labels = model_output['labels']
        scores = model_output['scores']
        option2scores = {labels[i]: scores[i] for i in range(len(choices))}
        real_scores = [option2scores[option] for option in choices]
        id2scores[ind] = real_scores
    return id2scores

def wsd(model, test_data):
    _, word2senses, coarse_wsd, _ = test_data
    word2scores = {}
    for word in tqdm(word2senses):
        ind2sense = word2senses[word]
        test_data = coarse_wsd[word]["test"]
        test_instances = [data[0] for data in test_data]
        predictions = []

        definitions = ["The {} mentioned before means {}.".format(word, sense) for _, sense in ind2sense.items()]
        definition2label = {definitions[i]: i for i in range(len(definitions))}

        instance2scores = {}
        for test_instance in tqdm(test_instances):
            model_output = model(test_instance, definitions)
            predictions.append(definition2label[model_output['labels'][0]])
            definition2score = {model_output['labels'][i]: model_output['scores'][i] for i in range(len(definitions))}
            instance2scores[test_instance] = [definition2score[definition] for definition in definitions]
        word2scores[word] = instance2scores
    return word2scores

def vicomte(model, test_data):
    test_data, _, prompts, _, candidates = test_data
    word2scores = {data[0]: [] for data in test_data}
    for prompt in prompts:
        for data in tqdm(test_data):
            question = "The object is {} .".format(data[0])
            choices = [prompt.format(candidate).replace("object", data[0]) for candidate in candidates]
            model_output = model(question, choices)
            labels = model_output['labels']
            scores = model_output['scores']
            option2scores = {labels[i]: scores[i] for i in range(len(choices))}
            pred_scores = [option2scores[option] for option in choices]
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