cache_folder = '/data1/yueyang/packages/huggingface/' # change to your own cache directory
import sys
import pickle
import copy
import torch
import numpy as np
from tqdm import tqdm
from dataloader import load_test_data
from evaluate import eval
import os
os.environ['TRANSFORMERS_CACHE'] = cache_folder

from transformers import GPTJForCausalLM, AutoTokenizer, GPT2Tokenizer, GPTNeoForCausalLM, OPTForCausalLM

def init_model(model_name):

    if model_name == "gpt-j-6B":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda:0")

    elif model_name == "gpt-neo-1.3B":
        tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").to("cuda:0")

    elif model_name == "gpt-neo-2.7B":
        tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B").to("cuda:0")

    elif "opt" in model_name:
        num_gpus = torch.cuda.device_count()
        max_memory = {i: "18GB" for i in range(num_gpus)} # adjust the max memory based on your gpu
        tokenizer = AutoTokenizer.from_pretrained("facebook/{}".format(model_name), use_fast=False)
        model = OPTForCausalLM.from_pretrained("facebook/{}".format(model_name), torch_dtype=torch.float16, device_map="auto", offload_folder="offload_folder", max_memory=max_memory, low_cpu_mem_usage=True)

    else:
        print("Error: model not supported!!!\nSupported models: [gpt-neo-1.3B, gpt-neo-2.7B, gpt-j-6B, opt-30b]")

    return tokenizer, model

def get_start_loc(tokenizer, prompt):
    tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"]
    return tokens.shape[1] - 1

def qasc(tokenizer, model, test_data):
    id2scores = {}
    for i, data in tqdm(enumerate(test_data)):
        question = "Question: " + data['question']['stem']
        scores = []
        for choice in data['question']['choices']:
            option = "The answer is {}.".format(choice['text'])
            prompt = question + " " + option
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
            labels = copy.deepcopy(inputs["input_ids"])
            loss = model(**inputs, labels=labels).loss
            scores.append(loss.item())
        id2scores[i] = np.reciprocal(scores)
    return id2scores

def arc(tokenizer, model, test_data):
    id2scores = {}
    for i, data in tqdm(enumerate(test_data)):
        question = "Question: " + data['question']['stem']
        scores = []
        start_loc = get_start_loc(tokenizer, question + " The answer is ")
        for choice in data['question']['choices']:
            option = "The answer is {}.".format(choice['text'])
            prompt = question + " " + option
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
            labels = copy.deepcopy(inputs["input_ids"])
            labels[0, :start_loc] = -100
            labels[0, -1] = -100
            loss = model(**inputs, labels=labels).loss
            scores.append(loss.item())
        id2scores[i] = np.reciprocal(scores)
    return id2scores

def sciq(tokenizer, model, test_data):
    id2scores = {}
    for i, data in tqdm(enumerate(test_data)):
        question = "Question: " + data['question']
        scores = []
        start_loc = get_start_loc(tokenizer, question + " The answer is ")
        for choice in ['distractor1', 'distractor2', 'distractor3', 'correct_answer']:
            option = "The answer is {}.".format(data[choice])
            prompt = question + " " + option
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
            labels = copy.deepcopy(inputs["input_ids"])
            labels[0, :start_loc] = -100
            labels[0, -1] = -100
            loss = model(**inputs, labels=labels).loss
            scores.append(loss.item())
        id2scores[i] = np.reciprocal(scores)

    return id2scores

def ag_news(tokenizer, model, test_data):
    prompt = "This example is {}."
    news_types = ["political news", "sports news", "business news", "technology news"]
    choices = [prompt.format(news_type) for news_type in news_types]
    id2scores = {}
    for ind in tqdm(list(test_data.keys())):
        title = test_data[ind]['title']
        sentence = test_data[ind]['sentence']
        question = title + sentence
        scores = []
        start_loc = get_start_loc(question + " This example is ")

        for option in choices:
            inputs = tokenizer(question, option, return_tensors="pt", truncation="only_first", max_length=512).to("cuda:0")
            labels = copy.deepcopy(inputs["input_ids"])
            loss = model(**inputs, labels=labels).loss
            scores.append(loss.item())

        id2scores[ind] = np.reciprocal(scores)
    return id2scores

def wsd(tokenizer, model, test_data):
    _, word2senses, coarse_wsd, _ = test_data
    word2scores = {}
    for word in word2senses:
        ind2sense = word2senses[word]
        test_data = coarse_wsd[word]["test"]
        test_instances = [data[0] for data in test_data]
        instance2scores = {}
        for test_instance in tqdm(test_instances):
            losses = []
            for _, sense in ind2sense.items():
                prompt = test_instance + " The {} mentioned before means {}.".format(word, sense)
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
                labels = copy.deepcopy(inputs["input_ids"])
                outputs = model(**inputs, labels=labels)
                losses.append(outputs.loss.item())
            instance2scores[test_instance] = np.reciprocal(losses)
        word2scores[word] = instance2scores
    return word2scores

def situation(tokenizer, model, test_data):
    situation_types = []
    with open("/cephfs/user/yueyang/visual_grounding/situation/data/label_names_new.txt", "r") as f:
        for line in f.readlines():
            situation_types.append(line[:-1])

    prompt = "This example is {}."
    choices = [prompt.format(situation_type) for situation_type in situation_types]

    id2scores = {}
    for ind in tqdm(list(test_data.keys())):
        question = test_data[ind]['sent']
        scores = []
        start_loc = get_start_loc(question + " This example is ")

        for option in choices:
            inputs = tokenizer(question, option, return_tensors="pt", truncation="only_first", max_length=512).to("cuda:0")
            labels = copy.deepcopy(inputs["input_ids"])
            labels[0, :start_loc] = -100
            labels[0, -1] = -100
            loss = model(**inputs, labels=labels).loss
            scores.append(loss.item())

        id2scores[ind] = np.reciprocal(scores)

    return id2scores

def vicomte(tokenizer, model, test_data):
    test_data, _, _, prompts, candidates = test_data
    word2scores = {data[0]: [] for data in test_data}

    for prompt in prompts:
        for data in tqdm(test_data):
            start_loc = get_start_loc(tokenizer, prompt.format(data[0])) + 1
            pred_scores = []
            for candidiate in candidates:
                sent = prompt.format(data[0]) + " {}.".format(candidiate)
                inputs = tokenizer(sent, return_tensors="pt").to("cuda:0")
                labels = copy.deepcopy(inputs["input_ids"])
                labels[0, :start_loc] = -100
                labels[0, start_loc+1:] = -100
                loss = model(**inputs, labels=labels).loss
                pred_scores.append(loss.item())
            pred_scores = np.reciprocal(pred_scores)
            word2scores[data[0]].append(pred_scores)
    return word2scores

def get_prediction_scores(dataset, model_name):
    test_data = load_test_data(dataset)
    tokenizer, model = init_model(model_name)

    if dataset == "qasc":
        scores = qasc(tokenizer, model, test_data)
    
    elif "arc" in dataset:
        scores = arc(tokenizer, model, test_data)

    elif dataset == "sciq":
        scores = sciq(tokenizer, model, test_data)
    
    elif dataset == "ag_news":
        scores = ag_news(tokenizer, model, test_data)

    elif dataset == "situation":
        scores = situation(tokenizer, model, test_data)
    
    elif dataset == "coarse_wsd":
        scores = wsd(tokenizer, model, test_data)
    
    elif "vicomte" in dataset:
        scores = vicomte(tokenizer, model, test_data)
    
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