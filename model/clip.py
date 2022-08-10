import pickle
from tqdm import tqdm
import torch as th
import clip
from dataloader import load_test_data
from evaluate import eval

device = "cuda" if th.cuda.is_available() else "cpu"
clip_model_name = "ViT-B/32"
model, preprocess = clip.load(clip_model_name, device=device)

def arc_qasc(model, test_data):
    '''ARC-Easy & ARC-Challenge & QASC'''
    id2scores = {}
    for i, data in tqdm(enumerate(test_data)):
        question = data['question']['stem']
        choices = [choice['text'] for choice in data['question']['choices']]
        question_embedding = model.encode(question)
        option_embedding = pickle.load()
        scores = util.cos_sim(question_embedding, option_embedding)[0].tolist()
        id2scores[i] = scores
    return id2scores

def sciq():
    pass

def ag_news():
    pass

def situation():
    pass

def wsd(model, test_data, imagine_type):
    pass

def vicomte(model, test_data, imagine_type):
    pass

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