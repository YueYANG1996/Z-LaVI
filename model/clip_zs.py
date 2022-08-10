import sys
import pickle
from tqdm import tqdm
import torch as th
import clip
from dataloader import load_test_data
from evaluate import eval

device = "cuda" if th.cuda.is_available() else "cpu"
clip_model_name = "ViT-B/32"
model, preprocess = clip.load(clip_model_name, device=device)

def preprocess_sentence(sentence):
    tokens = sentence.split(" ")
    while True:
        sentence = " ".join(tokens)
        try:
            clip.tokenize(sentence)
            break
        except:
            tokens = tokens[:-5]
    return sentence

def get_text_embeddings(sentences):
    with th.no_grad():
        text = clip.tokenize([preprocess_sentence(sentence) for sentence in sentences]).to(device)
        encoded_text = model.encode_text(text)
    encoded_text /= encoded_text.norm(dim=-1, keepdim=True)
    return encoded_text

def arc_qasc(model, test_data):
    '''ARC-Easy & ARC-Challenge & QASC'''
    # id2scores = {}
    # for i, data in tqdm(enumerate(test_data)):
    #     question = data['question']['stem']
    #     choices = [choice['text'] for choice in data['question']['choices']]
    #     question_embedding = model.encode(question)
    #     option_embedding = pickle.load()
    #     scores = util.cos_sim(question_embedding, option_embedding)[0].tolist()
    #     id2scores[i] = scores
    # return id2scores

def sciq():
    pass

def ag_news():
    pass

def situation(test_data, imagine_type):
    situation_types = []
    with open("../data/situation/label_names.txt", "r") as f:
        for line in f.readlines():
            situation_types.append(line[:-1])
    
    prompt = "A news image of {} situation."
    choices = [prompt.format(situation_type) for situation_type in situation_types]
    option_embedding = get_text_embeddings(choices)
    id2scores = {}
    for ind in tqdm(test_data):
        question_embedding = pickle.load(open("../image_features/{}/situation/{}.p".format(imagine_type, ind), "rb")).to(device)
        scores = th.mean(question_embedding @ option_embedding.T, dim=0).tolist()
        id2scores[ind] = scores
    return id2scores

def wsd(test_data, imagine_type):
    word2senses, _, coarse_wsd, _ = test_data
    word2scores = {}
    for word in word2senses:
        ind2sense = word2senses[word]
        test_data = coarse_wsd[word]["test"]
        test_instances = [data[0] for data in test_data]
        instance2scores = {}
        for test_instance in tqdm(test_instances):
            instance_embedding = get_text_embeddings([test_instance])
            scores = []
            for _, sense in ind2sense.items():
                sense_embedding = pickle.load(open("../image_features/{}/coarse_wsd/{}.p".format(imagine_type, sense), "rb")).to(device)
                similarity = th.mean(sense_embedding @ instance_embedding.T, dim=0)
                scores.append(float(similarity.cpu()))
            instance2scores[test_instance] = scores
        word2scores[word] = instance2scores
    return word2scores

def vicomte(test_data, imagine_type):
    pass

def get_prediction_scores(dataset, imagine_type):
    test_data = load_test_data(dataset)

    if "arc" in dataset or dataset == "qasc":
        scores = arc_qasc(model, test_data)
    
    elif dataset == "sciq":
        scores = sciq(model, test_data)
    
    elif dataset == "ag_news":
        scores = ag_news(model, test_data)

    elif dataset == "situation":
        scores = situation(test_data, imagine_type)
    
    elif dataset == "coarse_wsd":
        scores = wsd(test_data, imagine_type)
    
    elif "vicomte" in dataset:
        scores = vicomte(model, test_data)
    
    else:
        print("Error: dataset not supported!!!\nSupported datasets: [arc_easy, arc_challenge, qasc, sciq, ag_news, situation]")

    return scores

if __name__ == "__main__":
    # run model
    dataset, imagine_type = sys.argv[1:]
    prediction_scores = get_prediction_scores(dataset, imagine_type)

    # save scores for ensemble
    pickle.dump(prediction_scores, open("../output/{}/clip_{}.p".format(dataset, imagine_type), "wb"))

    # evaluate
    print("Dataset: {}\nModel: clip-{}\nPerformance:".format(dataset, imagine_type))
    eval(dataset, prediction_scores)