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
    
    elif dataset == "coarse_wsd":
        word2senses = pickle.load(open("../data/coarse_wsd/word2senses.p", "rb"))

        word2senses_short = {'apple': {0: 'company', 1: 'fruit'}, 'arm': {0: 'architecture', 1: 'limb'}, 
                             'bank': {0: 'institution', 1: 'geography'}, 'bass': {0: 'guitar', 1: 'voice type', 2: 'double'}, 
                             'bow': {0: 'ship', 1: 'weapon', 2: 'music'}, 'chair': {0: 'chairman', 1: 'furniture'}, 
                             'club': {0: 'organization', 1: 'nightclub', 2: 'weapon'}, 'crane': {0: 'machine', 1: 'bird'}, 
                             'deck': {0: 'ship', 1: 'building'}, 'digit': {0: 'numerical', 1: 'anatomy'}, 
                             'hood': {0: 'comics', 1: 'vehicle', 2: 'headgear'}, 'java': {0: 'island', 1: 'program'}, 
                             'mole': {0: 'animal', 1: 'espionage', 2: 'unit', 3: 'sauce', 4: 'architecture'}, 
                             'pitcher': {0: 'baseball', 1: 'container'}, 'pound': {0: 'mass', 1: 'currency'}, 
                             'seal': {0: 'pinniped', 1: 'musician', 2: 'emblem', 3: 'mechanical'}, 'spring': {0: 'hydrology', 1: 'season', 2: 'device'}, 
                             'square': {0: 'shape', 1: 'company', 2: 'town', 3: 'number'}, 'trunk': {0: 'botany', 1: 'automobile', 2: 'anatomy'}, 
                             'yard': {0: 'unit', 1: 'sail'}}

        coarse_wsd = pickle.load(open("../data/coarse_wsd/CoarseWSD-20.p", "rb"))
        sense2def = pickle.load(open("../data/coarse_wsd/sense2def.p", "rb"))
        test_data = (word2senses, word2senses_short, coarse_wsd, sense2def)
    
    elif "vicomte" in dataset:
        test_type = dataset.split("_")[1]
        dist_data = json.load(open("../data/vicomte/{}/{}-dist.jsonl".format(test_type, test_type), "rb"))
        test_data = pickle.load(open("/cephfs/user/yueyang/visual_grounding/ViComTe/data/{}/test.p".format(test_type), "rb"))

        if test_type == "color":
            prompts_1 = ["The object can be of color {} .", "The object has color {} .",
                         "The color of the object can be {} .", "The color of the object is {} .",
                         "{} object .", "This is a {} object .", "The object is of color {} ."]
            
            # the content of the prompts is the same, just different formats for different models
            prompts_2 = ["{} can be of color", "{} has color",
                         "The color of {} can be", "The color of the {} is",
                         "{} is", "This {} is", "{} is of color"]

            candidates = ['black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'silver', 'white', 'yellow']
    
        elif test_type == "material":
            prompts_1 = ["The object is made of {} .", "The object can be made of {} .",
                         "The object is made from {} .", "The object can be made from {} .",
                         "{} object .", "This is a {} object .", "{} is used to make the object ."]
            
            prompts_2 = ["{} is made of", "{} can be made of", 
                         "{} is made from", "{} can be made from",
                         "{} is", "This {} is", "{} is made by using"]

            candidates = ['bronze', 'ceramic', 'cloth', 'concrete', 'cotton', 'denim', 'glass', 'gold', 'iron', 'jade', 'leather', 'metal', 'paper', 'plastic', 'rubber', 'stone', 'tin', 'wood']

        elif test_type == "shape":
            prompts_1 = ["The object can be of shape {} .", "The object has shape of {} .",
                         "The object is of shape {} .", "The shape of the object can be {} .",
                         "The shape of the object is {} .", "{} .", "This is {} ."]
            
            prompts_2 = ["{} can be shape of", "{} has shape of",
                         "{} is of shape", "The shape of {} can be",
                         "The shape of the {} is", "{} is", "This {} is"]
            
            candidates = ['cross', 'heart', 'octagon', 'oval', 'polygon', 'rectangle', 'rhombus', 'round', 'semicircle', 'square', 'star', 'triangle']

        test_data = (test_data, dist_data, prompts_1, prompts_2, candidates)
    
    return test_data