# Z-LaVI

Official code for the paper "Z-LaVI: Zero-Shot Language Solver Fueled by Visual Imagination".

### Set up environments
We run our experiments using Python 3.9.12. You can install the required packages using:

```
pip install -r requirements.txt
```

## Directories
* `/data` stores the processed data of all datasets, including [AG News](https://huggingface.co/datasets/ag_news), [ARC](https://allenai.org/data/arc), [CoarseWSD-20](https://github.com/danlou/bert-disambiguation), [QASC](https://allenai.org/data/qasc), [SciQ](https://allenai.org/data/sciq), [Situation](https://github.com/CogComp/ZeroShotWiki), [ViComTe](https://github.com/chenyuheidizhang/vl-commonsense).
* `/model` stores the code to run the models
* `/output` save the models' predictions
* `/scripts` some useful scripts

## Get single model performace
First, you need to get the performance of all language models and clip. We have three categories of language models: prompt-based, latent-embedding-based, NLI-based and we evaluate on 7 datasets. 

* Models:
  * Prompt-based: gpt-neo-1.3B, gpt-neo-2.7B, gpt-j-6B, opt-30b
  * Latent-based: simcse, sbert
  * NLI-based: roberta, bart
* Datasets: ag_news, situation, arc_easy, arc_challenge, qasc, sciq, coarse_wsd, vicomte.

Navigate the `/model` directory and then:

### Prompt-based language models
`/model/prompt.py` is code for prompt-based models, to get the predictions, run:

```
python prompt.py dataset model 
```

replace the `dataset` and `model` with the exact name.

### Latent-embedding-based language models
`/model/latent.py` is code for latent-embedding-based models, to get the predictions, run:

```
python latent.py dataset model 
```

replace the `dataset` and `model` with the exact name.

### NLI-based language models
`/model/nli.py` is code for NLI-based models, to get the predictions, run:

```
python nli.py dataset model 
```

replace the `dataset` and `model` with the exact name.

### CLIP
To run the clip model, you first need to download the image embeddings: xxx. After unzip it in the `/Z-LaVI` folder, then you can run:

```
python clip_zs.py dataset imagine_type
```

We provide three types of imagination: `synthesis`, `recall` and `combine` which means generated images, web images and combine both types of images.
