import jax
import jax.numpy as jnp
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from flax.jax_utils import replicate
from dalle_mini import DalleBartProcessor
from flax.training.common_utils import shard_prng_key
import numpy as np
from PIL import Image
from functools import partial
import random
import os
from tqdm import tqdm
import json

# change cache directory if necessary
# os.environ["WANDB_CACHE_DIR"] = "" 

DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"
DALLE_COMMIT_ID = None
VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

def read_text(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            if line[-1] == "\n":
                line = line[:-1]
            if "\t" in line:
                data.append(line.split("\t"))
            else:
                data.append(line)
    return data

# model inference
print(jax.local_device_count())
model, params = DalleBart.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False)
vqgan, vqgan_params = VQModel.from_pretrained(VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False)
params = replicate(params)
vqgan_params = replicate(vqgan_params)

@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
def p_generate(
    tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
):
    return model.generate(
        **tokenized_prompt,
        prng_key=key,
        params=params,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        condition_scale=condition_scale,
    )


# decode image
@partial(jax.pmap, axis_name="batch")
def p_decode(indices, params):
    return vqgan.decode_code(indices, params=params)

processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)

def main(prompt, n_predictions = 8):
    # create a random key
    seed = random.randint(0, 2**32 - 1)
    key = jax.random.PRNGKey(seed)

    tokenized_prompt = processor([prompt])
    tokenized_prompt = replicate(tokenized_prompt)
    
    # We can customize generation parameters
    gen_top_k = None
    gen_top_p = None
    temperature = None
    cond_scale = 3.0

    images = []
    for i in tqdm(range(max(n_predictions // jax.device_count(), 1))):
        # get a new key
        key, subkey = jax.random.split(key)
        # generate images
        encoded_images = p_generate(
            tokenized_prompt,
            shard_prng_key(subkey),
            params,
            gen_top_k,
            gen_top_p,
            temperature,
            cond_scale,
        )
        # remove BOS
        encoded_images = encoded_images.sequences[..., 1:]
        # decode images
        decoded_images = p_decode(encoded_images, vqgan_params)
        decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
        for decoded_img in decoded_images:
            img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
            images.append(img)
    return images

if __name__ == "__main__":
    prompt = "A red apple."
    images = main(prompt, 1)
    images[0].save("apple.jpg")