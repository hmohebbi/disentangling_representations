
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--DATA", type=str)
parser.add_argument("--SPLIT", type=str)
parser.add_argument("--MODEL_NAME", type=str)
parser.add_argument("--LAYER", type=str, default=None)
parser.add_argument("--LEARNING_RATE", type=str)
args = parser.parse_args()

DATA = args.DATA
SPLIT = args.SPLIT
MODEL_NAME = args.MODEL_NAME
LAYER = args.LAYER
LEARNING_RATE = args.LEARNING_RATE


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__), "..")))

OBJECTIVE = "emotion"
LAYER = LAYER if LAYER == "all" else int(LAYER)
LEARNING_RATE = float(LEARNING_RATE)
SELECTED_GPU = 0
DATA_PATH = f"{os.environ['HOME']}/Projects/disentangling_representations/directory/data/"
LOAD_VIB_PATH = f"{os.environ['HOME']}/Projects/disentangling_representations/directory/models/vib/"
LOAD_PROBE_PATH = f"{os.environ['HOME']}/Projects/disentangling_representations/directory/models/sanity/{OBJECTIVE}/"
ALIGNMENT_BASE_PATH = f"{os.environ['HOME']}/Projects/disentangling_representations/directory/mfa/{DATA}/"
SAVE_WEIGHTS_PATH = f"{os.environ['HOME']}/Projects/disentangling_representations/directory/analysis/{OBJECTIVE}/{DATA}/{SPLIT}/{MODEL_NAME}/"

## Imports
import pickle
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
import pandas as pd
from plotnine import *
import torch
from datasets import load_from_disk, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2Model, HubertModel
from probing.probe import Probe, ProbeConfig
from utils import MODEL_NAME_MAPPER, PROCESSOR_MAPPER, NUM_CLASSES_MAPPER, get_frame_boundaries, add_mfa
from captum.attr import IntegratedGradients
import IPython.display as ipd
import spacy
from textblob import TextBlob
import parselmouth
nlp = spacy.load('en_core_web_sm')


if not os.path.exists(SAVE_WEIGHTS_PATH):
    os.makedirs(SAVE_WEIGHTS_PATH)

## GPU
if torch.cuda.is_available():     
    device = torch.device(f"cuda:{SELECTED_GPU}")
    print('We will use the GPU:', torch.cuda.get_device_name(SELECTED_GPU))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')

def summarize_attributions(attributions):
    attributions = torch.linalg.norm(attributions, axis=-1)
    attributions = attributions / attributions.sum()
    return attributions

# Load pre-trained model
processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_MAPPER[MODEL_NAME])
base_model = Wav2Vec2Model.from_pretrained(MODEL_NAME_MAPPER[MODEL_NAME]) if "wav2vec2" in MODEL_NAME else HubertModel.from_pretrained(MODEL_NAME_MAPPER[MODEL_NAME])
base_model.to(device)
base_model.eval()

# Load trainable probing clfs 
probe_config = ProbeConfig(
    input_dim=base_model.config.output_hidden_size if "wav2vec2" in MODEL_NAME else base_model.config.hidden_size,
    num_classes=NUM_CLASSES_MAPPER[OBJECTIVE],
    objective=OBJECTIVE,
    layer_weight_averaging=LAYER == "all",
    num_layers=base_model.config.num_hidden_layers if LAYER == "all" else None
    )
probe_model = Probe(probe_config)
postfix = f"_bs=8_lr={LEARNING_RATE}"
postfix += f"_layer={LAYER}"
probe_model.load_state_dict(torch.load(f'{LOAD_PROBE_PATH}hidden_state/{DATA}/{MODEL_NAME}/model{postfix}.pth', map_location=torch.device(device)))
probe_model.to(device)
probe_model.eval()

# Helper Functions
def wrapper_model(inputs):
    outputs = base_model(inputs, 
                         output_hidden_states=True,
                         return_dict=True
                        )
    hidden_states = torch.stack(outputs.hidden_states)
    logits, _ = probe_model(h=hidden_states[1:].permute(1, 0, 2, 3) if LAYER == "all" else hidden_states[LAYER].permute(1, 0, 2, 3), m=None, output_attentions=True)
    return logits

# Load data
dataset = load_from_disk(f"{DATA_PATH}{DATA}")[SPLIT]
dataset = dataset.select_columns(['audio', 'transcription', 'emotion'])
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
# load and integrate mfa alignments into the dataset
dataset = add_mfa(dataset, ALIGNMENT_BASE_PATH, SPLIT)

# Run
igs = [] 
for ex in range(dataset.num_rows):
    
    # inference and get attention weights
    input_values = processor(dataset[ex]["audio"]["array"], sampling_rate=dataset[ex]['audio']['sampling_rate']).input_values[0]
    input_values = torch.tensor(input_values, device=device).unsqueeze(0)
    with torch.no_grad():
        # forward base model
        outputs = base_model(input_values, 
                            output_hidden_states=True,
                            return_dict=True
                            )
        hidden_states = torch.stack(outputs.hidden_states)
        # forward probe 
        logits, _ = probe_model(h=hidden_states[1:].permute(1, 0, 2, 3) if LAYER == "all" else hidden_states[LAYER].permute(1, 0, 2, 3), m=None, output_attentions=True)

    # store preds and probs
    pred = torch.argmax(logits, dim=-1).detach().cpu().numpy().item()
    prob = torch.nn.functional.softmax(logits, dim=-1).max().detach().cpu().numpy().item()
    
    # compute grads
    # get representaions for silence baseline for IG
    input_baseline_values = processor(np.zeros_like(dataset[ex]["audio"]["array"]), sampling_rate=dataset[ex]['audio']['sampling_rate']).input_values[0]
    input_baseline_values = torch.tensor(input_baseline_values, device=device).unsqueeze(0)
    
    # IG
    ig = IntegratedGradients(wrapper_model)
    attributions = ig.attribute(input_values, 
                                baselines=input_baseline_values, 
                                target=pred, return_convergence_delta=False)
    print(attributions.shape)
    attributions = summarize_attributions(attributions).squeeze(0).detach().cpu().numpy()
    # store grads
    igs.append({'ig': attributions.tolist()})
    


# save preds and attentions
igs = pd.DataFrame(igs)
postfix = f"_layer={LAYER}"
igs.to_pickle(f'{SAVE_WEIGHTS_PATH}model_igs{postfix}.pkl')
