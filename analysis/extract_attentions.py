
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--DATA", type=str)
parser.add_argument("--SPLIT", type=str)
parser.add_argument("--DATA_S1", type=str)
parser.add_argument("--DATA_S2", type=str, default=None)
parser.add_argument("--LATENT_DIM", type=int)
parser.add_argument("--MODEL_NAME", type=str)
parser.add_argument("--LAYER_S1", type=str)
parser.add_argument("--LAYER_S2", type=str, default=None)
parser.add_argument("--LAYER", type=str, default=None)
parser.add_argument("--LEARNING_RATE", type=str)
args = parser.parse_args()

DATA = args.DATA
SPLIT = args.SPLIT
DATA_S1 = args.DATA_S1
DATA_S2 = args.DATA_S2
LATENT_DIM = args.LATENT_DIM
MODEL_NAME = args.MODEL_NAME
LAYER_S1 = args.LAYER_S1
LAYER_S2 = args.LAYER_S2
LAYER = args.LAYER
LEARNING_RATE = args.LEARNING_RATE


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__), "..")))

OBJECTIVE = "emotion"
LAYER_S1 = LAYER_S1 if LAYER_S1 == "all" else int(LAYER_S1)
LAYER_S2 = LAYER_S2 if LAYER_S2 in ["all", None] else int(LAYER_S2)
LAYER = LAYER if LAYER == "all" else int(LAYER)
LEARNING_RATE = float(LEARNING_RATE)
BETA_S1 = "incremental" 
BETA_S2 = "incremental" 
SELECTED_GPU = 0
DATA_PATH = f"{os.environ['HOME']}/Projects/disentanglement_framework/directory/data/"
LOAD_VIB_PATH = f"{os.environ['HOME']}/Projects/disentanglement_framework/directory/models/vib/"
LOAD_PROBE_PATH = f"{os.environ['HOME']}/Projects/disentanglement_framework/directory/models/sanity/{OBJECTIVE}/"
ALIGNMENT_BASE_PATH = f"{os.environ['HOME']}/Projects/disentanglement_framework/directory/mfa/{DATA}/"
SAVE_WEIGHTS_PATH = f"{os.environ['HOME']}/Projects/disentanglement_framework/directory/analysis/{OBJECTIVE}/{DATA}/{SPLIT}/{MODEL_NAME}/"

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
from vib.vib import VIB, VIBConfig
from probing.probe import Probe, ProbeConfig
from utils import MODEL_NAME_MAPPER, PROCESSOR_MAPPER, NUM_CLASSES_MAPPER, get_frame_boundaries, add_mfa
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

# Load pre-trained model
processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_MAPPER[MODEL_NAME])
base_model = Wav2Vec2Model.from_pretrained(MODEL_NAME_MAPPER[MODEL_NAME]) if "wav2vec2" in MODEL_NAME else HubertModel.from_pretrained(MODEL_NAME_MAPPER[MODEL_NAME])
base_model.to(device)
base_model.eval()

# Load trained stage 1 vib model 
stage1_config = VIBConfig(
    input_dim=base_model.config.output_hidden_size if "wav2vec2" in MODEL_NAME else base_model.config.hidden_size,
    latent_dim=LATENT_DIM,
    stage="1",
    num_classes=NUM_CLASSES_MAPPER["1"],
    layer_weight_averaging=LAYER_S1 == "all",
    num_layers=base_model.config.num_hidden_layers if LAYER_S1 == "all" else None
    )
stage1_vib = VIB(stage1_config)
postfix = f"_bs=1_lr={LEARNING_RATE}_dim={LATENT_DIM}"
postfix += f"_b={BETA_S1}"
postfix += f"_layer={LAYER_S1}"
stage1_vib.load_state_dict(torch.load(f'{LOAD_VIB_PATH}/1/{DATA_S1}/{MODEL_NAME}/model{postfix}.pth', map_location=torch.device(device)))
stage1_vib.to(device)
stage1_vib.eval()

# Load trained stage 2 vib model
stage2_config = VIBConfig(
        input_dim=base_model.config.output_hidden_size if "wav2vec2" in MODEL_NAME else base_model.config.hidden_size,
        latent_dim=LATENT_DIM,
        stage="2",
        num_classes=NUM_CLASSES_MAPPER["2"],
        layer_weight_averaging=LAYER_S2 == "all",
        num_layers=base_model.config.num_hidden_layers if LAYER_S2 == "all" else None
        )
stage2_vib = VIB(stage2_config)
postfix = f"_bs=8_lr={LEARNING_RATE}_dim={LATENT_DIM}"
postfix += f"_b={BETA_S1}_{BETA_S2}"
postfix += f"_layer={LAYER_S1}_{LAYER_S2}"
stage2_vib.load_state_dict(torch.load(f'{LOAD_VIB_PATH}2/{DATA_S1}_{DATA_S2}/{MODEL_NAME}/model{postfix}.pth', map_location=torch.device(device)))
stage2_vib.to(device)
stage2_vib.eval()

# Load trainable probing clfs in stage 1
probe_config = ProbeConfig(
    input_dim=LATENT_DIM,
    num_classes=NUM_CLASSES_MAPPER[OBJECTIVE],
    objective=OBJECTIVE,
    layer_weight_averaging=False,
    num_layers=None
    )
stage1_probe = Probe(probe_config)
postfix = f"_bs=8_lr={LEARNING_RATE}_dim={LATENT_DIM}"
postfix += f"_b={BETA_S1}" 
postfix += f"_layer={LAYER_S1}" 
stage1_probe.load_state_dict(torch.load(f'{LOAD_PROBE_PATH}1/{DATA_S1}/{DATA}/{MODEL_NAME}/model{postfix}.pth', map_location=torch.device(device)))
stage1_probe.to(device)
stage1_probe.eval()

# Load trainable probing clfs in stage 2
probe_config = ProbeConfig(
    input_dim=LATENT_DIM,
    num_classes=NUM_CLASSES_MAPPER[OBJECTIVE],
    objective=OBJECTIVE,
    layer_weight_averaging=False,
    num_layers=None
    )
stage2_probe = Probe(probe_config)
postfix = f"_bs=8_lr={LEARNING_RATE}_dim={LATENT_DIM}"
postfix += f"_b={BETA_S1}_{BETA_S2}"
postfix += f"_layer={LAYER_S1}_{LAYER_S2}"
stage2_probe.load_state_dict(torch.load(f'{LOAD_PROBE_PATH}2/{DATA_S1}_{DATA_S2}/{DATA}/{MODEL_NAME}/model{postfix}.pth', map_location=torch.device(device)))
stage2_probe.to(device)
stage2_probe.eval()

# Load trainable probing clfs on hidden states
probe_config = ProbeConfig(
    input_dim=base_model.config.output_hidden_size if "wav2vec2" in MODEL_NAME else base_model.config.hidden_size,
    num_classes=NUM_CLASSES_MAPPER[OBJECTIVE],
    objective=OBJECTIVE,
    layer_weight_averaging=LAYER == "all",
    num_layers=base_model.config.num_hidden_layers if LAYER == "all" else None
    )
h_probe = Probe(probe_config)
postfix = f"_bs=8_lr={LEARNING_RATE}"
postfix += f"_layer={LAYER}"
h_probe.load_state_dict(torch.load(f'{LOAD_PROBE_PATH}hidden_state/{DATA}/{MODEL_NAME}/model{postfix}.pth', map_location=torch.device(device)))
h_probe.to(device)
h_probe.eval()

# Load data
dataset = load_from_disk(f"{DATA_PATH}{DATA}")[SPLIT]
dataset = dataset.select_columns(['audio', 'transcription', 'emotion'])
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
# load and integrate mfa alignments into the dataset
dataset = add_mfa(dataset, ALIGNMENT_BASE_PATH, SPLIT)

# Run
vib_preds = [] 
vib_attentions = []
textual_probe_preds = []
textual_probe_attentions = [] 
acoustic_probe_preds = []
acoustic_probe_attentions = [] 
h_probe_preds = []
h_probe_attentions = [] 
for ex in range(dataset.num_rows):
    
    # inference and get attention weights
    input_values = processor(dataset[ex]["audio"]["array"], sampling_rate=dataset[ex]['audio']['sampling_rate']).input_values[0]
    input_values = torch.tensor(input_values, device=device).unsqueeze(0)
    with torch.no_grad():
        outputs = base_model(input_values, 
                            output_hidden_states=True,
                            return_dict=True
                            )
    hidden_states = torch.stack(outputs.hidden_states)
    
    # Forward VIB model
    with torch.no_grad():
        # Forward VIB
        _, z_s1, _  = stage1_vib(hidden_states[1:].permute(1, 0, 2, 3) if LAYER_S1 == "all" else hidden_states[LAYER_S1].permute(1, 0, 2, 3), m=None) 
        vib_s2_logits, text_frame_vib_attn, audio_frame_vib_attn, z_s2, var = stage2_vib(hidden_states[1:].permute(1, 0, 2, 3) if LAYER_S2 == "all" else hidden_states[LAYER_S2].permute(1, 0, 2, 3), m=None, cond=z_s1, output_attentions=True)
        # Forward Probe 
        probe_s1_logits, text_frame_probe_attn = stage1_probe(h=z_s1, m=None, output_attentions=True)
        probe_s2_logits, audio_frame_probe_attn = stage2_probe(h=z_s2, m=None, output_attentions=True)
        # Forward h probe
        h_probe_logits, h_frame_probe_attn = h_probe(h=hidden_states[1:].permute(1, 0, 2, 3) if LAYER == "all" else hidden_states[LAYER].permute(1, 0, 2, 3), m=None, output_attentions=True)
        

    # store preds and probs
    # vib stage 2
    pred = torch.argmax(vib_s2_logits, dim=-1).detach().cpu().numpy().item()
    prob = torch.nn.functional.softmax(vib_s2_logits, dim=-1).max().detach().cpu().numpy().item()
    vib_preds.append({'pred_2': pred, 'prob_2': prob})
    # probe stage 1
    pred = torch.argmax(probe_s1_logits, dim=-1).detach().cpu().numpy().item()
    prob = torch.nn.functional.softmax(probe_s1_logits, dim=-1).max().detach().cpu().numpy().item()
    textual_probe_preds.append({'pred_1': pred, 'prob_1': prob})
    # probe stage 2
    pred = torch.argmax(probe_s2_logits, dim=-1).detach().cpu().numpy().item()
    prob = torch.nn.functional.softmax(probe_s2_logits, dim=-1).max().detach().cpu().numpy().item()
    acoustic_probe_preds.append({'pred_2': pred, 'prob_2': prob})
    # h probe
    pred = torch.argmax(h_probe_logits, dim=-1).detach().cpu().numpy().item()
    prob = torch.nn.functional.softmax(h_probe_logits, dim=-1).max().detach().cpu().numpy().item()
    h_probe_preds.append({'pred': pred, 'prob': prob})

    # store attentions
    # vib stage 2
    vib_attentions.append({'textual_attention': text_frame_vib_attn.squeeze(0, -1).detach().cpu().numpy().tolist(), 'acoustic_attention': audio_frame_vib_attn.squeeze(0, -1).detach().cpu().numpy().tolist()})
    # probe stage 1 and 2
    textual_probe_attentions.append({'textual_attention': text_frame_probe_attn.squeeze(0, -1).detach().cpu().numpy().tolist()})
    acoustic_probe_attentions.append({'acoustic_attention': text_frame_probe_attn.squeeze(0, -1).detach().cpu().numpy().tolist()})
    # h probe 
    h_probe_attentions.append({'h_attention': text_frame_probe_attn.squeeze(0, -1).detach().cpu().numpy().tolist()})


# save preds and attentions
vib_preds = pd.DataFrame(vib_preds)
vib_attentions = pd.DataFrame(vib_attentions)
textual_probe_preds = pd.DataFrame(textual_probe_preds)
textual_probe_attentions = pd.DataFrame(textual_probe_attentions)
acoustic_probe_preds = pd.DataFrame(acoustic_probe_preds)
acoustic_probe_attentions = pd.DataFrame(acoustic_probe_attentions)
h_probe_preds = pd.DataFrame(h_probe_preds)
h_probe_attentions = pd.DataFrame(h_probe_attentions)

postfix = f"_dim={LATENT_DIM}_layer={LAYER_S1}_{LAYER_S2}"

vib_preds.to_pickle(f'{SAVE_WEIGHTS_PATH}vib_preds{postfix}.pkl')
vib_attentions.to_pickle(f'{SAVE_WEIGHTS_PATH}vib_attentions{postfix}.pkl')
textual_probe_preds.to_pickle(f'{SAVE_WEIGHTS_PATH}textual_probe_preds{postfix}.pkl')
textual_probe_attentions.to_pickle(f'{SAVE_WEIGHTS_PATH}textual_probe_attentions{postfix}.pkl')
acoustic_probe_preds.to_pickle(f'{SAVE_WEIGHTS_PATH}acoustic_probe_preds{postfix}.pkl')
acoustic_probe_attentions.to_pickle(f'{SAVE_WEIGHTS_PATH}acoustic_probe_attentions{postfix}.pkl')
h_probe_preds.to_pickle(f'{SAVE_WEIGHTS_PATH}h_probe_preds{postfix}.pkl')
h_probe_attentions.to_pickle(f'{SAVE_WEIGHTS_PATH}h_probe_attentions{postfix}.pkl')
