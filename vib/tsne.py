SAVE = False

SIZE = "base"
LATENT_DIM = 128
DATA = "RAVDESS"
DATA_S1 = "CommonVoice_LibriSpeech"
DATA_S2 = DATA_S1 + "_" + "IEMOCAP" 
LEARNING_RATE = 0.001 if SIZE == "base" else 0.0001
MODEL_NAME = "wav2vec2-base" if SIZE == "base" else "wav2vec2-large"
LAYER_S1 = "all"
LAYER_S2 = "all"

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__), "..")))

SPLITS = ["test"]
BATCH_SIZE = 8
SEED = 42
SELECTED_GPU = 0
DATA_PATH = f"{os.environ['HOME']}/Projects/disentanglement_framework/directory/data/"
LOAD_VIB_PATH = f"{os.environ['HOME']}/Projects/disentanglement_framework/directory/models/vib/"
SAVE_FIGURES_PATH = f"{os.environ['HOME']}/Projects/disentanglement_framework/directory/figures/"

## Imports
import pickle
from tqdm.auto import tqdm
import IPython.display as ipd
import numpy as np
import pandas as pd
from plotnine import *
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import concatenate_datasets, load_from_disk, Audio
from evaluate import load
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from vib import VIB, VIBConfig
from utils import MODEL_NAME_MAPPER, PROCESSOR_MAPPER, NUM_CLASSES_MAPPER

EMOTION_LABEL_MAPPER = {
    '0': 'Angry',
    '1': 'Happy',
    '2': 'Neutral',
    '3': 'Sad',
}

device = torch.device("cpu")

def load_list_of_data(data, splits):
    data_list = [load_from_disk(f"{DATA_PATH}{data}")[split].cast_column("audio", Audio(sampling_rate=16000)) for split in splits]
    return concatenate_datasets(data_list, axis=0)

def prepare_dataset(batch):
    batch['input_values'] = processor(batch["audio"]["array"], sampling_rate=batch['audio']['sampling_rate']).input_values[0]
    batch['labels'] = batch.pop('emotion')
    return batch

def collate_fn(batch):
    input_values = [{"input_values": x["input_values"]} for x in batch]
    features = processor.pad(input_values, padding=True, return_attention_mask=True, return_tensors="pt")
    labels = [x["labels"] for x in batch]
    features["labels"] = torch.tensor(labels)
    return features

# Load pre-trained model
processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_MAPPER[MODEL_NAME])
base_model = Wav2Vec2Model.from_pretrained(MODEL_NAME_MAPPER[MODEL_NAME])
base_model.to(device)
base_model.eval()

# Load trained stage 1 vib model
stage1_config = VIBConfig(
    input_dim=base_model.config.output_hidden_size,
    latent_dim=LATENT_DIM,
    stage="1",
    num_classes=NUM_CLASSES_MAPPER["1"],
    layer_weight_averaging=LAYER_S1 == "all",
    num_layers=base_model.config.num_hidden_layers if LAYER_S1 == "all" else None
    )
stage1_vib = VIB(stage1_config)
postfix = f"_bs=1_lr={LEARNING_RATE}_dim={LATENT_DIM}"
postfix += "_b=incremental"
postfix += f"_layer={LAYER_S1}"
stage1_vib.load_state_dict(torch.load(f'{LOAD_VIB_PATH}1/{DATA_S1}/{MODEL_NAME}/model{postfix}.pth', map_location=torch.device(device)))
stage1_vib.to(device)
stage1_vib.eval()

# Load trained stage 2 vib model
stage2_config = VIBConfig(
    input_dim=base_model.config.output_hidden_size,
    latent_dim=LATENT_DIM,
    stage="2",
    num_classes=NUM_CLASSES_MAPPER["2"],
    layer_weight_averaging=LAYER_S2 == "all",
    num_layers=base_model.config.num_hidden_layers if LAYER_S2 == "all" else None
    )
stage2_vib = VIB(stage2_config)
postfix = f"_bs=8_lr={LEARNING_RATE}_dim={LATENT_DIM}"
postfix += "_b=incremental_incremental"
postfix += f"_layer={LAYER_S1}_{LAYER_S2}"
stage2_vib.load_state_dict(torch.load(f'{LOAD_VIB_PATH}2/{DATA_S2}/{MODEL_NAME}/model{postfix}.pth', map_location=torch.device(device)))
stage2_vib.to(device)
stage2_vib.eval()


# Load data
data = load_list_of_data(data=DATA, splits=SPLITS).shuffle(seed=SEED)

dataset = data.map(prepare_dataset, keep_in_memory=True)
dataset = dataset.remove_columns(['audio', 'transcription'])

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False, pin_memory=True, num_workers=4) 

# Run Train set
Z_textual = np.zeros((dataset.num_rows, LATENT_DIM))
Z_acoustic = np.zeros((dataset.num_rows, LATENT_DIM))
for step, batch in enumerate(dataloader):
    batch = {k: v.to(device) for k, v in batch.items()}    
    with torch.no_grad():
        # attention_mask shouln't be passed to wave2vec 2.0 model
        outputs = base_model(batch["input_values"], 
                            output_hidden_states=True,
                            return_dict=True
                            )
    hidden_states = torch.stack(outputs.hidden_states)
    frame_mask = base_model._get_feature_vector_attention_mask(hidden_states.shape[2], batch["attention_mask"])
        
    # Forward VIB modelw
    with torch.no_grad():
        _, mu_text, _ = stage1_vib(hidden_states[1:].permute(1, 0, 2, 3) if LAYER_S1 == "all" else hidden_states[LAYER_S1].permute(1, 0, 2, 3), m=None)
        _, _, z_frame_attn, mu_audio, _ = outputs = stage2_vib(hidden_states[1:].permute(1, 0, 2, 3) if LAYER_S2 == "all" else hidden_states[LAYER_S2].permute(1, 0, 2, 3), m=frame_mask, cond=mu_text, output_attentions=True)
    
    # unweighted average over frames
    Z_textual[step*BATCH_SIZE: step*BATCH_SIZE+BATCH_SIZE] = mu_text.mean(dim=1).detach().cpu().numpy()
    Z_acoustic[step*BATCH_SIZE: step*BATCH_SIZE+BATCH_SIZE] = mu_audio.mean(dim=1).detach().cpu().numpy()



# TSNE
text_embedded = TSNE(n_components=2, perplexity=21, learning_rate='auto').fit_transform(Z_textual)
audio_embedded = TSNE(n_components=2, perplexity=21, learning_rate='auto').fit_transform(Z_acoustic)

emotion_labels = [EMOTION_LABEL_MAPPER[str(e)] for e in data['emotion']]

df_text = pd.DataFrame({
    'x': text_embedded[:, 0],
    'y': text_embedded[:, 1],
    'Emotion': emotion_labels,
    'Transcription': data['transcription'],
    'Stage': r"$z^{textual}$"
})

df_audio = pd.DataFrame({
    'x': audio_embedded[:, 0],
    'y': audio_embedded[:, 1],
    'Emotion': emotion_labels,
    'Transcription': data['transcription'],
    'Stage': r"$z^{acoustic}$"
})

df_combined = pd.concat([df_text, df_audio])
df_combined['Stage'] = pd.Categorical(df_combined['Stage'], categories=[r"$z^{textual}$", r"$z^{acoustic}$"])

palette = sns.color_palette("hsv", len(EMOTION_LABEL_MAPPER)).as_hex()

g = (ggplot(df_combined, aes(x='x', y='y', color='Emotion', shape='Transcription'))
    + geom_point(size=3)
    + scale_color_manual(values=palette)
    + theme(legend_position='top', legend_title_align='center')
    + theme(plot_title=element_text(ha='center'))
    + theme(axis_title_x=element_blank(), axis_title_y=element_blank())
    + facet_wrap('~Stage')
)
g = g + guides(color=guide_legend(nrow=2), shape=guide_legend(nrow=2))
print(g)
if SAVE:
    ggsave(g, f'{SAVE_FIGURES_PATH}/tsne_{LAYER_S1}_{LAYER_S2}_{LATENT_DIM}_{SIZE}.pdf')


