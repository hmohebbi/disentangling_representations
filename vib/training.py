
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--STAGE", type=str)
parser.add_argument("--DATA_S1", type=str)
parser.add_argument("--DATA_S2", type=str, default=None)
parser.add_argument("--LATENT_DIM", type=int)
parser.add_argument("--MODEL_NAME", type=str)
parser.add_argument("--LAYER_S1", type=str)
parser.add_argument("--LAYER_S2", type=str, default=None)
parser.add_argument("--LEARNING_RATE", type=str)
parser.add_argument("--BETA_S1", type=str)
parser.add_argument("--BETA_S2", type=str, default=None)
parser.add_argument("--SEED", type=str)
parser.add_argument("--NO_IB", action='store_true')
args = parser.parse_args()

STAGE = args.STAGE
DATA_S1 = args.DATA_S1
DATA_S2 = args.DATA_S2
LATENT_DIM = args.LATENT_DIM
MODEL_NAME = args.MODEL_NAME
LAYER_S1 = args.LAYER_S1
LAYER_S2 = args.LAYER_S2
LEARNING_RATE = args.LEARNING_RATE
BETA_S1 = args.BETA_S1
BETA_S2 = args.BETA_S2
NO_IB = args.NO_IB
SEED = args.SEED


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__), "..")))

DATA = DATA_S1 if STAGE == "1" else DATA_S2
print(DATA)
OBJECTIVE = "emotion" if DATA_S2 == "IEMOCAP" else "speaker_id"
BATCH_SIZE = 1 if STAGE == "1" else 8
LAYER_S1 = LAYER_S1 if LAYER_S1 == "all" else int(LAYER_S1)
LAYER_S2 = LAYER_S2 if LAYER_S2 in ["all", None] else int(LAYER_S2)
LEARNING_RATE = float(LEARNING_RATE)
SEED = int(SEED)
BETA = BETA_S1 if STAGE == "1" else BETA_S2
EPOCHS = 50
EVAL_FREQ = 10
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.005
MAX_GRAD_NORM = 1
SELECTED_GPU = 0
DATA_ = DATA_S1 if STAGE == "1" else DATA_S1 + "_" + DATA_S2
DATA_PATH = f"{os.environ['HOME']}/Projects/disentanglement_framework/directory/data/"
LOAD_STAGE1_PATH = f"{os.environ['HOME']}/Projects/disentanglement_framework/directory/models/vib/1/{DATA_S1}/{MODEL_NAME}/"
SAVE_REPORTS_PATH = f"{os.environ['HOME']}/Projects/disentanglement_framework/directory/reports/vib/{STAGE}/{DATA_}/{MODEL_NAME}/"
SAVE_MODEL_PATH = f"{os.environ['HOME']}/Projects/disentanglement_framework/directory/models/vib/{STAGE}/{DATA_}/{MODEL_NAME}/"

## Imports
import pickle
import numpy as np
import matplotlib.pyplot as plt
import uuid
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import concatenate_datasets, load_from_disk, Audio
from evaluate import load
from transformers import Wav2Vec2Processor, Wav2Vec2Model, HubertModel
from transformers import get_cosine_schedule_with_warmup
from vib import VIB, VIBConfig
from utils import MODEL_NAME_MAPPER, PROCESSOR_MAPPER, NUM_CLASSES_MAPPER, METRIC_MAPPER

if not os.path.exists(SAVE_REPORTS_PATH):
    os.makedirs(SAVE_REPORTS_PATH)
if not os.path.exists(SAVE_MODEL_PATH):
    os.makedirs(SAVE_MODEL_PATH)


## GPU
if torch.cuda.is_available():     
    device = torch.device(f"cuda:{SELECTED_GPU}")
    print('We will use the GPU:', torch.cuda.get_device_name(SELECTED_GPU))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')

def load_list_of_data(data, split):
    data_list = [load_from_disk(f"{DATA_PATH}{d}")[split].select_columns(['audio', 'transcription'] if STAGE == "1" else ['audio', OBJECTIVE]).cast_column("audio", Audio(sampling_rate=16000)) for d in data]
    return concatenate_datasets(data_list, axis=0)

def prepare_dataset(batch):
    batch['input_values'] = processor(batch["audio"]["array"], sampling_rate=batch['audio']['sampling_rate']).input_values[0]
    if STAGE == "1":
        with processor.as_target_processor():
            batch['labels'] = processor(batch["transcription"]).input_ids
    else:
        batch['labels'] = batch.pop(OBJECTIVE)
    return batch

def collate_fn(batch):
    input_values = [{"input_values": x["input_values"]} for x in batch]
    features = processor.pad(input_values, padding=True, return_attention_mask=True, return_tensors="pt")
    if STAGE == "1": # transcription labels
        labels = [{"input_ids": x["labels"]} for x in batch]
        with processor.as_target_processor():
            labels = processor.pad(labels, padding=True, return_tensors="pt")
        # replacing padding with -100 to ignore loss correctly
        features["labels"] = labels['input_ids'].masked_fill(labels.attention_mask.ne(1), -100)
    else: # emotion/speaker id labels
        labels = [x["labels"] for x in batch]
        features["labels"] = torch.tensor(labels)
    return features

# Load pre-trained model
processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_MAPPER[MODEL_NAME])
base_model = Wav2Vec2Model.from_pretrained(MODEL_NAME_MAPPER[MODEL_NAME]) if "wav2vec2" in MODEL_NAME else HubertModel.from_pretrained(MODEL_NAME_MAPPER[MODEL_NAME])
base_model.to(device)
base_model.eval()

# Load trained stage 1 vib model 
if STAGE == "2":
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
    if NO_IB:
        postfix += "_noib" 
    else:
        postfix += f"_b={BETA_S1}"
    postfix += f"_layer={LAYER_S1}"
    stage1_vib.load_state_dict(torch.load(f'{LOAD_STAGE1_PATH}model{postfix}.pth', map_location=torch.device(device)))
    stage1_vib.to(device)
    stage1_vib.eval()

# Load trainable clfs
layer_weight_averaging = (STAGE == "1" and LAYER_S1 == "all") or (STAGE == "2" and LAYER_S2 == "all")
vib_config = VIBConfig(
    input_dim=base_model.config.output_hidden_size if "wav2vec2" in MODEL_NAME else base_model.config.hidden_size,
    latent_dim=LATENT_DIM,
    stage=STAGE,
    num_classes=NUM_CLASSES_MAPPER["speaker_id"] if OBJECTIVE == "speaker_id" and STAGE == "2" else NUM_CLASSES_MAPPER[STAGE],
    layer_weight_averaging=layer_weight_averaging,
    num_layers=base_model.config.num_hidden_layers if layer_weight_averaging else None
    )
model = VIB(vib_config)
model.to(device)
model.train()


# Load data
train_data = load_list_of_data(data=DATA.split("_"), split='train').shuffle(seed=SEED)
test_data = load_list_of_data(data=DATA.split("_"), split='test')

train_data = train_data.map(prepare_dataset, keep_in_memory=True)
test_data = test_data.map(prepare_dataset, keep_in_memory=True)
train_data = train_data.remove_columns('audio')
test_data = test_data.remove_columns('audio')

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True, pin_memory=True, num_workers=4) 
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False, pin_memory=True, num_workers=4) 

training_steps = len(train_dataloader)
total_training_steps = EPOCHS * training_steps

# Load metrics & optimizer
metric = load(METRIC_MAPPER[STAGE], experiment_id=str(uuid.uuid4()))
if STAGE == "1":
    metric_2 = load('cer', experiment_id=str(uuid.uuid4()))
optimizer = AdamW(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=np.ceil(WARMUP_RATIO * total_training_steps), 
                                            num_training_steps=total_training_steps)

beta_reach_steps = (EPOCHS - 5) * training_steps
beta = 0.1 if BETA == "incremental" else float(BETA)
BETA_INCREMENT = (1.0 - beta) / beta_reach_steps if BETA == "incremental" else 0


# Run
train_losses = {'Task': [], 'Info': [], 'Total': []}
test_performances = []
if STAGE == "1":
    test_performances_2 = []
for epoch in range(EPOCHS):
    model.train()
    for step, batch in enumerate(train_dataloader):
        # Feature extraction (from pre-trained speech model)
        batch = {k: v.to(device) for k, v in batch.items()}    
        with torch.no_grad():
            # attention_mask shouln't be passed to wave2vec 2.0 model
            outputs = base_model(batch["input_values"], 
                                 output_hidden_states=True,
                                 return_dict=True
                                )
        
        hidden_states = torch.stack(outputs.hidden_states)
        frame_mask = base_model._get_feature_vector_attention_mask(hidden_states.shape[2], batch["attention_mask"])
        
        # Forward VIB model
        if STAGE == "1":
            logits, mu, var = model(hidden_states[1:].permute(1, 0, 2, 3) if LAYER_S1 == "all" else hidden_states[LAYER_S1].permute(1, 0, 2, 3), m=None, noise=not NO_IB)
        else: # Stage 2
            with torch.no_grad():
                _, cond, _  = stage1_vib(hidden_states[1:].permute(1, 0, 2, 3) if LAYER_S1 == "all" else hidden_states[LAYER_S1].permute(1, 0, 2, 3), m=None, noise=not NO_IB) 
            outputs = model(hidden_states[1:].permute(1, 0, 2, 3) if LAYER_S2 == "all" else hidden_states[LAYER_S2].permute(1, 0, 2, 3), m=frame_mask, cond=cond, output_attentions=True, noise=not NO_IB)
            logits, cond_frame_attn, z_frame_attn, mu, var = outputs


        # Info loss
        if NO_IB:
            info_loss = 0.0
        else:
            info_loss = -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var, dim=-1)
            info_loss = torch.masked_select(info_loss, frame_mask).mean() # ignore padded time steps
        
        # Task loss
        if STAGE == "1":
            # CTC loss
            tr_labels_mask = batch['labels'] >= 0
            input_lengths = frame_mask.sum(-1)
            target_lengths = tr_labels_mask.sum(-1)
            flattened_targets = batch['labels'].masked_select(tr_labels_mask)
            if flattened_targets.max() >= base_model.config.vocab_size:
                raise ValueError("Label values must be <= vocab_size")
            
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)
            with torch.backends.cudnn.flags(enabled=False):
                task_loss = torch.nn.functional.ctc_loss(log_probs,
                                                    flattened_targets,
                                                    input_lengths,
                                                    target_lengths,
                                                    blank=processor.tokenizer.pad_token_id, # 0 
                                                    reduction='mean', 
                                                    zero_infinity=False
                                                    )
        else: # Stage 2
            # cross entropy for emotion/speaker id loss
            task_loss = torch.nn.functional.cross_entropy(logits, batch['labels'], ignore_index=-100)

        # Total loss
        total_loss = task_loss + beta * info_loss 

        # perform optimization
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # store records
        train_losses['Task'].append(task_loss.item())
        if not NO_IB:
            train_losses['Info'].append(info_loss.item())
            train_losses['Total'].append(total_loss.item())
        if BETA == "incremental":
            beta = min(beta + BETA_INCREMENT, 1.0)


    # Evaluating on test set 
    if (epoch + 1) % EVAL_FREQ == 0:
        model.eval()
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}    
            with torch.no_grad():
                # attention_mask shouln't be passed to wave2vec 2.0 model
                outputs = base_model(batch["input_values"], 
                                    output_hidden_states=True,
                                    return_dict=True
                                    )
            hidden_states = torch.stack(outputs.hidden_states)
            frame_mask = base_model._get_feature_vector_attention_mask(hidden_states.shape[2], batch["attention_mask"])
            
            # Forward VIB model
            if STAGE == "1":
                with torch.no_grad():
                    logits, mu, var = model(hidden_states[1:].permute(1, 0, 2, 3) if LAYER_S1 == "all" else hidden_states[LAYER_S1].permute(1, 0, 2, 3), m=None)
            else: # Stage 2
                with torch.no_grad():
                    _, cond, _  = stage1_vib(hidden_states[1:].permute(1, 0, 2, 3) if LAYER_S1 == "all" else hidden_states[LAYER_S1].permute(1, 0, 2, 3), m=None) 
                    outputs = model(hidden_states[1:].permute(1, 0, 2, 3) if LAYER_S2 == "all" else hidden_states[LAYER_S2].permute(1, 0, 2, 3), m=frame_mask, cond=cond, output_attentions=True)
                logits, cond_frame_attn, z_frame_attn, mu, var = outputs
                
            # Performance
            preds = torch.argmax(logits, dim=-1)
            if STAGE == "1": # wer for stage 1
                predictions = processor.batch_decode(preds)
                batch['labels'][batch['labels'] == -100] = processor.tokenizer.pad_token_id
                references = processor.batch_decode(batch['labels'], group_tokens=False) 
            else: # accuracy for stage 2
                predictions = preds
                references = batch['labels']
            metric.add_batch(predictions=predictions, references=references)
            if STAGE == "1":
                metric_2.add_batch(predictions=predictions, references=references)

        # computing eval metrics
        perf = metric.compute() if STAGE == "1" else metric.compute()[METRIC_MAPPER[STAGE]]
        print(f"Test {METRIC_MAPPER[STAGE]}: {perf}")
        test_performances.append(perf)
        if STAGE == "1":
            perf_2 = metric_2.compute()
            print(f"Test CER: {perf_2}")
            test_performances_2.append(perf_2)


# Saving reports
postfix = f"_bs={BATCH_SIZE}_lr={LEARNING_RATE}_dim={LATENT_DIM}"
if NO_IB:
    postfix += "_noib" 
else:
    postfix += f"_b={BETA_S1}" if STAGE == "1" else f"_b={BETA_S1}_{BETA_S2}"
postfix += f"_layer={LAYER_S1}" if STAGE == "1" else f"_layer={LAYER_S1}_{LAYER_S2}"
print(postfix)
with open(f"{SAVE_REPORTS_PATH}train_losses{postfix}.pkl", 'wb') as f:
    pickle.dump(train_losses, f)
with open(f"{SAVE_REPORTS_PATH}test_{METRIC_MAPPER[STAGE]}{postfix}.pkl", 'wb') as f:
    pickle.dump(test_performances, f)
if STAGE == "1":
    with open(f"{SAVE_REPORTS_PATH}/test_cer{postfix}.pkl", 'wb') as f:
        pickle.dump(test_performances_2, f)

# save model
torch.save(model.state_dict(), f'{SAVE_MODEL_PATH}model{postfix}.pth')
if (STAGE == "1" and LAYER_S1 == "all") or (STAGE == "2" and LAYER_S2 == "all"):
    layer_weights = torch.nn.functional.softmax(model.layer_weights, dim=0).detach().cpu().numpy().tolist()
    with open(f"{SAVE_MODEL_PATH}layer-weights{postfix}.pkl", 'wb') as f:
        pickle.dump(layer_weights, f)

