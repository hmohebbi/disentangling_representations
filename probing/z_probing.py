
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--OBJECTIVE", type=str)
parser.add_argument("--DATA", type=str)
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

OBJECTIVE = args.OBJECTIVE
DATA = args.DATA
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
SEED = args.SEED
NO_IB = args.NO_IB


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__), "..")))

print(DATA)
BATCH_SIZE = 1 if OBJECTIVE == "ctc" else 8
LAYER_S1 = LAYER_S1 if LAYER_S1 == "all" else int(LAYER_S1)
LAYER_S2 = LAYER_S2 if (not LAYER_S2 or LAYER_S2 == "all") else int(LAYER_S2)
LEARNING_RATE = float(LEARNING_RATE)
BETA = BETA_S1 if STAGE == "1" else BETA_S2
SEED = int(SEED)
EPOCHS = 50 
EVAL_FREQ = 10
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.005
MAX_GRAD_NORM = 1
SELECTED_GPU = 0
DATA_S2 = DATA_S1 + "_" + DATA_S2
DATA_ = DATA_S1 if STAGE == "1" else DATA_S2
DATA_PATH = f"{os.environ['HOME']}/Projects/disentangling_representations/directory/data/"
LOAD_VIB_PATH = f"{os.environ['HOME']}/Projects/disentangling_representations/directory/models/vib/"
SAVE_REPORTS_PATH = f"{os.environ['HOME']}/Projects/disentangling_representations/directory/reports/sanity/{OBJECTIVE}/{STAGE}/{DATA_}/{DATA}/{MODEL_NAME}/"
SAVE_MODEL_PATH = f"{os.environ['HOME']}/Projects/disentangling_representations/directory/models/sanity/{OBJECTIVE}/{STAGE}/{DATA_}/{DATA}/{MODEL_NAME}/"

## Imports
import pickle
import numpy as np
import uuid
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import concatenate_datasets, load_from_disk, Audio
from evaluate import load
from transformers import Wav2Vec2Processor, Wav2Vec2Model, HubertModel
from transformers import get_cosine_schedule_with_warmup
from probing.probe import Probe, ProbeConfig
from vib.vib import VIB, VIBConfig
from utils import MODEL_NAME_MAPPER, PROCESSOR_MAPPER, METRIC_MAPPER, NUM_CLASSES_MAPPER
from utils import normalize, discretize

if not os.path.exists(SAVE_MODEL_PATH):
    os.makedirs(SAVE_MODEL_PATH)
if not os.path.exists(SAVE_REPORTS_PATH):
    os.makedirs(SAVE_REPORTS_PATH)


## GPU
if torch.cuda.is_available():     
    device = torch.device(f"cuda:{SELECTED_GPU}")
    print('We will use the GPU:', torch.cuda.get_device_name(SELECTED_GPU))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')


def load_list_of_data(data, split):
    data_list = [load_from_disk(f"{DATA_PATH}{d}")[split].select_columns(['audio', 'transcription'] if OBJECTIVE == "ctc" else ['audio', OBJECTIVE]).cast_column("audio", Audio(sampling_rate=16000)) for d in data]
    return concatenate_datasets(data_list, axis=0)

def prepare_dataset(batch):
    batch['input_values'] = processor(batch["audio"]["array"], sampling_rate=batch['audio']['sampling_rate']).input_values[0]
    if OBJECTIVE == "ctc":
        with processor.as_target_processor():
            batch['labels'] = processor(batch["transcription"]).input_ids
    else:
        batch['labels'] = batch.pop(OBJECTIVE)
    return batch

def collate_fn(batch):
    input_values = [{"input_values": x["input_values"]} for x in batch]
    features = processor.pad(input_values, padding=True, return_attention_mask=True, return_tensors="pt")
    if OBJECTIVE == "ctc":
        labels = [{"input_ids": x["labels"]} for x in batch]
        with processor.as_target_processor():
            labels = processor.pad(labels, padding=True, return_tensors="pt")
        # replacing padding with -100 to ignore loss correctly
        features["labels"] = labels['input_ids'].masked_fill(labels.attention_mask.ne(1), -100)
    else: # emotion labels
        labels = [x["labels"] for x in batch]
        features["labels"] = torch.tensor(labels)
    return features


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
if NO_IB:
    postfix += "_noib" 
else:
    postfix += f"_b={BETA_S1}"
postfix += f"_layer={LAYER_S1}"
stage1_vib.load_state_dict(torch.load(f'{LOAD_VIB_PATH}1/{DATA_S1}/{MODEL_NAME}/model{postfix}.pth', map_location=torch.device(device)))
stage1_vib.to(device)
stage1_vib.eval()

# Load trained stage 2 vib model
if STAGE == "2":
    stage2_config = VIBConfig(
        input_dim=base_model.config.output_hidden_size if "wav2vec2" in MODEL_NAME else base_model.config.hidden_size,
        latent_dim=LATENT_DIM,
        stage="2",
        num_classes=NUM_CLASSES_MAPPER["2"] if DATA_S2.split('_')[-1] == "IEMOCAP" else NUM_CLASSES_MAPPER["speaker_id"],
        layer_weight_averaging=LAYER_S2 == "all",
        num_layers=base_model.config.num_hidden_layers if LAYER_S2 == "all" else None
        )
    stage2_vib = VIB(stage2_config)
    postfix = f"_bs=8_lr={LEARNING_RATE}_dim={LATENT_DIM}"
    if NO_IB:
        postfix += "_noib" 
    else:
        postfix += f"_b={BETA_S1}_{BETA_S2}"
    postfix += f"_layer={LAYER_S1}_{LAYER_S2}"
    stage2_vib.load_state_dict(torch.load(f'{LOAD_VIB_PATH}2/{DATA_S2}/{MODEL_NAME}/model{postfix}.pth', map_location=torch.device(device)))
    stage2_vib.to(device)
    stage2_vib.eval()

# Load trainable probing clfs
probe_config = ProbeConfig(
    input_dim=LATENT_DIM,
    num_classes=NUM_CLASSES_MAPPER[OBJECTIVE],
    objective=OBJECTIVE,
    layer_weight_averaging=False,
    num_layers=None
    )
model = Probe(probe_config)
model.to(device)
model.train()

# Load data
train_data = load_list_of_data(data=DATA.split("_"), split='train').shuffle(seed=SEED)
test_data = load_list_of_data(data=DATA.split("_"), split='test')

# preprocess regression labels
if OBJECTIVE in ['duration', 'mean_intensity', 'mean_pitch', 'std_pitch', 'local_jitter', 'local_shimmer']:
    train_data, test_data = discretize(train_data, test_data, OBJECTIVE)


train_data = train_data.map(prepare_dataset, keep_in_memory=True)
test_data = test_data.map(prepare_dataset, keep_in_memory=True)
train_data = train_data.remove_columns('audio')
test_data = test_data.remove_columns('audio')

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True, pin_memory=True, num_workers=4) 
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False, pin_memory=True, num_workers=4) 

training_steps = len(train_dataloader)
total_training_steps = EPOCHS * training_steps

# Load metrics & optimizer
metric = load(METRIC_MAPPER[OBJECTIVE], experiment_id=str(uuid.uuid4()))
if OBJECTIVE == "ctc":
    metric_2 = load('cer', experiment_id=str(uuid.uuid4()))
optimizer = AdamW(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=np.ceil(WARMUP_RATIO * total_training_steps), 
                                            num_training_steps=total_training_steps)

# Run
train_losses = {'Task': []}
test_performances = []
if OBJECTIVE == "ctc":
    test_performances_2 = []
for epoch in range(EPOCHS):
    model.train()
    for batch in train_dataloader:
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
        with torch.no_grad():
            _, z, _ = stage1_vib(hidden_states[1:].permute(1, 0, 2, 3) if LAYER_S1 == "all" else hidden_states[LAYER_S1].permute(1, 0, 2, 3), m=None)
            if STAGE == "2":
                _, _, z_frame_attn, z, _ = stage2_vib(hidden_states[1:].permute(1, 0, 2, 3) if LAYER_S2 == "all" else hidden_states[LAYER_S2].permute(1, 0, 2, 3), m=frame_mask, cond=z, output_attentions=True)
    

        # Forward Probe model
        outputs = model(h=z,
                       m=None if OBJECTIVE == "ctc" else frame_mask, 
                       output_attentions=True
                       )
        logits = outputs[0]
        
        # CTC loss
        if OBJECTIVE == "ctc":
            tr_labels_mask = batch['labels'] >= 0
            input_lengths = frame_mask.sum(-1)
            target_lengths = tr_labels_mask.sum(-1)
            flattened_targets = batch['labels'].masked_select(tr_labels_mask)
            if flattened_targets.max() >= base_model.config.vocab_size:
                raise ValueError("Label values must be <= vocab_size")
            
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)
            with torch.backends.cudnn.flags(enabled=False):
                loss = torch.nn.functional.ctc_loss(log_probs,
                                                    flattened_targets,
                                                    input_lengths,
                                                    target_lengths,
                                                    blank=processor.tokenizer.pad_token_id, # 0 
                                                    reduction='mean', 
                                                    zero_infinity=False
                                                    )
        # classification loss
        else: 
            loss = torch.nn.functional.cross_entropy(logits, batch['labels'], ignore_index=-100)
        
        # perform optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # store records
        train_losses['Task'].append(loss.item())


    # Evaluating on test set 
    if (epoch + 1) % EVAL_FREQ == 0:
        model.eval()
        for batch in test_dataloader:
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
        
            # Forward VIB modelw
            with torch.no_grad():
                _, z, _ = stage1_vib(hidden_states[1:].permute(1, 0, 2, 3) if LAYER_S1 == "all" else hidden_states[LAYER_S1].permute(1, 0, 2, 3), m=None)
                if STAGE == "2":
                    _, _, z_frame_attn, z, _ = stage2_vib(hidden_states[1:].permute(1, 0, 2, 3) if LAYER_S2 == "all" else hidden_states[LAYER_S2].permute(1, 0, 2, 3), m=frame_mask, cond=z, output_attentions=True)
        

            # Forward Probe model
            with torch.no_grad():
                outputs = model(h=z,
                            m=None if OBJECTIVE == "ctc" else frame_mask, 
                            output_attentions=True
                            )
            logits = outputs[0]
                
            # Performance
            preds = torch.argmax(logits, dim=-1)
            if OBJECTIVE == "ctc":
                predictions = processor.batch_decode(preds)
                batch['labels'][batch['labels'] == -100] = processor.tokenizer.pad_token_id
                references = processor.batch_decode(batch['labels'], group_tokens=False) 
            else:
                predictions = preds
                references = batch['labels']
            metric.add_batch(predictions=predictions, references=references)
            if OBJECTIVE == "ctc":
                metric_2.add_batch(predictions=predictions, references=references)

        # computing eval metrics
        perf = metric.compute() if OBJECTIVE == "ctc" else metric.compute()[METRIC_MAPPER[OBJECTIVE]]
        print(f"Test {METRIC_MAPPER[OBJECTIVE]}: {perf}")
        test_performances.append(perf)
        if OBJECTIVE == "ctc":
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
with open(f"{SAVE_REPORTS_PATH}/train_losses{postfix}.pkl", 'wb') as f:
    pickle.dump(train_losses, f)
with open(f"{SAVE_REPORTS_PATH}/test_{METRIC_MAPPER[OBJECTIVE]}{postfix}.pkl", 'wb') as f:
    pickle.dump(test_performances, f)
if OBJECTIVE == "ctc":
    with open(f"{SAVE_REPORTS_PATH}/test_cer{postfix}.pkl", 'wb') as f:
        pickle.dump(test_performances_2, f)

# save model
torch.save(model.state_dict(), f'{SAVE_MODEL_PATH}model{postfix}.pth')


