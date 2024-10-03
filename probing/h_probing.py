
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--OBJECTIVE", type=str)
parser.add_argument("--DATA", type=str)
parser.add_argument("--MODEL_NAME", type=str)
parser.add_argument("--LAYER", type=str)
parser.add_argument("--LEARNING_RATE", type=str)
parser.add_argument("--SEED", type=str)
args = parser.parse_args()

OBJECTIVE = args.OBJECTIVE
DATA = args.DATA
MODEL_NAME = args.MODEL_NAME
LAYER = args.LAYER
LEARNING_RATE = args.LEARNING_RATE
SEED = args.SEED


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__), "..")))

print(DATA)
BATCH_SIZE = 1 if OBJECTIVE == "ctc" else 8
LAYER = LAYER if LAYER == "all" else int(LAYER)
STAGE = "hidden_state"
LEARNING_RATE = float(LEARNING_RATE)
SEED = int(SEED)
EPOCHS = 50
EVAL_FREQ = 10
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.005
MAX_GRAD_NORM = 1
SELECTED_GPU = 0
DATA_PATH = f"{os.environ['HOME']}/Projects/disentanglement_framework/directory/data/"
SAVE_REPORTS_PATH = f"{os.environ['HOME']}/Projects/disentanglement_framework/directory/reports/sanity/{OBJECTIVE}/{STAGE}/{DATA}/{MODEL_NAME}/{SEED}/"
SAVE_MODEL_PATH = f"{os.environ['HOME']}/Projects/disentanglement_framework/directory/models/sanity/{OBJECTIVE}/{STAGE}/{DATA}/{MODEL_NAME}/{SEED}/"

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
from probe import Probe, ProbeConfig
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
    else: 
        labels = [x["labels"] for x in batch]
        features["labels"] = torch.tensor(labels)
    return features


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
        
        # Forward Probe model
        outputs = model(h=hidden_states[1:].permute(1, 0, 2, 3) if LAYER == "all" else hidden_states[LAYER].permute(1, 0, 2, 3),
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
        
            # Forward probing model
            with torch.no_grad():
                outputs = model(h=hidden_states[1:].permute(1, 0, 2, 3) if LAYER == "all" else hidden_states[LAYER].permute(1, 0, 2, 3),
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
postfix = f"_bs={BATCH_SIZE}_lr={LEARNING_RATE}"
postfix += f"_layer={LAYER}"
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

if LAYER == "all":
    layer_weights = torch.nn.functional.softmax(model.layer_weights, dim=0).detach().cpu().numpy().tolist()
    with open(f"{SAVE_MODEL_PATH}layer-weights{postfix}.pkl", 'wb') as f:
        pickle.dump(layer_weights, f)