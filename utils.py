import sys, os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 

MODEL_NAME_MAPPER = {
    'wav2vec2-base': "facebook/wav2vec2-base", # pre-trained
    'wav2vec2-base-960h': "facebook/wav2vec2-base-960h", # fine-tuned
    'hubert-base-ls960': "facebook/hubert-base-ls960", # pre-trained
    'wav2vec2-large': "facebook/wav2vec2-large", # pre-trained
    'wav2vec2-large-960h': "facebook/wav2vec2-large-960h", # fine-tuned
    'hubert-large-ll60k': "facebook/hubert-large-ll60k", # pre-trained
    'hubert-large-ls960-ft': "facebook/hubert-large-ls960-ft", # fine-tuned
}
PROCESSOR_MAPPER = { # use tokenizers of finetuned models
    'wav2vec2-base': "facebook/wav2vec2-base-960h", 
    'wav2vec2-base-960h': "facebook/wav2vec2-base-960h",
    'hubert-base-ls960': "facebook/hubert-large-ls960-ft",
    'wav2vec2-large': "facebook/wav2vec2-large-960h",
    'wav2vec2-large-960h': "facebook/wav2vec2-large-960h",
    'hubert-large-ll60k': "facebook/hubert-large-ls960-ft", 
    'hubert-large-ls960-ft': "facebook/hubert-large-ls960-ft",
}

EMOTION_LABEL_MAPPER = {
    '0': 'ang',
    '1': 'hap',
    '2': 'neu',
    '3': 'sad',
}

GENDER_LABEL_MAPPER = {
    '0': 'female',
    '1': 'male',
}

METRIC_MAPPER = {
    'ctc': 'wer',
    'emotion': 'accuracy',
    '1': 'wer',
    '2': 'accuracy',
    'duration': 'accuracy', 
    'mean_intensity': 'accuracy',
    'mean_pitch': 'accuracy',
    'std_pitch': 'accuracy',
    'local_jitter': 'accuracy',
    'local_shimmer': 'accuracy',
    'gender': 'accuracy',
    'speaker_id': 'accuracy',
}

NUM_CLASSES_MAPPER = {
    'ctc': 32,
    'emotion': 4,
    '1': 32,
    '2': 4,
    'duration': 4,
    'mean_intensity': 4,
    'mean_pitch': 4,
    'std_pitch': 4,
    'local_jitter': 4,
    'local_shimmer': 4,
    'gender': 2,
    'speaker_id': 24,
}


def normalize(train_data, test_data, key):
    # fit for train
    normalizer = MinMaxScaler().fit(np.array(train_data[key]).reshape(-1, 1))
    # transform train
    normalized_values = normalizer.transform(np.array(train_data[key]).reshape(-1, 1)).flatten().tolist()
    train_data = train_data.remove_columns(key)
    train_data = train_data.add_column(key, normalized_values)
    # transform test
    normalized_values = normalizer.transform(np.array(test_data[key]).reshape(-1, 1)).flatten().tolist()
    test_data = test_data.remove_columns(key)
    test_data = test_data.add_column(key, normalized_values)
    return train_data, test_data


def discretize(train_data, test_data, key):
    # train data
    discretized_train_values, bin_edges = pd.qcut(pd.Series(train_data[key]), 4, labels=[0, 1, 2, 3], retbins=True)
    discretized_train_values = discretized_train_values.to_numpy().astype(int)
    train_data = train_data.remove_columns(key)
    train_data = train_data.add_column(key, discretized_train_values)
    
    # discretize the test data according to bins in train data
    discretized_test_values = pd.cut(pd.Series(test_data[key]), bins=bin_edges, labels=[0, 1, 2, 3], include_lowest=True)
    discretized_test_values = discretized_test_values.to_numpy().astype(int)
    test_data = test_data.remove_columns(key)
    test_data = test_data.add_column(key, discretized_test_values)
    
    return train_data, test_data


def get_frame_boundaries(start, end, total_frames, total_audio_time):
    start = total_frames * start / total_audio_time
    end = total_frames * end / total_audio_time
    start = np.ceil(start).astype('int')
    end = np.ceil(end).astype('int')
    return start, end


def add_mfa(dataset, alignment_path, split):
    alignments = []
    alignment_path = f"{alignment_path}{split}/outputs/"
    file_ids = [int(f.split('.')[0]) for f in os.listdir(alignment_path) if f.endswith('.TextGrid')]
    file_ids = np.sort(file_ids)
    for ex in range(dataset.num_rows):
        if ex not in file_ids:
            alignments.append(None)           
            continue
        lines = open(f"{alignment_path}{ex}.TextGrid", "r").readlines()
        num_intervals = int(lines[13].strip().split('=')[-1])
        mfa_intervals = []
        for it in range(num_intervals):
            xmin = float(lines[15+it*4].split("=")[-1].strip())
            xmax = float(lines[16+it*4].split("=")[-1].strip())
            text = lines[17+it*4].split("=")[-1].strip()[1:-1]
            if text != "":
                mfa_intervals.append({'start': xmin, 'end': xmax, 'word': text})
        alignments.append(mfa_intervals)
    # add mfa info to the dataset
    dataset = dataset.add_column("mfa_intervals", alignments)
    # filter examples with unsuccessfull mfa 
    dataset = dataset.select(file_ids)
    return dataset