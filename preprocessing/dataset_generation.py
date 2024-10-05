import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__), "..")))

SAVE_DATA_PATH = f"{os.environ['HOME']}/Projects/disentangling_representations/directory/data/"
SEED = 42
MAX_SECOND = 14.0

import re
import parselmouth
from parselmouth.praat import call
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
from datasets import Audio, Value, ClassLabel, DatasetDict, Dataset
from datasets import load_dataset, concatenate_datasets
import pandas as pd 
from sklearn.model_selection import train_test_split
import json

if not os.path.exists(SAVE_DATA_PATH):
    os.makedirs(SAVE_DATA_PATH)


def balance(data, key):
    unique_labels, label_counts = np.unique(data[key], return_counts=True)
    min_count = label_counts.min()
    balanced_datasets = []
    for l in unique_labels:
        selected_indices = np.where(np.array(data[key]) == l)[0]
        np.random.shuffle(selected_indices)
        one_class_data = data.select(selected_indices[:min_count])
        balanced_datasets.append(one_class_data)
    balanced_data = concatenate_datasets(balanced_datasets, axis=0).shuffle(seed=SEED)
    return balanced_data

def extract_audio_features(data):         
    features = {'duration': [],
                'mean_intensity': [],
                'mean_pitch': [],
                'std_pitch': [],
                'local_jitter': [],
                'local_shimmer': [],
                }
    
    for ex in range(len(data)):
        duration = len(data[ex]['audio']['array']) / data[ex]['audio']['sampling_rate']
        sound = parselmouth.Sound(data[ex]['audio']['array'], data[ex]['audio']['sampling_rate'])
        unit = "Hertz"
        f0min, f0max = 75, 600 # Hz
        pitch = sound.to_pitch()
        # times = pitch.xs()
        mean_pitch = call(pitch, "Get mean", 0, 0, unit) 
        std_pitch = call(pitch, "Get standard deviation", 0 ,0, unit)
        pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
        local_jitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        local_shimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        mean_intensity = sound.get_intensity()
        
        if pd.isna(mean_pitch):
            mean_pitch = 0
        if pd.isna(std_pitch):
            std_pitch = 0
        if pd.isna(local_jitter):
            local_jitter = 0
        if pd.isna(local_shimmer):
            local_shimmer = 0

        features['duration'].append(duration)
        features['mean_intensity'].append(mean_intensity)
        features['mean_pitch'].append(mean_pitch)
        features['std_pitch'].append(std_pitch)
        features['local_jitter'].append(local_jitter)
        features['local_shimmer'].append(local_shimmer)

    
    data = data.add_column('duration', features['duration'])
    data = data.add_column('mean_intensity', features['mean_intensity'])
    data = data.add_column('mean_pitch', features['mean_pitch'])
    data = data.add_column('std_pitch', features['std_pitch'])
    data = data.add_column('local_jitter', features['local_jitter'])
    data = data.add_column('local_shimmer', features['local_shimmer'])

    return data


# Load IEMOCAP original release
def extract_utterance_id(path):
    return path.split('/')[-1][:-4]

iemocap_org_release_path = "/home/anonymized/IEMOCAP_full_release/"
meta_path = "/home/anonymized/Projects/disentangling_representations/directory/metadata/"
iemocap = []
for session in [1, 2, 3, 4, 5]:
    for split in ['train', 'test']:
        with open(f"{meta_path}Session{session}/{split}_meta_data.json", 'r') as f:
            json_data = json.load(f)
        df = pd.DataFrame(json_data['meta_data'])
        df['path'] = iemocap_org_release_path + df['path']
        df['utterance_id'] = df['path'].apply(extract_utterance_id)
        iemocap.append(df)
iemocap = pd.concat(iemocap)
iemocap = iemocap.drop_duplicates() # s3prl setting use each utterance in 1 set and 4 training set (5 repetition in total)
iemocap = iemocap.rename(columns={'label': 'emotion'})

transcriptions = []
file_names = iemocap['utterance_id'].apply(lambda x: "_".join(x.split('_')[:-1])).values
for file in file_names:
    session = file[4]
    uts = []
    trs = []        
    with open(f"{iemocap_org_release_path}Session{session}/dialog/transcriptions/{file}.txt", 'r') as f:
        for line in f:
            parts = line.split(': ')
            ut = parts[0].split(' ')[0]
            if ut not in iemocap['utterance_id'].values:
                continue
            tr = parts[1].strip()
            uts.append(ut)
            trs.append(tr)
    transcriptions.append(pd.DataFrame({'utterance_id': uts, 
                                        'transcription': trs}))
transcriptions = pd.concat(transcriptions)
transcriptions = transcriptions.drop_duplicates()

# merge data with transcriptions
iemocap = pd.merge(iemocap, transcriptions, on='utterance_id')
def labeling(label):
    if label == "ang":
        return 0
    elif label == "hap":
        return 1
    elif label == "neu":
        return 2
    elif label == "sad":
        return 3
    else:
        return None
iemocap['emotion'] = iemocap['emotion'].apply(labeling)
iemocap = iemocap.drop('utterance_id', axis=1)
iemocap = iemocap.rename(columns={'speaker': 'speaker_id'})
iemocap['gender'] = iemocap['speaker_id'].apply(lambda x: 0 if x[-1] == "F" else 1) # 0 female 1 male
iemocap['speaker_id'], _ = pd.factorize(iemocap['speaker_id'])
iemocap = iemocap.rename(columns={'path': 'audio'})

# convert it to hugging dataset
iemocap = Dataset.from_dict(iemocap).cast_column("audio", Audio(sampling_rate=16000))

# filtering out long audios (> MAX_SECOND)
iemocap = iemocap.filter(lambda example: len(example['audio']['array'])/example['audio']['sampling_rate'] <= MAX_SECOND)

# filtering out emotion clues like: [LAGHTER]
iemocap = iemocap.filter(lambda example: '[' not in example['transcription'] and ']' not in example['transcription'])
# clean transcriptions and make them uppercase
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
iemocap = iemocap.map(lambda example: {'transcription': re.sub(chars_to_ignore_regex, '', example["transcription"]).strip().upper()})


# balancing emotion class with downsampling
iemocap = iemocap.shuffle(seed=SEED)
iemocap = balance(iemocap, key='emotion')
# balanced split based on emotion label, speaker id, and gender
stratify_on = [f"{c1}_{c2}_{c3}" for c1, c2, c3 in zip(iemocap['emotion'], iemocap['speaker_id'], iemocap['gender'])]
train_indices, test_indices = train_test_split(range(len(iemocap)), test_size=0.25, stratify=stratify_on, random_state=SEED)

# speaker id and gender labels are not correct in iemocap files
iemocap = iemocap.remove_columns(["speaker_id", "gender"])

iemocap_dataset = DatasetDict()
iemocap_dataset['train'] = iemocap.select(train_indices)
iemocap_dataset['test'] = iemocap.select(test_indices)

# Extract audio features of IEMOCAP
iemocap_dataset['train'] = extract_audio_features(iemocap_dataset['train'])
iemocap_dataset['test'] = extract_audio_features(iemocap_dataset['test'])

# save              
iemocap_dataset.save_to_disk(f"{SAVE_DATA_PATH}IEMOCAP")



# load RAVDESS
ravdess = load_dataset("narad/ravdess", split='train')
ravdess = ravdess.rename_column("text", "transcription")
ravdess = ravdess.rename_column("labels", "emotion")
ravdess = ravdess.rename_column("speaker_id", "speaker_ids")
ravdess = ravdess.rename_column("speaker_gender", "gender")
# keep only 4 classes: ang, hap, neu, sad
ravdess = ravdess.filter(lambda example: example['emotion'] in [0, 2, 3, 4])
def labeling(example):
    if example["emotion"] == 0:
        example["emotion"] = 2
    elif example["emotion"] == 2:
        example["emotion"] = 1
    elif example["emotion"] == 3:
        example["emotion"] = 3
    elif example["emotion"] == 4:
        example["emotion"] = 0
    else:
        print("Wrong emotion label!")
        return None
    return example
ravdess = ravdess.map(labeling)

# filtering out long audios (> MAX_SECOND)
ravdess = ravdess.filter(lambda example: len(example['audio']['array'])/example['audio']['sampling_rate'] <= MAX_SECOND)

# make transcriptions uppercase
ravdess = ravdess.map(lambda example: {'transcription': example["transcription"].strip().upper()})

# casting
ravdess = ravdess.cast_column("emotion", ClassLabel(num_classes=4, names=['ang', 'hap', 'neu', 'sad']))
ravdess = ravdess.cast_column("audio", Audio(sampling_rate=16000))

# gender and speaker id
ravdess = ravdess.map(lambda example: {"gender": 0 if example["gender"] == "female" else 1})
speaker_ids, _ = pd.factorize(ravdess['speaker_ids'])
ravdess = ravdess.add_column("speaker_id", speaker_ids)
ravdess = ravdess.remove_columns("speaker_ids")
ravdess = ravdess.shuffle(seed=SEED)

# balancing emotion class with downsampling
ravdess = balance(ravdess, key='emotion')
ravdess_dataset = DatasetDict()
ravdess_dataset['test'] = ravdess

# Extract audio features of RAVDESS
ravdess_dataset['test'] = extract_audio_features(ravdess_dataset['test'])

# save
ravdess_dataset.save_to_disk(f"{SAVE_DATA_PATH}RAVDESS")


# load Librispeech
librispeech = DatasetDict()
librispeech['train'], librispeech['test'] = load_dataset("librispeech_asr", split=['train.clean.100', 'test.clean'])
# renaming and removing and casting
librispeech['train'] = librispeech['train'].rename_column("text", "transcription")
librispeech['test'] = librispeech['test'].rename_column("text", "transcription")
librispeech['train'] = librispeech['train'].remove_columns(["speaker_id", "file", "chapter_id", "id"])
librispeech['test'] = librispeech['test'].remove_columns(["speaker_id", "file", "chapter_id", "id"])
librispeech['train'] = librispeech['train'].cast_column("audio", Audio(sampling_rate=16000))
librispeech['test'] = librispeech['test'].cast_column("audio", Audio(sampling_rate=16000))


# filtering out long audios (> MAX_SECOND)
librispeech['train'] = librispeech['train'].filter(lambda example: len(example['audio']['array'])/example['audio']['sampling_rate'] <= MAX_SECOND)
librispeech['test'] = librispeech['test'].filter(lambda example: len(example['audio']['array'])/example['audio']['sampling_rate'] <= MAX_SECOND)

# select a subset of train data due to low memory budjet and also 10h is very sufficient to learn ctc from pretrained w2v2 based on their table 1
librispeech['train'] = librispeech['train'].shuffle(seed=SEED).select(range(4000))
librispeech['test'] = librispeech['test'].shuffle(seed=SEED).select(range(1000))

# Extract audio features of librispeech
librispeech['train'] = extract_audio_features(librispeech['train'])
librispeech['test'] = extract_audio_features(librispeech['test'])

# save
librispeech.save_to_disk(f"{SAVE_DATA_PATH}LibriSpeech")


# load CommonVoice 
commonvoice = load_dataset("mozilla-foundation/common_voice_17_0", "en", split='train', token="anonymized")

# choose only US accent (same as accent in our emotion data) with gender and speaker id info (with more than 200 freq)
unique_ids, counts = np.unique(np.array(commonvoice['client_id']), return_counts=True)
freq_dict = dict(zip(unique_ids, counts))
freqs = np.vectorize(freq_dict.get)(np.array(commonvoice['client_id']))

selected_indices = np.where((np.array(commonvoice['accent']) == 'United States English') & 
                            ((np.array(commonvoice['gender']) == 'male_masculine') | (np.array(commonvoice['gender']) == 'female_feminine')) & 
                            (freqs > 200)
                            )[0]
commonvoice = commonvoice.select(selected_indices)

# renaming and removing and casting
commonvoice = commonvoice.remove_columns(["path", "up_votes", "down_votes", "accent", "age", "locale", "segment", "variant"])
commonvoice = commonvoice.rename_column("sentence", "transcription")
commonvoice = commonvoice.cast_column("audio", Audio(sampling_rate=16000))

# clean transcriptions and make them uppercase
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
commonvoice = commonvoice.map(lambda example: {'transcription': re.sub(chars_to_ignore_regex, '', example["transcription"]).strip().upper()})


# filtering out long audios (> MAX_SECOND)
commonvoice = commonvoice.filter(lambda example: len(example['audio']['array'])/example['audio']['sampling_rate'] <= MAX_SECOND)
commonvoice.save_to_disk(f"{SAVE_DATA_PATH}CommonVoice")

# keep only 24 unique speakers (12 male and 12 female)
male_speakers = commonvoice.select(np.where(np.array(commonvoice['gender']) == 'male_masculine')[0])
unique_male_ids, male_counts = np.unique(np.array(male_speakers['client_id']), return_counts=True)
del male_speakers
female_speakers = commonvoice.select(np.where(np.array(commonvoice['gender']) == 'female_feminine')[0])
unique_female_ids, female_counts = np.unique(np.array(female_speakers['client_id']), return_counts=True)
del female_speakers
# select speakers for which there are at least 200 samples
eligible_male_ids = unique_male_ids[male_counts > 250]
eligible_female_ids = unique_female_ids[female_counts > 250]
# select 12 male and 12 female speakers randomly
np.random.seed(SEED)
selected_male_speakers = np.random.choice(eligible_male_ids, 12, replace=False)
selected_female_speakers = np.random.choice(eligible_female_ids, 12, replace=False)
# filter dataset based on selected speakers
selected_speakers = np.concatenate((selected_male_speakers, selected_female_speakers))
selected_indices = [i for i, client_id in enumerate(commonvoice['client_id']) if client_id in selected_speakers]
commonvoice = commonvoice.select(selected_indices)

# balance by down sampling
commonvoice = balance(commonvoice, key='client_id')

# speaker ids and gender
spk_ids, _ = pd.factorize(commonvoice['client_id'])
commonvoice = commonvoice.add_column('speaker_id', spk_ids)
commonvoice = commonvoice.remove_columns("client_id")
commonvoice = commonvoice.map(lambda x: {'gender': 0 if x['gender'] == "female_feminine" else 1})

# select a subset of data due to low memory budjet and also 10h is very sufficient to learn ctc from pretrained w2v2 based on their table 1
commonvoice = commonvoice.shuffle(seed=SEED).select(range(5000))
# split based on balanced emotion label, speaker id, and gender
stratify_on = [f"{c1}_{c2}" for c1, c2 in zip(commonvoice['speaker_id'], commonvoice['gender'])]
train_indices, test_indices = train_test_split(range(len(commonvoice)), test_size=0.2, stratify=stratify_on, random_state=SEED)
commonvoice_dataset = DatasetDict()
commonvoice_dataset['train'] = commonvoice.select(train_indices)
commonvoice_dataset['test'] = commonvoice.select(test_indices)

# Extract audio features of commonvoice
commonvoice_dataset['train'] = extract_audio_features(commonvoice_dataset['train'])
commonvoice_dataset['test'] = extract_audio_features(commonvoice_dataset['test'])

# save              
commonvoice_dataset.save_to_disk(f"{SAVE_DATA_PATH}CommonVoice")


