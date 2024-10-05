import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__), "../..")))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--DATA")
parser.add_argument("--SPLIT")
args = parser.parse_args()

DATA = args.DATA
SPLIT = args.SPLIT

# DATA = "IEMOCAP"
# SPLIT = "test" 

DATA_PATH = f"{os.environ['HOME']}/Projects/disentangling_representations/directory/data/"
CORPUS_PATH = f"{os.environ['HOME']}/Projects/disentangling_representations/directory/mfa/{DATA}/{SPLIT}/"

# imports 
from tqdm.auto import tqdm
from datasets import load_from_disk, Audio
import subprocess

if not os.path.exists(CORPUS_PATH):
    os.makedirs(CORPUS_PATH)

# load audio dataset
data = load_from_disk(f"{DATA_PATH}{DATA}")[SPLIT]
data = data.select_columns(['audio', 'emotion', 'transcription'])
data = data.cast_column("audio", Audio(sampling_rate=16_000))


# build corpus
iemocap_org_release_path = "/home/anonymized/IEMOCAP_full_release/"
progress_bar = tqdm(range(data.num_rows))
for ex in range(data.num_rows):
    
    # audio
    path = data[ex]['audio']['path']
    # get full audio path
    result = subprocess.run(
        ["find", "/home/anonymized/IEMOCAP_full_release/", "-type", "f", "-name", path],
        stdout=subprocess.PIPE,
        text=True
    )
    audio_path = result.stdout.strip()
    # copy to local
    os.system(f"cp {audio_path} {CORPUS_PATH}{ex}.{audio_path.split('.')[-1]}")
    
    # text
    transcript = data[ex]['transcription']
    file = open(f"{CORPUS_PATH}{ex}.txt", 'w')
    file.write(transcript)
    file.close()

    progress_bar.update(1)