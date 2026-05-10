"""
00_split_dataset.py
Usage : python steps/00_split_dataset.py
"""

import os, json, random
from pathlib import Path
import numpy as np

BASE       = str(Path(__file__).resolve().parent.parent)
VIDEOS     = os.path.join(BASE, 'videos')
TEXTS      = os.path.join(BASE, 'texts')
SPLIT_OUT  = os.path.join(BASE, 'results', 'split.json')

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
SEED        = 42

os.makedirs(os.path.join(BASE, 'results'), exist_ok=True)

# Collecter les paires valides
all_videos = sorted([f for f in os.listdir(VIDEOS) if f.lower().endswith('.mp4')])
valid_ids, missing_text = [], []

for v in all_videos:
    vid_id   = os.path.splitext(v)[0]
    txt_path = os.path.join(TEXTS, vid_id + '.txt')
    if os.path.exists(txt_path):
        valid_ids.append(vid_id)
    else:
        missing_text.append(vid_id)

print(f'Videos totales     : {len(all_videos)}')
print(f'Paires valides     : {len(valid_ids)}')
print(f'Sans texte (exclus): {len(missing_text)}')

# Split
random.seed(SEED)
np.random.seed(SEED)
ids = valid_ids.copy()
random.shuffle(ids)

n       = len(ids)
n_train = int(n * TRAIN_RATIO)
n_val   = int(n * VAL_RATIO)
n_test  = n - n_train - n_val

train_ids = ids[:n_train]
val_ids   = ids[n_train:n_train + n_val]
test_ids  = ids[n_train + n_val:]

print(f'\nTrain : {len(train_ids)} ({len(train_ids)/n:.1%})')
print(f'Val   : {len(val_ids)}   ({len(val_ids)/n:.1%})')
print(f'Test  : {len(test_ids)}  ({len(test_ids)/n:.1%})')

assert len(set(train_ids) & set(val_ids))  == 0
assert len(set(train_ids) & set(test_ids)) == 0
assert len(set(val_ids)   & set(test_ids)) == 0

# Sauvegarde
if os.path.exists(SPLIT_OUT):
    print(f'\n split.json existe deja. Supprimer manuellement pour recreer.')
else:
    split = {
        'seed' : SEED,
        'ratio': {'train': TRAIN_RATIO, 'val': VAL_RATIO, 'test': TEST_RATIO},
        'train': train_ids,
        'val'  : val_ids,
        'test' : test_ids,
    }
    with open(SPLIT_OUT, 'w', encoding='utf-8') as f:
        json.dump(split, f, ensure_ascii=False, indent=2)
    print(f'\nsplit.json cree : {SPLIT_OUT}')