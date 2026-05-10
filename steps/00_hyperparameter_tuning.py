"""
00_hyperparameter_tuning.py
Emplacement : PFA_Sign2Text/steps/00_hyperparameter_tuning.py

Group 0 — Hyperparameter Tuning (runs BEFORE the ablation study)
  Purpose : Find the optimal base hyperparameters that all other groups will use.
  Base    : No CNN, no segmentation, word tokenizer, scratch embedding, D_IN=372
  Method  : One-variable-at-a-time (OVAT) — 7 variables × 3 values = 21 experiments

  Default config (center point, used as fixed value for all other variables):
    D_MODEL=128 | LAYERS=2+2 | LR=3e-4 | BATCH=8 | HEADS=4 | D_FF=512 | DROPOUT=0.3

  Experiments:
    g0_d1  : D_model=64    | g0_d2  : D_model=128*  | g0_d3  : D_model=256
    g0_l1  : Layers=1+1    | g0_l2  : Layers=2+2*   | g0_l3  : Layers=3+3
    g0_r1  : LR=1e-4       | g0_r2  : LR=3e-4*      | g0_r3  : LR=5e-4
    g0_b1  : Batch=4       | g0_b2  : Batch=8*       | g0_b3  : Batch=16
    g0_h1  : Heads=2       | g0_h2  : Heads=4*       | g0_h3  : Heads=8
    g0_f1  : D_ff=256      | g0_f2  : D_ff=512*      | g0_f3  : D_ff=1024
    g0_dr1 : Dropout=0.1   | g0_dr2 : Dropout=0.2    | g0_dr3 : Dropout=0.3*
  (* = default value)

Usage :
    cd PFA_Sign2Text

    # Run all 21 experiments sequentially
    python steps/00_hyperparameter_tuning.py

    # Run a single experiment
    python steps/00_hyperparameter_tuning.py --exp g0_d1

    # Run a specific group only
    python steps/00_hyperparameter_tuning.py --group d
    python steps/00_hyperparameter_tuning.py --group l
    python steps/00_hyperparameter_tuning.py --group r
    python steps/00_hyperparameter_tuning.py --group b
    python steps/00_hyperparameter_tuning.py --group h
    python steps/00_hyperparameter_tuning.py --group f
    python steps/00_hyperparameter_tuning.py --group dr

    # Print summary of completed experiments
    python steps/00_hyperparameter_tuning.py --summary

    # Reset a single experiment and rerun
    python steps/00_hyperparameter_tuning.py --exp g0_d1 --reset

Output :
    results/g0_*/val_metrics.json
    results/g0_*/test_metrics.json
    results/g0_*/train_log.jsonl
    results/g0_*/training_curves.png
    results/g0_summary.json         <- full summary after all experiments
    results/g0_best_config.json     <- optimal hyperparameter values found
    results/g0_summary_figure.png   <- bar chart of all results
"""

import os, sys, json, time, math, argparse, glob
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from sacrebleu.metrics import BLEU as SacreBLEU
from jiwer import wer as jiwer_wer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import re

# ──────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────
BASE        = str(Path(__file__).resolve().parent.parent)
RESULTS_DIR = os.path.join(BASE, 'results')
MODELS_DIR  = os.path.join(BASE, 'models')
W2ID_PATH   = os.path.join(BASE, 'tokenizer_word', 'word2id.json')
ID2W_PATH   = os.path.join(BASE, 'tokenizer_word', 'id2word.json')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device = {device}')

# ──────────────────────────────────────────────────────────────
# FIXED SETTINGS
# ──────────────────────────────────────────────────────────────
D_IN         = 372
USE_CNN      = False   # No CNN — tested later in Group 1
USE_SEGMENTS = True    # Segments — same data distribution as ablation study

# Group 0 specific settings
# Full videos — absolute BLEU will be low (5-20).
# What matters is the RELATIVE ranking between hyperparameter values.
# Early stopping on BLEU-4 — same criterion as ablation study.
MAX_EPOCHS    = 100     # Same as ablation
PATIENCE      = 15      # Same as ablation
BLEU_EVERY    = 2       # Evaluate every 2 epochs — segments converge faster
WARMUP_STEPS  = 100     # Same as ablation
MAX_GRAD_NORM = 1.0
LABEL_SMOOTH  = 0.2     # Same as ablation
WEIGHT_DECAY  = 0.01
NUM_WORKERS   = 0

# ──────────────────────────────────────────────────────────────
# DEFAULT HYPERPARAMETERS (center point)
# ──────────────────────────────────────────────────────────────
DEFAULTS = {
    'd_model' : 128,
    'num_enc' : 2,
    'num_dec' : 2,
    'lr'      : 3e-4,
    'batch'   : 8,
    'n_head'  : 4,
    'd_ff'    : 512,
    'dropout' : 0.3,
}

# ──────────────────────────────────────────────────────────────
# EXPERIMENT GRID — 21 experiments
# ──────────────────────────────────────────────────────────────
EXPERIMENTS = {
    'g0_d1' : {'d_model': 64,               'group': 'd',  'label': 'D_model=64'},
    'g0_d2' : {'d_model': 128,              'group': 'd',  'label': 'D_model=128 [DEFAULT]'},
    'g0_d3' : {'d_model': 256,              'group': 'd',  'label': 'D_model=256'},
    'g0_l1' : {'num_enc': 1, 'num_dec': 1,  'group': 'l',  'label': 'Layers=1+1'},
    'g0_l2' : {'num_enc': 2, 'num_dec': 2,  'group': 'l',  'label': 'Layers=2+2 [DEFAULT]'},
    'g0_l3' : {'num_enc': 3, 'num_dec': 3,  'group': 'l',  'label': 'Layers=3+3'},
    'g0_r1' : {'lr': 1e-4,                  'group': 'r',  'label': 'LR=1e-4'},
    'g0_r2' : {'lr': 3e-4,                  'group': 'r',  'label': 'LR=3e-4 [DEFAULT]'},
    'g0_r3' : {'lr': 5e-4,                  'group': 'r',  'label': 'LR=5e-4'},
    'g0_b1' : {'batch': 4,                  'group': 'b',  'label': 'Batch=4'},
    'g0_b2' : {'batch': 8,                  'group': 'b',  'label': 'Batch=8 [DEFAULT]'},
    'g0_b3' : {'batch': 16,                 'group': 'b',  'label': 'Batch=16'},
    'g0_h1' : {'n_head': 2,                 'group': 'h',  'label': 'Heads=2'},
    'g0_h2' : {'n_head': 4,                 'group': 'h',  'label': 'Heads=4 [DEFAULT]'},
    'g0_h3' : {'n_head': 8,                 'group': 'h',  'label': 'Heads=8'},
    'g0_f1' : {'d_ff': 256,                 'group': 'f',  'label': 'D_ff=256'},
    'g0_f2' : {'d_ff': 512,                 'group': 'f',  'label': 'D_ff=512 [DEFAULT]'},
    'g0_f3' : {'d_ff': 1024,                'group': 'f',  'label': 'D_ff=1024'},
    'g0_dr1': {'dropout': 0.1,              'group': 'dr', 'label': 'Dropout=0.1'},
    'g0_dr2': {'dropout': 0.2,              'group': 'dr', 'label': 'Dropout=0.2'},
    'g0_dr3': {'dropout': 0.3,              'group': 'dr', 'label': 'Dropout=0.3 [DEFAULT]'},
}

GROUP_INFO = {
    'd' : {'name': 'D_model',         'vals': [64,   128,   256]},
    'l' : {'name': 'Layers',          'vals': ['1+1','2+2','3+3']},
    'r' : {'name': 'Learning Rate',   'vals': ['1e-4','3e-4','5e-4']},
    'b' : {'name': 'Batch Size',      'vals': [4,    8,     16]},
    'h' : {'name': 'Attention Heads', 'vals': [2,    4,     8]},
    'f' : {'name': 'D_ff',            'vals': [256,  512,   1024]},
    'dr': {'name': 'Dropout',         'vals': [0.1,  0.2,   0.3]},
}

EXP_ORDER = [
    'g0_d1','g0_d2','g0_d3',
    'g0_l1','g0_l2','g0_l3',
    'g0_r1','g0_r2','g0_r3',
    'g0_b1','g0_b2','g0_b3',
    'g0_h1','g0_h2','g0_h3',
    'g0_f1','g0_f2','g0_f3',
    'g0_dr1','g0_dr2','g0_dr3',
]


def load_best_so_far() -> dict:
    """
    Load best hyperparameter values from all completed groups.
    Each completed group contributes its best value to the running config.
    This implements sequential OVAT with carry-forward:
      Group D best → used as fixed D_model for all subsequent groups
      Group H best → used as fixed N_head for all subsequent groups
      etc.
    """
    best = deepcopy(DEFAULTS)

    # Group ordering — each group updates the carry-forward config
    group_to_hp = {
        'd' : ['d_model'],
        'h' : ['n_head'],
        'r' : ['lr'],
        'b' : ['batch'],
        'dr': ['dropout'],
        'f' : ['d_ff'],
        'l' : ['num_enc', 'num_dec'],
    }

    for grp_key in ['d', 'h', 'r', 'b', 'dr', 'f', 'l']:
        exps = [e for e in EXP_ORDER if EXPERIMENTS[e]['group'] == grp_key]
        best_bleu = -1; best_cfg_update = None
        for exp_id in exps:
            vp = os.path.join(RESULTS_DIR, exp_id, 'val_metrics.json')
            if os.path.exists(vp):
                with open(vp, encoding='utf-8') as f:
                    vm = json.load(f)
                b4 = vm.get('bleu4', 0)
                if b4 > best_bleu:
                    best_bleu = b4
                    best_cfg_update = vm.get('cfg', {})
        if best_cfg_update:
            for hp in group_to_hp[grp_key]:
                if hp in best_cfg_update:
                    best[hp] = best_cfg_update[hp]

    return best


def get_cfg(exp_id: str) -> dict:
    """
    Build config for this experiment using carry-forward best values.
    Only the variable being tested in this experiment changes.
    All other values come from the best found in previously completed groups.
    """
    # Start from carry-forward best (not hard-coded DEFAULTS)
    base = load_best_so_far()

    # Override only the variable this experiment tests
    exp_overrides = {k: v for k, v in EXPERIMENTS[exp_id].items()
                     if k not in ('group', 'label')}
    base.update(exp_overrides)

    # n_head must divide d_model
    while base['d_model'] % base['n_head'] != 0 and base['n_head'] > 1:
        base['n_head'] //= 2

    return base


# ──────────────────────────────────────────────────────────────
# TOKENIZER
# ──────────────────────────────────────────────────────────────
print('Loading word tokenizer...')
with open(W2ID_PATH, encoding='utf-8') as f:
    word2id = json.load(f)
with open(ID2W_PATH, encoding='utf-8') as f:
    id2word = json.load(f)

VOCAB_SIZE = len(word2id)
PAD_ID = word2id.get('<PAD>', 3)
BOS_ID = word2id.get('<BOS>', 1)
EOS_ID = word2id.get('<EOS>', 2)

_HARAKAT = re.compile(r'[\u064B-\u0652\u0670\u0655\u0653\u0654]')
_TATWEEL = re.compile(r'\u0640')

def clean_arabic(text):
    text = _TATWEEL.sub('', text)
    text = _HARAKAT.sub('', text)
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def encode(text):
    return [word2id.get(w, 0) for w in clean_arabic(text).split()]

def decode_ids(ids):
    skip = {BOS_ID, EOS_ID, PAD_ID}
    return ' '.join(id2word.get(str(i), '') for i in ids
                    if i not in skip and id2word.get(str(i), ''))


# ──────────────────────────────────────────────────────────────
# DATASET — full videos, no segmentation
# ──────────────────────────────────────────────────────────────
def augment_landmarks(X):
    X_aug = X.copy() + np.random.normal(0, 0.02, X.shape).astype(np.float32)
    T = X_aug.shape[0]
    if np.random.rand() < 0.5 and T > 4:
        ml    = max(1, min(int(T * np.random.uniform(0.05, 0.15)), T // 3))
        start = np.random.randint(0, max(1, T - ml))
        X_aug[start:start + ml] = 0.0
    return X_aug


class VideoDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        vid, npz_path, text = self.samples[i]
        data = np.load(npz_path)
        X    = data['X'].astype(np.float32)
        mask = data['mask'].astype(np.int8)
        if self.augment:
            X = augment_landmarks(X)
        return {'vid': vid, 'X': X, 'mask': mask, 'text': text}


def collate_fn(batch):
    B     = len(batch)
    D     = batch[0]['X'].shape[1]
    T_max = max(it['X'].shape[0] for it in batch)
    X      = torch.zeros((B, T_max, D), dtype=torch.float32)
    X_mask = torch.zeros((B, T_max), dtype=torch.bool)
    y_raw  = []
    for i, item in enumerate(batch):
        T = item['X'].shape[0]
        X[i, :T]      = torch.from_numpy(item['X'])
        X_mask[i, :T] = torch.from_numpy(item['mask'].astype(np.int64)) > 0
        y_raw.append(item['text'])

    encoded = [encode(t) for t in y_raw]
    max_tl  = max(len(e) for e in encoded) + 2
    y_inp   = torch.full((B, max_tl), PAD_ID, dtype=torch.long)
    y_tgt   = torch.full((B, max_tl), PAD_ID, dtype=torch.long)
    y_mask  = torch.ones((B, max_tl), dtype=torch.bool)
    # Segment texts are short (5-15 words) — no truncation needed
    max_tl = max(len(e) for e in encoded) + 2
    y_inp  = torch.full((B, max_tl), PAD_ID, dtype=torch.long)
    y_tgt  = torch.full((B, max_tl), PAD_ID, dtype=torch.long)
    y_mask = torch.ones((B, max_tl), dtype=torch.bool)
    for i, ids in enumerate(encoded):
        ids_in  = [BOS_ID] + ids + [EOS_ID]
        ids_tgt = ids + [EOS_ID, PAD_ID]
        L = min(len(ids_in), max_tl)
        y_inp[i, :L]  = torch.tensor(ids_in[:L],  dtype=torch.long)
        y_tgt[i, :L]  = torch.tensor(ids_tgt[:L], dtype=torch.long)
        y_mask[i, :L] = False
    return {'X': X, 'X_mask': X_mask,
            'y_inp': y_inp, 'y_tgt': y_tgt,
            'y_mask': y_mask, 'y_raw': y_raw}


def build_loaders(batch_size):
    """
    Overlapping segments — same data distribution as ablation study (Groups 1-4).
    532 videos → 2,359 training segments / 66 → 285 val / 67 → 285 test.
    Each segment has its own matching partial text annotation.
    No CNN applied — CNN effect is tested separately in Group 1.
    """
    split_path = os.path.join(RESULTS_DIR, 'split.json')
    seg_path   = os.path.join(RESULTS_DIR, 'segments_manifest.json')
    with open(split_path, encoding='utf-8') as f:
        split = json.load(f)
    with open(seg_path, encoding='utf-8') as f:
        manifest = json.load(f)

    train_ids = set(split['train'])
    val_ids   = set(split['val'])
    test_ids  = set(split.get('test', []))

    # All segments — each has its own matching text
    all_s = [(e['seg_id'], e['npz_path'], e['text'])
             for e in manifest
             if os.path.exists(e['npz_path']) and e.get('text', '').strip()]

    train_s = [(v,p,t) for v,p,t in all_s if v.rsplit('_seg',1)[0] in train_ids]
    val_s   = [(v,p,t) for v,p,t in all_s if v.rsplit('_seg',1)[0] in val_ids]
    test_s  = [(v,p,t) for v,p,t in all_s if v.rsplit('_seg',1)[0] in test_ids]

    kw = dict(batch_size=batch_size, num_workers=NUM_WORKERS,
               collate_fn=collate_fn, pin_memory=(device.type == 'cuda'))
    train_loader = DataLoader(VideoDataset(train_s, augment=True),  shuffle=True,  **kw)
    val_loader   = DataLoader(VideoDataset(val_s,   augment=False), shuffle=False, **kw)
    test_loader  = DataLoader(VideoDataset(test_s,  augment=False), shuffle=False, **kw) \
                   if test_s else None
    print(f'  Train={len(train_s)} | Val={len(val_s)} | Test={len(test_s)} '
          f'(overlapping segments, no CNN)')
    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────────────────────
# MODEL — no CNN
# ──────────────────────────────────────────────────────────────
class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class VideoEncoder(nn.Module):
    def __init__(self, d_in, d_model, nhead, num_layers, d_ff, dropout):
        super().__init__()
        self.proj    = nn.Linear(d_in, d_model)
        self.pe      = SinusoidalPE(d_model)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.ln      = nn.LayerNorm(d_model)

    def forward(self, X, X_mask_bool):
        pad_mask = ~X_mask_bool
        h   = self.pe(self.proj(X))
        mem = self.encoder(h, src_key_padding_mask=pad_mask)
        return self.ln(mem), pad_mask


class TextDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, d_ff, dropout, pad_id):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pe      = SinusoidalPE(d_model)
        dec_layer    = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.ln      = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, y_inp, mem, mem_kpm, y_pad_mask):
        L   = y_inp.shape[1]
        y   = self.pe(self.embed(y_inp))
        tgt_mask = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=y.device), diagonal=1)
        h = self.decoder(tgt=y, memory=mem, tgt_mask=tgt_mask,
                         tgt_key_padding_mask=y_pad_mask,
                         memory_key_padding_mask=mem_kpm)
        return self.lm_head(self.ln(h))


class Sign2Text(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = VideoEncoder(
            d_in=D_IN, d_model=cfg['d_model'], nhead=cfg['n_head'],
            num_layers=cfg['num_enc'], d_ff=cfg['d_ff'], dropout=cfg['dropout'])
        self.decoder = TextDecoder(
            vocab_size=VOCAB_SIZE, d_model=cfg['d_model'], nhead=cfg['n_head'],
            num_layers=cfg['num_dec'], d_ff=cfg['d_ff'], dropout=cfg['dropout'],
            pad_id=PAD_ID)

    def forward(self, batch):
        X      = batch['X'].to(device)
        X_mask = batch['X_mask'].to(device)
        y_inp  = batch['y_inp'].to(device)
        y_tgt  = batch['y_tgt'].to(device)
        y_mask = batch['y_mask'].to(device)
        mem, kpm = self.encoder(X, X_mask)
        return self.decoder(y_inp, mem, kpm, y_mask), y_tgt


# ──────────────────────────────────────────────────────────────
# LOSS / ACCURACY / SCHEDULER
# ──────────────────────────────────────────────────────────────
class SequenceLoss(nn.Module):
    def __init__(self, pad_id, label_smoothing=0.1):
        super().__init__()
        self.crit = nn.CrossEntropyLoss(ignore_index=pad_id,
                                        label_smoothing=label_smoothing)
    def forward(self, logits, y_tgt):
        B, L, V = logits.shape
        return self.crit(logits.view(B*L, V), y_tgt.view(B*L))


def token_accuracy(logits, y_tgt):
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        mask = (y_tgt != PAD_ID)
        return (pred[mask] == y_tgt[mask]).sum().item() / max(1, mask.sum().item())


class WarmupCosine:
    def __init__(self, optimizer, base_lr, warmup_steps, total_steps, min_lr=1e-6):
        self.opt = optimizer; self.base_lr = base_lr
        self.warmup = max(1, warmup_steps)
        self.total  = max(self.warmup + 1, total_steps)
        self.min_lr = min_lr; self.step_n = 0

    def step(self):
        self.step_n += 1
        if self.step_n <= self.warmup:
            lr = self.base_lr * (self.step_n / float(self.warmup))
        else:
            p  = min((self.step_n-self.warmup)/float(self.total-self.warmup), 1.0)
            lr = self.min_lr + (self.base_lr-self.min_lr)*0.5*(1+math.cos(math.pi*p))
        for g in self.opt.param_groups: g['lr'] = lr

    def get_lr(self): return self.opt.param_groups[0]['lr']


# ──────────────────────────────────────────────────────────────
# GREEDY GENERATION
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def greedy_generate(model, X, X_mask, max_len=80):
    B = X.shape[0]
    mem, mem_kpm = model.encoder(X, X_mask)
    y    = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_len):
        ym  = torch.zeros(B, y.shape[1], dtype=torch.bool, device=device)
        lg  = model.decoder(y, mem, mem_kpm, ym)
        nxt = lg[:, -1, :].argmax(-1, keepdim=True)
        nxt[done] = PAD_ID
        y    = torch.cat([y, nxt], 1)
        done = done | (nxt.squeeze(1) == EOS_ID)
        if done.all(): break
    seqs = []
    for i in range(B):
        ids = y[i, 1:].tolist()
        if EOS_ID in ids: ids = ids[:ids.index(EOS_ID)]
        seqs.append([t for t in ids if t not in (PAD_ID, BOS_ID, EOS_ID)])
    return seqs


# ──────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────
def compute_metrics(hyps, refs):
    res = {}
    for n in [1, 2, 3, 4]:
        m = SacreBLEU(max_ngram_order=n, tokenize='char')
        res[f'bleu{n}'] = round(float(m.corpus_score(hyps, [refs]).score), 2)
    ch = [' '.join(list(h)) if h else ' ' for h in hyps]
    cr = [' '.join(list(r)) if r else ' ' for r in refs]
    try:    res['cer'] = round(jiwer_wer(cr, ch)*100, 2)
    except: res['cer'] = 100.0
    return res


@torch.no_grad()
def run_inference(model, loader, show=2):
    model.eval()
    hyps, refs, exs = [], [], []
    for batch in loader:
        X  = batch['X'].to(device)
        Xm = batch['X_mask'].to(device)
        yr = batch['y_raw']
        seqs = greedy_generate(model, X, Xm)
        for ids, ref in zip(seqs, yr):
            pred = decode_ids(ids) if ids else ''
            hyps.append(pred); refs.append(ref)
            if len(exs) < show: exs.append({'pred': pred, 'ref': ref})
    return hyps, refs, exs


# ──────────────────────────────────────────────────────────────
# TRAIN / EVAL
# ──────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scheduler):
    model.train()
    total_loss = total_acc = total_tok = 0
    t0 = time.time()
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        logits, y_tgt = model(batch)
        loss = criterion(logits, y_tgt)
        loss.backward()
        clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step(); scheduler.step()
        with torch.no_grad():
            acc  = token_accuracy(logits, y_tgt)
            ntok = (y_tgt != PAD_ID).sum().item()
        total_loss += loss.item()*ntok
        total_acc  += acc*ntok
        total_tok  += ntok
    return (total_loss/max(1,total_tok),
            total_acc/max(1,total_tok),
            time.time()-t0)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = total_acc = total_tok = 0
    for batch in loader:
        logits, y_tgt = model(batch)
        loss  = criterion(logits, y_tgt)
        acc   = token_accuracy(logits, y_tgt)
        ntok  = (y_tgt != PAD_ID).sum().item()
        total_loss += loss.item()*ntok
        total_acc  += acc*ntok
        total_tok  += ntok
    return total_loss/max(1,total_tok), total_acc/max(1,total_tok)


# ──────────────────────────────────────────────────────────────
# TRAINING CURVES
# ──────────────────────────────────────────────────────────────
def plot_curves(exp_id, log_path, curves_path):
    if not os.path.exists(log_path): return
    rows = []
    with open(log_path, encoding='utf-8') as f:
        for line in f:
            try: rows.append(json.loads(line))
            except: pass
    if not rows: return
    df  = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'{exp_id} — {EXPERIMENTS[exp_id]["label"]}', fontsize=10)
    axes[0].plot(df['epoch'], df['train_loss'], label='Train', color='steelblue')
    axes[0].plot(df['epoch'], df['val_loss'],   label='Val',   color='tomato', ls='--')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(df['epoch'], df['train_acc'], label='Train', color='steelblue')
    axes[1].plot(df['epoch'], df['val_acc'],   label='Val',   color='tomato', ls='--')
    axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    bdf = df[df['val_bleu4'].notna()]
    if len(bdf):
        axes[2].plot(bdf['epoch'], bdf['val_bleu4'], 'o-', color='#378ADD', linewidth=2)
        bi = bdf['val_bleu4'].idxmax()
        axes[2].axvline(bdf.loc[bi,'epoch'], color='red', ls='--', alpha=0.5)
        axes[2].annotate(f'Best:{bdf.loc[bi,"val_bleu4"]:.1f}',
                         xy=(bdf.loc[bi,'epoch'], bdf.loc[bi,'val_bleu4']),
                         xytext=(bdf.loc[bi,'epoch']+1, bdf.loc[bi,'val_bleu4']-1),
                         fontsize=8, color='red')
    axes[2].set_title('BLEU-4'); axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(curves_path, dpi=130, bbox_inches='tight'); plt.close()


# ──────────────────────────────────────────────────────────────
# RUN ONE EXPERIMENT
# ──────────────────────────────────────────────────────────────
def run_experiment(exp_id: str, reset: bool = False):
    cfg   = get_cfg(exp_id)
    label = EXPERIMENTS[exp_id]['label']

    exp_dir  = os.path.join(RESULTS_DIR, exp_id)
    ckpt_dir = os.path.join(MODELS_DIR,  exp_id)
    log_path = os.path.join(exp_dir, 'train_log.jsonl')
    crv_path = os.path.join(exp_dir, 'training_curves.png')
    val_path = os.path.join(exp_dir, 'val_metrics.json')
    tst_path = os.path.join(exp_dir, 'test_metrics.json')

    for d in [exp_dir, ckpt_dir]: os.makedirs(d, exist_ok=True)

    # Skip if already done
    if not reset and os.path.exists(val_path):
        with open(val_path, encoding='utf-8') as f: vm = json.load(f)
        print(f'  {exp_id} already done — Val BLEU-4={vm.get("bleu4","?")} (skip)')
        return vm

    if reset:
        for p in [log_path, crv_path, val_path, tst_path]:
            if os.path.exists(p): os.remove(p)
        for f in glob.glob(os.path.join(ckpt_dir, '*.pt')): os.remove(f)

    carry = load_best_so_far()
    print(f'\n{"="*60}')
    print(f'  {exp_id} — {label}')
    print(f'  D={cfg["d_model"]} | L={cfg["num_enc"]}+{cfg["num_dec"]} | '
          f'LR={cfg["lr"]:.0e} | B={cfg["batch"]} | '
          f'H={cfg["n_head"]} | Dff={cfg["d_ff"]} | Drop={cfg["dropout"]}')
    print(f'  [Carry-forward base: D={carry["d_model"]} H={carry["n_head"]} '
          f'LR={carry["lr"]:.0e} B={carry["batch"]} '
          f'Dff={carry["d_ff"]} Drop={carry["dropout"]}]')
    print(f'{"="*60}')

    train_loader, val_loader, test_loader = build_loaders(cfg['batch'])
    model     = Sign2Text(cfg).to(device)
    n_params  = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'  Parameters: {n_params:.2f}M')

    criterion = SequenceLoss(PAD_ID, label_smoothing=LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg['lr'], weight_decay=WEIGHT_DECAY)
    scheduler = WarmupCosine(optimizer, base_lr=cfg['lr'],
                             warmup_steps=WARMUP_STEPS,
                             total_steps=MAX_EPOCHS * len(train_loader))

    best_bleu  = 0.0
    no_improve = 0      # patience counter on val BLEU-4 (same as ablation)
    best_path  = None

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc, tr_t = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        val_bleu4 = None
        if epoch % BLEU_EVERY == 0 or epoch == 1:
            hyps, refs, exs = run_inference(model, val_loader, show=2)
            m = compute_metrics(hyps, refs)
            val_bleu4 = m['bleu4']
            print(f'  Ep{epoch:03d} | TL={tr_loss:.4f} VL={val_loss:.4f} | '
                  f'B4={val_bleu4:.2f} | patience={no_improve}/{PATIENCE} | {tr_t/60:.1f}min')
            for ex in exs:
                ref_short = ex["ref"][:80] + ('...' if len(ex["ref"]) > 80 else '')
                print(f'    REF: {ref_short}')
                print(f'    PRD: {ex["pred"][:80] or "(empty)"}')

            if val_bleu4 > best_bleu:
                best_bleu = val_bleu4; no_improve = 0
                if best_path and os.path.exists(best_path): os.remove(best_path)
                best_path = os.path.join(ckpt_dir,
                    f'best_bleu{val_bleu4:.2f}_ep{epoch:03d}.pt')
                torch.save({'model_state': model.state_dict(),
                            'epoch': epoch, 'val_bleu4': val_bleu4,
                            'val_loss': float(val_loss),
                            'cfg': cfg, 'exp_id': exp_id}, best_path)
            else:
                no_improve += BLEU_EVERY
        else:
            no_improve += 1

        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'epoch': epoch,
                'train_loss': round(tr_loss, 6), 'train_acc': round(tr_acc, 6),
                'val_loss': round(float(val_loss), 6), 'val_acc': round(float(val_acc), 6),
                'val_bleu4': val_bleu4,
            }, ensure_ascii=False) + '\n')

        if epoch % 10 == 0: plot_curves(exp_id, log_path, crv_path)

        if no_improve >= PATIENCE:
            print(f'  Early stopping ep{epoch}. Best BLEU-4={best_bleu:.2f}')
            break

    plot_curves(exp_id, log_path, crv_path)

    if best_path and os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])

    # Final val
    val_loss_f, _ = evaluate(model, val_loader, criterion)
    hyps_v, refs_v, exs_v = run_inference(model, val_loader, show=3)
    vm = compute_metrics(hyps_v, refs_v)
    vm.update({'val_loss': round(float(val_loss_f), 6),
               'exp_id': exp_id, 'label': label,
               'group': EXPERIMENTS[exp_id]['group'],
               'cfg': cfg, 'n_params_M': round(n_params, 3),
               'use_cnn': False, 'use_segments': False,
               'tokenizer': 'word', 'embedding': 'scratch',
               'examples': exs_v})
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(vm, f, ensure_ascii=False, indent=2)

    if test_loader:
        test_loss_f, _ = evaluate(model, test_loader, criterion)
        hyps_t, refs_t, exs_t = run_inference(model, test_loader, show=3)
        tm = compute_metrics(hyps_t, refs_t)
        tm.update({'test_loss': round(float(test_loss_f), 6),
                   'exp_id': exp_id, 'label': label, 'cfg': cfg, 'examples': exs_t})
        with open(tst_path, 'w', encoding='utf-8') as f:
            json.dump(tm, f, ensure_ascii=False, indent=2)

    print(f'  DONE {exp_id} | Val BLEU-4={vm["bleu4"]:.2f} | CER={vm["cer"]:.1f}%')
    return vm


# ──────────────────────────────────────────────────────────────
# SUMMARY + BEST CONFIG
# ──────────────────────────────────────────────────────────────
def print_summary():
    print(f'\n{"="*70}')
    print('GROUP 0 — HYPERPARAMETER TUNING SUMMARY')
    print(f'{"="*70}')

    results = {}
    for exp_id in EXP_ORDER:
        vp = os.path.join(RESULTS_DIR, exp_id, 'val_metrics.json')
        if os.path.exists(vp):
            with open(vp, encoding='utf-8') as f: results[exp_id] = json.load(f)

    best_per_group = {}
    for grp_key, grp_info in GROUP_INFO.items():
        exps = [e for e in EXP_ORDER if EXPERIMENTS[e]['group'] == grp_key]
        print(f"\n── {grp_info['name']} ──")
        print(f"  {'Exp':<10} {'Label':<35} {'Val B4':>8} {'CER%':>7} {'Params':>8}")
        print(f"  {'-'*72}")
        best_b4 = -1; best_exp = None
        for exp_id in exps:
            if exp_id in results:
                r = results[exp_id]
                b4  = r.get('bleu4', 0)
                cer = r.get('cer', 100)
                np_ = r.get('n_params_M', 0)
                lbl = EXPERIMENTS[exp_id]['label']
                star = ' ★' if b4 > best_b4 else ''
                print(f"  {exp_id:<10} {lbl:<35} {b4:>8.2f} {cer:>7.1f} {np_:>7.2f}M{star}")
                if b4 > best_b4: best_b4 = b4; best_exp = exp_id
            else:
                print(f"  {exp_id:<10} {EXPERIMENTS[exp_id]['label']:<35} {'—':>8}")
        if best_exp:
            best_per_group[grp_key] = {
                'exp_id': best_exp, 'label': EXPERIMENTS[best_exp]['label'],
                'val_bleu4': results[best_exp].get('bleu4', 0),
                'cfg': results[best_exp].get('cfg', {}),
            }

    print(f'\n{"="*70}')
    print('OPTIMAL CONFIGURATION FOR 06_train_experiments.py')
    print(f'{"="*70}')
    best_cfg = deepcopy(DEFAULTS)
    for grp_key, best in best_per_group.items():
        cfg = best['cfg']
        print(f"  {GROUP_INFO[grp_key]['name']:<22}: {best['label']} "
              f"(Val B4={best['val_bleu4']:.2f})")
        if grp_key == 'd':  best_cfg['d_model']  = cfg.get('d_model',  best_cfg['d_model'])
        if grp_key == 'l':
            best_cfg['num_enc'] = cfg.get('num_enc', best_cfg['num_enc'])
            best_cfg['num_dec'] = cfg.get('num_dec', best_cfg['num_dec'])
        if grp_key == 'r':  best_cfg['lr']       = cfg.get('lr',       best_cfg['lr'])
        if grp_key == 'b':  best_cfg['batch']     = cfg.get('batch',    best_cfg['batch'])
        if grp_key == 'h':  best_cfg['n_head']    = cfg.get('n_head',   best_cfg['n_head'])
        if grp_key == 'f':  best_cfg['d_ff']      = cfg.get('d_ff',     best_cfg['d_ff'])
        if grp_key == 'dr': best_cfg['dropout']   = cfg.get('dropout',  best_cfg['dropout'])

    print(f'\n  Copy to 06_train_experiments.py:')
    print(f'  ─────────────────────────────────')
    print(f'  D_MODEL  = {best_cfg["d_model"]}')
    print(f'  NUM_ENC  = {best_cfg["num_enc"]}')
    print(f'  NUM_DEC  = {best_cfg["num_dec"]}')
    print(f'  LR       = {best_cfg["lr"]:.0e}')
    print(f'  BATCH    = {best_cfg["batch"]}')
    print(f'  N_HEAD   = {best_cfg["n_head"]}')
    print(f'  D_FF     = {best_cfg["d_ff"]}')
    print(f'  DROPOUT  = {best_cfg["dropout"]}')

    summary = {'completed': len(results), 'total': len(EXP_ORDER),
               'best_per_group': best_per_group, 'optimal_cfg': best_cfg,
               'results': {k: {'bleu4': v.get('bleu4'), 'cer': v.get('cer'),
                               'label': v.get('label'), 'cfg': v.get('cfg')}
                           for k, v in results.items()}}
    with open(os.path.join(RESULTS_DIR, 'g0_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(RESULTS_DIR, 'g0_best_config.json'), 'w', encoding='utf-8') as f:
        json.dump(best_cfg, f, ensure_ascii=False, indent=2)
    print(f'\n  Saved: results/g0_summary.json + results/g0_best_config.json')
    return best_cfg


# ──────────────────────────────────────────────────────────────
# SUMMARY FIGURE
# ──────────────────────────────────────────────────────────────
def plot_summary():
    results = {}
    for exp_id in EXP_ORDER:
        vp = os.path.join(RESULTS_DIR, exp_id, 'val_metrics.json')
        if os.path.exists(vp):
            with open(vp, encoding='utf-8') as f: results[exp_id] = json.load(f)
    if not results: return

    grp_colors = {'d':'#378ADD','l':'#EF9F27','r':'#7F77DD',
                  'b':'#1D9E75','h':'#E05C5C','f':'#9E6B1D','dr':'#5C9E8A'}
    fig, axes = plt.subplots(1, 7, figsize=(28, 6), sharey=False)
    fig.suptitle('Group 0 — Hyperparameter Tuning: Val BLEU-4', fontsize=13)

    for ax, (grp_key, grp_info) in zip(axes, GROUP_INFO.items()):
        exps  = [e for e in EXP_ORDER if EXPERIMENTS[e]['group'] == grp_key]
        vals  = [results[e]['bleu4'] if e in results else 0 for e in exps]
        lbls  = [EXPERIMENTS[e]['label'].replace('[DEFAULT]','*').split('=')[-1]
                 for e in exps]
        color = grp_colors[grp_key]
        bars  = ax.bar(range(len(exps)), vals, color=color+'bb',
                       edgecolor=color, linewidth=0.8, width=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                        f'{val:.1f}', ha='center', va='bottom',
                        fontsize=9, fontweight='bold')
        ax.set_xticks(range(len(exps)))
        ax.set_xticklabels(lbls, fontsize=8)
        ax.set_title(grp_info['name'], fontsize=9, fontweight='bold')
        ax.set_ylabel('Val BLEU-4'); ax.grid(True, axis='y', alpha=0.3)
        valid = [v for v in vals if v > 0]
        if valid: ax.set_ylim(0, max(valid)*1.35)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, 'g0_summary_figure.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Figure → {out}')


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',     type=str, default=None,
                        help='Single experiment id (e.g. g0_d1)')
    parser.add_argument('--group',   type=str, default=None,
                        help='Run one variable group (d/l/r/b/h/f/dr)')
    parser.add_argument('--summary', action='store_true',
                        help='Print summary of all completed experiments')
    parser.add_argument('--reset',   action='store_true',
                        help='Reset and rerun (use with --exp or --group)')
    args = parser.parse_args()

    if args.summary:
        print_summary(); plot_summary(); sys.exit(0)

    if args.exp:
        if args.exp not in EXPERIMENTS:
            print(f'Unknown: {args.exp}. Valid: {list(EXPERIMENTS.keys())}')
            sys.exit(1)
        run_experiment(args.exp, reset=args.reset)
        print_summary(); sys.exit(0)

    if args.group:
        grp = args.group.lower()
        if grp not in GROUP_INFO:
            print(f'Unknown group: {grp}. Valid: {list(GROUP_INFO.keys())}')
            sys.exit(1)
        to_run = [e for e in EXP_ORDER if EXPERIMENTS[e]['group'] == grp]
        print(f'\nRunning group {grp.upper()} — {GROUP_INFO[grp]["name"]} '
              f'({len(to_run)} experiments)')
        for exp_id in to_run: run_experiment(exp_id, reset=args.reset)
        print_summary(); sys.exit(0)

    # Run all 21
    print(f'\n{"="*60}')
    print(f'GROUP 0 — HYPERPARAMETER TUNING  |  21 experiments')
    print(f'No CNN | No segmentation | Word tokenizer | Scratch')
    print(f'Device: {device}')
    print(f'{"="*60}')

    for i, exp_id in enumerate(EXP_ORDER, 1):
        run_experiment(exp_id, reset=args.reset)
        print(f'  Progress: {i}/21')

    print_summary()
    plot_summary()
    print('\nGroup 0 complete.')
    print('Copy the optimal values shown above into 06_train_experiments.py')
