"""
06_train_experiments.py
Emplacement : PFA_Sign2Text/steps/06_train_experiments.py

Script UNIQUE pour lancer TOUS les experiments du plan d'ablation.

Groupe 1 — Architecture
  g1_e1 : scratch, no CNN, no segmentation       → landmarks/  (372 dims)
  g1_e2 : scratch, no CNN, + segmentation        → segments/   (372 dims)
  g1_e3 : scratch, + CNN, + segmentation         → segments/   (372 dims)  [BASELINE]

Groupe 2 — Landmarks / Face
  g2_e1 : 468 face pts complets                  → segments_full/ (1629 dims)
  g2_e2 : 49 face pts selectionnes               → segments/      (372 dims)
  g2_e3 : mains + pose seulement                 → segments/      (225 dims, filtre runtime)

Groupe 3 — Tokenizer
  g3_e1 : word tokenizer           → segments/ (372 dims)
  g3_e2 : BPE 2k vocab             → segments/ (372 dims)
  g3_e3 : BPE 4k vocab             → segments/ (372 dims)

Groupe 4 — Embedding & Fine-tuning
  g4_e1  : scratch embedding        → segments/ (372 dims)
  g4_e2  : FastText, unfreeze ep30  → segments/ (372 dims)
  g4_e3a : AraBERT fully frozen     → segments/ (372 dims)
  g4_e3b : AraBERT unfreeze ep30    → segments/ (372 dims)
  g4_e3c : AraBERT fully trainable  → segments/ (372 dims)

Usage :
    python steps/06_train_experiments.py --exp g1_e1
    python steps/06_train_experiments.py --group 4
    python steps/06_train_experiments.py --all
    python steps/06_train_experiments.py --list
    python steps/06_train_experiments.py --exp g1_e1 --reset

Installation :
    pip install torch sacrebleu jiwer matplotlib pandas
    pip install transformers   # pour g4_e3a/b/c
"""

import os, sys, json, time, math, argparse, glob, shutil, importlib.util
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sacrebleu.metrics import BLEU as SacreBLEU
from jiwer import wer as jiwer_wer

# ──────────────────────────────────────────────────────────────
# CHEMINS GLOBAUX
# ──────────────────────────────────────────────────────────────
BASE        = str(Path(__file__).resolve().parent.parent)
RESULTS_DIR = os.path.join(BASE, 'results')
MODELS_DIR  = os.path.join(BASE, 'models')
BEST_DIR    = os.path.join(MODELS_DIR, 'best')

sys.path.insert(0, BASE)

for _d in [RESULTS_DIR, MODELS_DIR, BEST_DIR]:
    os.makedirs(_d, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device = {device}')

# ──────────────────────────────────────────────────────────────
# HYPERPARAMETRES COMMUNS
# ──────────────────────────────────────────────────────────────
D_MODEL         = 128
N_HEAD          = 4
NUM_ENC         = 2
NUM_DEC         = 2
D_FF            = 512
DROPOUT         = 0.3
LR              = 3e-4
WEIGHT_DECAY    = 1e-2
LABEL_SMOOTHING = 0.2
NUM_EPOCHS      = 100
WARMUP_STEPS    = 100
MAX_GRAD_NORM   = 1.0
PATIENCE        = 15
BLEU_EVERY      = 2
BATCH_SIZE      = 8
NUM_WORKERS     = 0

# ──────────────────────────────────────────────────────────────
# DEFINITION DES EXPERIMENTS
#
# face_mode       : 'keep_all' = no filtering | 'hands_pose' = cols 0..224 only
# use_full_lm     : True = segments_full/ (1629 dims) | False = segments/ (372 dims)
# unfreeze_epoch  : epoch after which embedding is unfrozen
#                   0    = trainable from start
#                   30   = progressive unfreeze at epoch 30
#                   9999 = never unfrozen (fully frozen)
# ──────────────────────────────────────────────────────────────
EXPERIMENTS = {

    # ── Groupe 1 : Architecture ────────────────────────────────
    'g1_e1': {
        'group'          : 1,
        'name'           : 'Scratch, no CNN, no segmentation',
        'use_cnn'        : False,
        'use_segments'   : False,
        'tok_kind'       : 'word',
        'emb_kind'       : 'scratch',
        'face_mode'      : 'keep_all',
        'use_full_lm'    : False,
        'unfreeze_epoch' : 30,
    },
    'g1_e2': {
        'group'          : 1,
        'name'           : 'Scratch, no CNN, + segmentation',
        'use_cnn'        : False,
        'use_segments'   : True,
        'tok_kind'       : 'word',
        'emb_kind'       : 'scratch',
        'face_mode'      : 'keep_all',
        'use_full_lm'    : False,
        'unfreeze_epoch' : 30,
    },
    'g1_e3': {
        'group'          : 1,
        'name'           : 'Scratch, + CNN, + segmentation  [BASELINE]',
        'use_cnn'        : True,
        'use_segments'   : True,
        'tok_kind'       : 'word',
        'emb_kind'       : 'scratch',
        'face_mode'      : 'keep_all',
        'use_full_lm'    : False,
        'unfreeze_epoch' : 30,
    },

    # ── Groupe 2 : Landmarks / Face ────────────────────────────
    'g2_e1': {
        'group'          : 2,
        'name'           : 'Full face 468 pts (D=1629)',
        'use_cnn'        : True,
        'use_segments'   : True,
        'tok_kind'       : 'word',
        'emb_kind'       : 'scratch',
        'face_mode'      : 'keep_all',
        'use_full_lm'    : True,
        'unfreeze_epoch' : 30,
    },
    'g2_e2': {
        'group'          : 2,
        'name'           : '49 face pts selectionnes (D=372)',
        'use_cnn'        : True,
        'use_segments'   : True,
        'tok_kind'       : 'word',
        'emb_kind'       : 'scratch',
        'face_mode'      : 'keep_all',
        'use_full_lm'    : False,
        'unfreeze_epoch' : 30,
    },
    'g2_e3': {
        'group'          : 2,
        'name'           : 'Mains + pose seulement (D=225)',
        'use_cnn'        : True,
        'use_segments'   : True,
        'tok_kind'       : 'word',
        'emb_kind'       : 'scratch',
        'face_mode'      : 'hands_pose',
        'use_full_lm'    : False,
        'unfreeze_epoch' : 30,
    },

    # ── Groupe 3 : Tokenizer ───────────────────────────────────
    'g3_e1': {
        'group'          : 3,
        'name'           : 'Word tokenizer  [BASELINE]',
        'use_cnn'        : True,
        'use_segments'   : True,
        'tok_kind'       : 'word',
        'emb_kind'       : 'scratch',
        'face_mode'      : 'keep_all',
        'use_full_lm'    : False,
        'unfreeze_epoch' : 30,
    },
    'g3_e2': {
        'group'          : 3,
        'name'           : 'BPE 2k vocab',
        'use_cnn'        : True,
        'use_segments'   : True,
        'tok_kind'       : 'bpe2k',
        'emb_kind'       : 'scratch',
        'face_mode'      : 'keep_all',
        'use_full_lm'    : False,
        'unfreeze_epoch' : 30,
    },
    'g3_e3': {
        'group'          : 3,
        'name'           : 'BPE 4k vocab',
        'use_cnn'        : True,
        'use_segments'   : True,
        'tok_kind'       : 'bpe4k',
        'emb_kind'       : 'scratch',
        'face_mode'      : 'keep_all',
        'use_full_lm'    : False,
        'unfreeze_epoch' : 30,
    },

    # ── Groupe 4 : Embedding & Fine-tuning ────────────────────
    'g4_e1': {
        'group'          : 4,
        'name'           : 'Scratch embedding  [BASELINE]',
        'use_cnn'        : True,
        'use_segments'   : True,
        'tok_kind'       : 'word',
        'emb_kind'       : 'scratch',
        'face_mode'      : 'keep_all',
        'use_full_lm'    : False,
        'unfreeze_epoch' : 30,
    },
    'g4_e2': {
        'group'          : 4,
        'name'           : 'FastText embedding, unfreeze ep30',
        'use_cnn'        : True,
        'use_segments'   : True,
        'tok_kind'       : 'word',
        'emb_kind'       : 'fasttext',
        'face_mode'      : 'keep_all',
        'use_full_lm'    : False,
        'unfreeze_epoch' : 30,
    },
    'g4_e3a': {
        'group'          : 4,
        'name'           : 'AraBERT fully frozen',
        'use_cnn'        : True,
        'use_segments'   : True,
        'tok_kind'       : 'word',
        'emb_kind'       : 'arabert',
        'face_mode'      : 'keep_all',
        'use_full_lm'    : False,
        'unfreeze_epoch' : 9999,   # never unfrozen
    },
    'g4_e3b': {
        'group'          : 4,
        'name'           : 'AraBERT progressive unfreeze ep30',
        'use_cnn'        : True,
        'use_segments'   : True,
        'tok_kind'       : 'word',
        'emb_kind'       : 'arabert',
        'face_mode'      : 'keep_all',
        'use_full_lm'    : False,
        'unfreeze_epoch' : 30,    # unfreeze after epoch 30
    },
    'g4_e3c': {
        'group'          : 4,
        'name'           : 'AraBERT fully trainable from ep0',
        'use_cnn'        : True,
        'use_segments'   : True,
        'tok_kind'       : 'word',
        'emb_kind'       : 'arabert',
        'face_mode'      : 'keep_all',
        'use_full_lm'    : False,
        'unfreeze_epoch' : 0,     # trainable from epoch 1
    },
}


# ──────────────────────────────────────────────────────────────
# COMPOSANTS DU MODELE
# ──────────────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class PermuteLayerNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class VideoEncoder(nn.Module):
    def __init__(self, d_in, d_model, nhead, num_layers, d_ff, dropout, use_cnn=True):
        super().__init__()
        self.use_cnn = use_cnn
        self.proj    = nn.Linear(d_in, d_model)
        if use_cnn:
            self.cnn = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, bias=False),
                PermuteLayerNorm(d_model), nn.GELU(), nn.Dropout(dropout * 0.5),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, bias=False),
                nn.GELU(),
            )
        self.pe      = SinusoidalPE(d_model)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.ln      = nn.LayerNorm(d_model)

    def forward(self, X, X_mask_bool):
        pad_mask = ~X_mask_bool
        h = self.proj(X)
        if self.use_cnn:
            h = h + self.cnn(h.transpose(1, 2)).transpose(1, 2)
        h   = self.pe(h)
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

    def set_pretrained_embeddings(self, matrix, freeze=True):
        emb_dim  = matrix.shape[1]
        orig_dim = self.embed.embedding_dim
        self.embed = nn.Embedding.from_pretrained(
            matrix, freeze=freeze, padding_idx=self.embed.padding_idx)
        if emb_dim != orig_dim:
            self.emb_proj = nn.Linear(emb_dim, orig_dim, bias=False)
        else:
            self.emb_proj = None

    def forward(self, y_inp, mem, mem_kpm, y_pad_mask):
        L   = y_inp.shape[1]
        emb = self.embed(y_inp)
        if hasattr(self, 'emb_proj') and self.emb_proj is not None:
            emb = self.emb_proj(emb)
        y        = self.pe(emb)
        tgt_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=y.device), diagonal=1)
        h = self.decoder(tgt=y, memory=mem, tgt_mask=tgt_mask,
                         tgt_key_padding_mask=y_pad_mask,
                         memory_key_padding_mask=mem_kpm)
        return self.lm_head(self.ln(h))


class Sign2Text(nn.Module):
    def __init__(self, d_in, vocab_size, d_model, nhead, num_enc, num_dec,
                 d_ff, dropout, pad_id, use_cnn=True):
        super().__init__()
        self.encoder = VideoEncoder(d_in, d_model, nhead, num_enc, d_ff, dropout, use_cnn)
        self.decoder = TextDecoder(vocab_size, d_model, nhead, num_dec, d_ff, dropout, pad_id)
        self.pad_id  = pad_id

    def forward(self, batch):
        X      = batch['X'].to(device)
        X_mask = batch['X_mask'].to(device)
        y_inp  = batch['y_inp'].to(device)
        y_tgt  = batch['y_tgt'].to(device)
        y_mask = batch['y_mask'].to(device)
        mem, mem_kpm = self.encoder(X, X_mask)
        logits = self.decoder(y_inp, mem, mem_kpm, y_mask)
        return logits, y_tgt


class SequenceLoss(nn.Module):
    def __init__(self, pad_id, label_smoothing=0.2):
        super().__init__()
        self.crit = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=label_smoothing)

    def forward(self, logits, y_tgt):
        B, L, V = logits.shape
        return self.crit(logits.view(B * L, V), y_tgt.view(B * L))


def token_accuracy(logits, y_tgt, pad_id):
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        mask = (y_tgt != pad_id)
        return (pred[mask] == y_tgt[mask]).sum().item() / max(1, mask.sum().item())


# ──────────────────────────────────────────────────────────────
# SCHEDULER
# ──────────────────────────────────────────────────────────────

class WarmupCosine:
    def __init__(self, optimizer, base_lr, warmup_steps, total_steps, min_lr=1e-6):
        self.opt     = optimizer
        self.base_lr = base_lr
        self.warmup  = max(1, warmup_steps)
        self.total   = max(self.warmup + 1, total_steps)
        self.min_lr  = min_lr
        self.step_n  = 0

    def step(self):
        self.step_n += 1
        if self.step_n <= self.warmup:
            lr = self.base_lr * (self.step_n / float(self.warmup))
        else:
            progress = min((self.step_n - self.warmup) / float(self.total - self.warmup), 1.0)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        for g in self.opt.param_groups:
            g['lr'] = lr

    def get_lr(self):
        return self.opt.param_groups[0]['lr']


# ──────────────────────────────────────────────────────────────
# METRIQUES : BLEU-1/2/3/4, ROUGE-L (LCS pur Python), CER
# ──────────────────────────────────────────────────────────────

def _lcs_length(x, y):
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(curr[j-1], prev[j])
        prev = curr
    return prev[n]


def _rouge_l_sentence(hypothesis, reference):
    h = list(hypothesis.strip())
    r = list(reference.strip())
    if not h or not r:
        return 0.0
    lcs  = _lcs_length(h, r)
    prec = lcs / len(h)
    rec  = lcs / len(r)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def compute_all_metrics(hypotheses, references):
    if not hypotheses or not references:
        return dict(bleu1=0, bleu2=0, bleu3=0, bleu4=0,
                    rouge_l=0, cer=0, exact_match=0)

    results = {}

    for n in [1, 2, 3, 4]:
        metric = SacreBLEU(max_ngram_order=n, tokenize='char')
        score  = metric.corpus_score(hypotheses, [references])
        results[f'bleu{n}'] = round(float(score.score), 2)

    rl_scores = [_rouge_l_sentence(h, r) for h, r in zip(hypotheses, references)]
    results['rouge_l'] = round(100.0 * sum(rl_scores) / max(1, len(rl_scores)), 2)

    chars_hyp = [' '.join(list(h)) if h else ' ' for h in hypotheses]
    chars_ref = [' '.join(list(r)) if r else ' ' for r in references]
    try:
        cer_val = jiwer_wer(chars_ref, chars_hyp) * 100
    except Exception:
        cer_val = 100.0
    results['cer'] = round(cer_val, 2)

    exact = sum(h.strip() == r.strip() for h, r in zip(hypotheses, references))
    results['exact_match'] = round(100.0 * exact / max(1, len(hypotheses)), 2)

    return results


# ──────────────────────────────────────────────────────────────
# FILTRAGE COLONNES LANDMARKS A RUNTIME
# ──────────────────────────────────────────────────────────────

def make_face_slice(face_mode):
    if face_mode == 'keep_all':
        return None
    elif face_mode == 'hands_pose':
        return list(range(0, 225))
    return None


# ──────────────────────────────────────────────────────────────
# CHARGEMENT DATALOADER
# ──────────────────────────────────────────────────────────────

def load_dataloader_for_exp(cfg):
    tok_kind      = cfg['tok_kind']
    use_segments  = cfg['use_segments']
    use_full_lm   = cfg.get('use_full_lm', False)

    dl_path = os.path.join(BASE, 'steps', 'dataloader.py')
    if not os.path.exists(dl_path):
        dl_path = os.path.join(BASE, 'steps', '04_build_dataloader_v2.py')

    manifest_orig = os.path.join(BASE, 'results', 'segments_manifest.json')
    manifest_full = os.path.join(BASE, 'results', 'segments_full_manifest.json')
    manifest_bak  = os.path.join(BASE, 'results', 'segments_manifest_backup.json')
    switched = False

    try:
        if use_full_lm and use_segments:
            if not os.path.exists(manifest_full):
                raise FileNotFoundError(
                    'segments_full_manifest.json introuvable.\n'
                    'Lancer : python steps/02d_segment_full_face.py'
                )
            shutil.copy2(manifest_orig, manifest_bak)
            shutil.copy2(manifest_full, manifest_orig)
            switched = True
            print('  Manifeste → segments_full_manifest.json (D=1629)')

        spec   = importlib.util.spec_from_file_location('dataloader_mod', dl_path)
        dl_mod = importlib.util.module_from_spec(spec)
        dl_mod.TOKENIZER_KIND = 'word' if tok_kind == 'word' else 'bpe'
        dl_mod.USE_SEGMENTS   = use_segments

        if tok_kind == 'bpe4k':
            tok4k_dir = os.path.join(BASE, 'tokenizer_4k')
            if not os.path.exists(tok4k_dir):
                raise FileNotFoundError(
                    'tokenizer_4k/ introuvable.\n'
                    'Creer le dossier et lancer 03_build_tokenizer_bpe.py avec VOCAB_SIZE=4000'
                )
            dl_mod.TOK_BPE_DIR = tok4k_dir

        spec.loader.exec_module(dl_mod)

        train_loader, val_loader, test_loader = dl_mod.build_loaders(
            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
            use_segments=use_segments
        )

    finally:
        if switched and os.path.exists(manifest_bak):
            shutil.copy2(manifest_bak, manifest_orig)
            os.remove(manifest_bak)
            print('  Manifeste restaure → segments_manifest.json')

    return (train_loader, val_loader, test_loader,
            dl_mod.PAD_ID, dl_mod.BOS_ID, dl_mod.EOS_ID,
            dl_mod.VOCAB_SIZE, dl_mod.decode_ids)


# ──────────────────────────────────────────────────────────────
# CHARGEMENT EMBEDDING
# ──────────────────────────────────────────────────────────────

def load_embedding_matrix(cfg, word2id, vocab_size):
    kind = cfg['emb_kind']

    if kind == 'scratch':
        return None

    if kind == 'fasttext':
        ft_path = os.path.join(BASE, 'embeddings', 'cc.ar.300.bin')
        if not os.path.exists(ft_path):
            print(f'  [WARN] FastText non trouve : {ft_path}')
            print('         Telecharger : https://fasttext.cc/docs/en/crawl-vectors.html')
            print('         Fallback : scratch embedding.')
            return None
        try:
            import fasttext
            ft      = fasttext.load_model(ft_path)
            emb_dim = ft.get_dimension()
            matrix  = torch.zeros(vocab_size, emb_dim)
            for word, idx in word2id.items():
                if idx < vocab_size:
                    matrix[idx] = torch.tensor(ft.get_word_vector(word))
            print(f'  FastText matrix : {tuple(matrix.shape)}')
            return matrix
        except ImportError:
            print('  [WARN] pip install fasttext requis. Fallback scratch.')
            return None
    if kind == 'arabert':
        # Utiliser la matrice pre-alignee sur notre vocabulaire
        aligned_path = os.path.join(BASE, 'embeddings', 'arabert_aligned.pt')
        if not os.path.exists(aligned_path):
            print(f'  [WARN] arabert_aligned.pt introuvable : {aligned_path}')
            print(f'         Lancer : python steps/prepare_arabert_embeddings.py')
            print(f'         Fallback : scratch embedding.')
            return None
        matrix = torch.load(aligned_path, map_location='cpu')
        print(f'  AraBERT aligned matrix : {tuple(matrix.shape)}')
        return matrix


# ──────────────────────────────────────────────────────────────
# GENERATION GREEDY
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def greedy_generate(model, X, X_mask, bos_id, eos_id, pad_id, max_len=80):
    model.eval()
    B            = X.shape[0]
    mem, mem_kpm = model.encoder(X, X_mask)
    y    = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len):
        y_mask = torch.zeros(B, y.shape[1], dtype=torch.bool, device=device)
        logits = model.decoder(y, mem, mem_kpm, y_mask)
        nxt    = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        nxt[done] = pad_id
        y    = torch.cat([y, nxt], dim=1)
        done = done | (nxt.squeeze(1) == eos_id)
        if done.all():
            break

    seqs = []
    for i in range(B):
        ids = y[i, 1:].tolist()
        if eos_id in ids:
            ids = ids[:ids.index(eos_id)]
        ids = [t for t in ids if t not in (pad_id, bos_id, eos_id)]
        seqs.append(ids)
    return seqs


# ──────────────────────────────────────────────────────────────
# BEAM SEARCH
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def beam_search_generate(model, X, X_mask, bos_id, eos_id, pad_id,
                          beam_size=3, max_len=80):
    """
    Beam search decoding — one sample at a time (B=1).
    Returns list[int] of token IDs for the best sequence.
    """
    model.eval()
    assert X.shape[0] == 1, "beam search: batch size must be 1"

    mem, mem_kpm = model.encoder(X, X_mask)

    # Each beam: (score, token_ids)
    beams = [(0.0, [bos_id])]
    completed = []

    for _ in range(max_len):
        all_candidates = []
        for score, seq in beams:
            if seq[-1] == eos_id:
                completed.append((score, seq))
                continue
            y_inp  = torch.tensor([seq], dtype=torch.long, device=device)
            y_mask = torch.zeros(1, len(seq), dtype=torch.bool, device=device)
            logits = model.decoder(y_inp, mem, mem_kpm, y_mask)
            log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)
            topk_vals, topk_ids = log_probs.topk(beam_size)
            for val, tok in zip(topk_vals.tolist(), topk_ids.tolist()):
                all_candidates.append((score + val, seq + [tok]))

        if not all_candidates:
            break

        # Keep top beam_size
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        beams = all_candidates[:beam_size]

        # If all beams ended
        if all(seq[-1] == eos_id for _, seq in beams):
            completed.extend(beams)
            break

    if completed:
        completed.sort(key=lambda x: x[0] / max(1, len(x[1])), reverse=True)
        best_seq = completed[0][1]
    else:
        best_seq = beams[0][1]

    # Remove BOS/EOS/PAD
    best_seq = [t for t in best_seq[1:] if t not in (bos_id, eos_id, pad_id)]
    return best_seq


@torch.no_grad()
def run_inference(model, loader, decode_ids_fn, bos_id, eos_id, pad_id,
                  max_batches=None, show_examples=3, beam_size=1):
    """
    beam_size=1 : greedy decoding
    beam_size>1 : beam search (slower, processes one sample at a time)
    """
    model.eval()
    hyps, refs, examples = [], [], []

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        X      = batch['X'].to(device)
        X_mask = batch['X_mask'].to(device)
        y_raw  = batch['y_raw']

        if beam_size <= 1:
            seqs = greedy_generate(model, X, X_mask, bos_id, eos_id, pad_id)
        else:
            seqs = []
            for i in range(X.shape[0]):
                seq = beam_search_generate(
                    model, X[i:i+1], X_mask[i:i+1],
                    bos_id, eos_id, pad_id, beam_size=beam_size)
                seqs.append(seq)

        for ids, ref in zip(seqs, y_raw):
            pred = decode_ids_fn(ids) if ids else ''
            hyps.append(pred)
            refs.append(ref)
            if len(examples) < show_examples:
                examples.append({'pred': pred, 'ref': ref})

    return hyps, refs, examples


# ──────────────────────────────────────────────────────────────
# TRAIN / EVAL
# ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scheduler, pad_id, print_every=20):
    model.train()
    total_loss = total_acc = total_tok = 0
    t0 = time.time()

    for it, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        logits, y_tgt = model(batch)
        loss = criterion(logits, y_tgt)
        loss.backward()
        clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            acc  = token_accuracy(logits, y_tgt, pad_id)
            ntok = (y_tgt != pad_id).sum().item()
        total_loss += loss.item() * ntok
        total_acc  += acc * ntok
        total_tok  += ntok

        if it % print_every == 0:
            print(f'  it {it:04d}/{len(loader)} | loss={loss.item():.4f} | '
                  f'acc={acc:.3f} | lr={scheduler.get_lr():.2e}')

    return (total_loss / max(1, total_tok),
            total_acc  / max(1, total_tok),
            time.time() - t0)


@torch.no_grad()
def evaluate(model, loader, criterion, pad_id):
    model.eval()
    total_loss = total_acc = total_tok = 0
    for batch in loader:
        logits, y_tgt = model(batch)
        loss  = criterion(logits, y_tgt)
        acc   = token_accuracy(logits, y_tgt, pad_id)
        ntok  = (y_tgt != pad_id).sum().item()
        total_loss += loss.item() * ntok
        total_acc  += acc * ntok
        total_tok  += ntok
    return total_loss / max(1, total_tok), total_acc / max(1, total_tok)


# ──────────────────────────────────────────────────────────────
# COURBES
# ──────────────────────────────────────────────────────────────

def plot_curves(log_path, out_path, exp_name):
    if not os.path.exists(log_path):
        return
    rows = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if not rows:
        return

    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(exp_name, fontsize=11)

    axes[0].plot(df['epoch'], df['train_loss'], label='Train loss', color='steelblue')
    axes[0].plot(df['epoch'], df['val_loss'],   label='Val loss',   color='tomato', linestyle='--')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(df['epoch'], df['train_acc'], label='Train acc', color='steelblue')
    axes[1].plot(df['epoch'], df['val_acc'],   label='Val acc',   color='tomato', linestyle='--')
    axes[1].set_title('Token accuracy')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    bleu_df = df[df['val_bleu4'].notna()]
    if len(bleu_df) > 0:
        axes[2].plot(bleu_df['epoch'], bleu_df['val_bleu4'],
                     'o-', color='seagreen', linewidth=2, label='Val BLEU-4')
        best_i  = bleu_df['val_bleu4'].idxmax()
        best_ep = bleu_df.loc[best_i, 'epoch']
        best_bl = bleu_df.loc[best_i, 'val_bleu4']
        axes[2].axvline(x=best_ep, color='red', linestyle='--', alpha=0.5)
        axes[2].annotate(f'Best: {best_bl:.1f}\n(ep {int(best_ep)})',
                         xy=(best_ep, best_bl),
                         xytext=(best_ep + 1, max(0, best_bl - 1)),
                         fontsize=8, color='red')
        axes[2].set_title('BLEU-4')
        axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('BLEU-4')
        axes[2].legend(); axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'BLEU pas encore calcule', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Courbes → {out_path}')


# ──────────────────────────────────────────────────────────────
# BOUCLE D'ENTRAINEMENT D'UN EXPERIMENT
# ──────────────────────────────────────────────────────────────

def run_experiment(exp_id, cfg, args):
    exp_dir  = os.path.join(RESULTS_DIR, exp_id)
    ckpt_dir = os.path.join(MODELS_DIR, exp_id)
    os.makedirs(exp_dir,  exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    log_path          = os.path.join(exp_dir, 'train_log.jsonl')
    curves_path       = os.path.join(exp_dir, 'training_curves.png')
    val_metrics_path  = os.path.join(exp_dir, 'val_metrics.json')
    test_metrics_path = os.path.join(exp_dir, 'test_metrics.json')

    unfreeze_ep = cfg.get('unfreeze_epoch', 30)

    print('\n' + '=' * 65)
    print(f'  EXPERIMENT : {exp_id}  |  {cfg["name"]}')
    print(f'  Groupe : {cfg["group"]}')
    print(f'  CNN={cfg["use_cnn"]} | seg={cfg["use_segments"]} | '
          f'tok={cfg["tok_kind"]} | emb={cfg["emb_kind"]} | '
          f'face={cfg["face_mode"]} | full_lm={cfg.get("use_full_lm", False)} | '
          f'unfreeze_ep={unfreeze_ep}')
    print('=' * 65)

    if args.reset:
        for p in [log_path, curves_path, val_metrics_path, test_metrics_path]:
            if os.path.exists(p):
                os.remove(p)
                print(f'  Supprime : {p}')
        for f in glob.glob(os.path.join(ckpt_dir, '*.pt')):
            os.remove(f)

    # ── DataLoader ────────────────────────────────────────────
    print('\n[1/4] Chargement DataLoader...')
    (train_loader, val_loader, test_loader,
     PAD_ID, BOS_ID, EOS_ID, VOCAB_SIZE, decode_ids) = load_dataloader_for_exp(cfg)

    print(f'  Train={len(train_loader.dataset)} | Val={len(val_loader.dataset)} | '
          f'Test={len(test_loader.dataset) if test_loader else 0}')

    # ── Dimension d'entree ────────────────────────────────────
    sample   = next(iter(train_loader))
    D_IN_raw = sample['X'].shape[-1]
    face_slice = make_face_slice(cfg['face_mode'])
    D_IN = len(face_slice) if face_slice is not None else D_IN_raw
    print(f'  D_IN_raw={D_IN_raw} → D_IN effectif={D_IN} (face_mode={cfg["face_mode"]})')

    def apply_face_slice(batch):
        if face_slice is not None:
            batch = dict(batch)
            batch['X'] = batch['X'][:, :, face_slice]
        return batch

    # ── Modele ────────────────────────────────────────────────
    print('\n[2/4] Construction du modele...')
    model = Sign2Text(
        d_in=D_IN, vocab_size=VOCAB_SIZE,
        d_model=D_MODEL, nhead=N_HEAD,
        num_enc=NUM_ENC, num_dec=NUM_DEC,
        d_ff=D_FF, dropout=DROPOUT,
        pad_id=PAD_ID, use_cnn=cfg['use_cnn']
    ).to(device)

    if cfg['emb_kind'] != 'scratch':
        word2id = {}
        try:
            w2i_path = os.path.join(BASE, 'tokenizer_word', 'word2id.json')
            with open(w2i_path, 'r', encoding='utf-8') as f:
                word2id = json.load(f)
        except Exception:
            pass
        emb_matrix = load_embedding_matrix(cfg, word2id, VOCAB_SIZE)
        if emb_matrix is not None:
            emb_matrix = emb_matrix.to(device)
            model.decoder.set_pretrained_embeddings(emb_matrix, freeze=True)
            model.decoder.to(device)  # move emb_proj to GPU too
            if unfreeze_ep == 0:
                # Fully trainable from start
                model.decoder.embed.weight.requires_grad = True
                print(f'  Embedding pre-entraine ({cfg["emb_kind"]}), '
                      f'entierement entrainable depuis ep0.')
            elif unfreeze_ep >= 9999:
                print(f'  Embedding pre-entraine ({cfg["emb_kind"]}), '
                      f'entierement gele (jamais degele).')
            else:
                print(f'  Embedding pre-entraine ({cfg["emb_kind"]}), '
                      f'gele — degel prevu a epoch {unfreeze_ep}.')

    total_p = sum(p.numel() for p in model.parameters())
    print(f'  Parametres : {total_p/1e6:.2f}M')

    criterion   = SequenceLoss(PAD_ID, label_smoothing=LABEL_SMOOTHING).to(device)
    optimizer   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = NUM_EPOCHS * len(train_loader)
    scheduler   = WarmupCosine(optimizer, base_lr=LR,
                                warmup_steps=WARMUP_STEPS, total_steps=total_steps)

    # ── Boucle d'entrainement ─────────────────────────────────
    print('\n[3/4] Entrainement...')
    best_bleu         = 0.0
    epochs_no_improve = 0
    best_path         = None
    unfreeze_done     = (unfreeze_ep == 0)  # already done if ep0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f'\n===== Epoch {epoch}/{NUM_EPOCHS} =====')

        # Progressive unfreeze logic
        if (cfg['emb_kind'] != 'scratch'
                and not unfreeze_done
                and unfreeze_ep < 9999
                and epoch > unfreeze_ep):
            model.decoder.embed.weight.requires_grad = True
            unfreeze_done = True
            print(f'  Embedding degele (epoch {epoch} > {unfreeze_ep})')

        class FilteredLoader:
            def __init__(self, loader, fn):
                self.loader = loader
                self.fn = fn
            def __iter__(self):
                for b in self.loader:
                    yield self.fn(b)
            def __len__(self):
                return len(self.loader)

        f_train = FilteredLoader(train_loader, apply_face_slice)
        f_val   = FilteredLoader(val_loader,   apply_face_slice)

        tr_loss, tr_acc, tr_time = train_one_epoch(
            model, f_train, optimizer, criterion, scheduler, PAD_ID)
        print(f'TRAIN | loss={tr_loss:.4f} | acc={tr_acc:.3f} | {tr_time/60:.1f}min')

        val_loss, val_acc = evaluate(model, f_val, criterion, PAD_ID)
        print(f'VALID | loss={val_loss:.4f} | acc={val_acc:.3f}')

        val_bleu4 = None
        if epoch % BLEU_EVERY == 0 or epoch == 1:
            hyps, refs, examples = run_inference(
                model, f_val, decode_ids, BOS_ID, EOS_ID, PAD_ID,
                show_examples=3, beam_size=1)
            metrics   = compute_all_metrics(hyps, refs)
            val_bleu4 = metrics['bleu4']
            print(f'  BLEU-4={val_bleu4:.2f} | ROUGE-L={metrics["rouge_l"]:.2f} | '
                  f'CER={metrics["cer"]:.1f}%')

            for j, ex in enumerate(examples, 1):
                print(f'  [{j}] REF  : {ex["ref"]}')
                print(f'       PRED : {ex["pred"] or "(vide)"}')

            if val_bleu4 > best_bleu:
                best_bleu         = val_bleu4
                epochs_no_improve = 0
                if best_path and os.path.exists(best_path):
                    os.remove(best_path)
                best_path = os.path.join(
                    ckpt_dir, f'best_bleu{val_bleu4:.2f}_ep{epoch:03d}.pt')
                torch.save({
                    'model_state'  : model.state_dict(),
                    'optim_state'  : optimizer.state_dict(),
                    'epoch'        : epoch,
                    'val_loss'     : float(val_loss),
                    'val_bleu4'    : val_bleu4,
                    'val_metrics'  : metrics,
                    'exp_id'       : exp_id,
                    'exp_name'     : cfg['name'],
                    'hparams'      : {
                        'D_MODEL'       : D_MODEL,  'N_HEAD'  : N_HEAD,
                        'NUM_ENC'       : NUM_ENC,  'NUM_DEC' : NUM_DEC,
                        'D_FF'          : D_FF,     'DROPOUT' : DROPOUT,
                        'VOCAB_SIZE'    : VOCAB_SIZE, 'D_IN'  : D_IN,
                        'use_cnn'       : cfg['use_cnn'],
                        'face_mode'     : cfg['face_mode'],
                        'emb_kind'      : cfg['emb_kind'],
                        'unfreeze_epoch': unfreeze_ep,
                    }
                }, best_path)
                print(f'  NOUVEAU BEST BLEU-4={best_bleu:.2f} → {os.path.basename(best_path)}')
            else:
                epochs_no_improve += BLEU_EVERY
                print(f'  Patience {epochs_no_improve}/{PATIENCE} (best={best_bleu:.2f})')
        else:
            epochs_no_improve += 1

        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'epoch'     : epoch,
                'train_loss': round(tr_loss, 6),
                'train_acc' : round(tr_acc,  6),
                'val_loss'  : round(float(val_loss), 6),
                'val_acc'   : round(float(val_acc),  6),
                'val_bleu4' : val_bleu4,
            }, ensure_ascii=False) + '\n')

        if epoch % 5 == 0:
            plot_curves(log_path, curves_path, cfg['name'])

        if epochs_no_improve >= PATIENCE:
            print(f'\nEarly stopping epoch {epoch}. Best BLEU-4={best_bleu:.2f}')
            break

    plot_curves(log_path, curves_path, cfg['name'])

    # ── Evaluation finale VAL ─────────────────────────────────
    print('\n[4/4] Evaluation finale...')
    if best_path and os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        print(f'  Checkpoint recharge : {os.path.basename(best_path)}')

    f_val = FilteredLoader(val_loader, apply_face_slice)
    val_loss_final, val_acc_final = evaluate(model, f_val, criterion, PAD_ID)
    hyps_val, refs_val, ex_val   = run_inference(
        model, f_val, decode_ids, BOS_ID, EOS_ID, PAD_ID,
        show_examples=5, beam_size=1)
    val_final_metrics = compute_all_metrics(hyps_val, refs_val)
    val_final_metrics.update({
        'val_loss'       : round(float(val_loss_final), 6),
        'val_acc'        : round(float(val_acc_final),  6),
        'exp_id'         : exp_id,
        'exp_name'       : cfg['name'],
        'group'          : cfg['group'],
        'best_checkpoint': best_path,
        'examples'       : ex_val,
    })
    with open(val_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(val_final_metrics, f, ensure_ascii=False, indent=2)
    print(f'\n  VAL  BLEU-4={val_final_metrics["bleu4"]:.2f} | '
          f'ROUGE-L={val_final_metrics["rouge_l"]:.2f} | '
          f'CER={val_final_metrics["cer"]:.1f}%')

    # ── Evaluation finale TEST ────────────────────────────────
    test_final_metrics = None
    if test_loader is not None:
        f_test = FilteredLoader(test_loader, apply_face_slice)
        test_loss_final, test_acc_final = evaluate(model, f_test, criterion, PAD_ID)
        hyps_test, refs_test, ex_test   = run_inference(
            model, f_test, decode_ids, BOS_ID, EOS_ID, PAD_ID,
            show_examples=5, beam_size=1)
        test_final_metrics = compute_all_metrics(hyps_test, refs_test)
        test_final_metrics.update({
            'test_loss': round(float(test_loss_final), 6),
            'test_acc' : round(float(test_acc_final),  6),
            'exp_id'   : exp_id,
            'exp_name' : cfg['name'],
            'group'    : cfg['group'],
            'examples' : ex_test,
        })
        with open(test_metrics_path, 'w', encoding='utf-8') as f:
            json.dump(test_final_metrics, f, ensure_ascii=False, indent=2)
        print(f'  TEST BLEU-4={test_final_metrics["bleu4"]:.2f} | '
              f'ROUGE-L={test_final_metrics["rouge_l"]:.2f} | '
              f'CER={test_final_metrics["cer"]:.1f}%')
    else:
        print('  Test set non disponible.')

    # ── Mise a jour meilleur modele global ────────────────────
    global_best_file = os.path.join(BEST_DIR, 'global_best.json')
    current_bleu     = val_final_metrics['bleu4']
    save_as_global   = False

    if os.path.exists(global_best_file):
        with open(global_best_file, 'r') as f:
            gb = json.load(f)
        if current_bleu > gb.get('val_bleu4', 0):
            save_as_global = True
    else:
        save_as_global = True

    if save_as_global and best_path and os.path.exists(best_path):
        dest = os.path.join(BEST_DIR, 'best_model.pt')
        shutil.copy2(best_path, dest)
        with open(global_best_file, 'w') as f:
            json.dump({
                'exp_id'     : exp_id,
                'exp_name'   : cfg['name'],
                'val_bleu4'  : current_bleu,
                'source_ckpt': best_path,
                'saved_to'   : dest,
            }, f, ensure_ascii=False, indent=2)
        print(f'\n  NOUVEAU MEILLEUR MODELE GLOBAL ! BLEU-4={current_bleu:.2f}')
        print(f'  Sauvegarde application : {dest}')

    print(f'\n  Experiment {exp_id} termine. Resultats : {exp_dir}')
    return val_final_metrics, test_final_metrics


# ──────────────────────────────────────────────────────────────
# POINT D'ENTREE
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ablation experiments Sign2Text')
    parser.add_argument('--exp',   type=str, help='ID experiment (ex: g1_e1)')
    parser.add_argument('--group', type=int, help='Groupe a lancer (1-4)')
    parser.add_argument('--all',   action='store_true', help='Lancer tous les experiments')
    parser.add_argument('--reset', action='store_true', help='Effacer logs avant de lancer')
    parser.add_argument('--list',  action='store_true', help='Lister les experiments')
    args = parser.parse_args()

    if args.list:
        print('\nExperiments disponibles :')
        for eid, cfg in EXPERIMENTS.items():
            full = '[full_lm]' if cfg.get('use_full_lm') else ''
            uf   = f'[unfreeze={cfg.get("unfreeze_epoch",30)}]' if cfg['emb_kind'] != 'scratch' else ''
            print(f'  {eid:8s} | G{cfg["group"]} | {cfg["name"]} {full}{uf}')
        sys.exit(0)

    to_run = []
    if args.exp:
        if args.exp not in EXPERIMENTS:
            print(f'Experiment inconnu : {args.exp}')
            print(f'Disponibles : {list(EXPERIMENTS.keys())}')
            sys.exit(1)
        to_run = [args.exp]
    elif args.group:
        to_run = [eid for eid, cfg in EXPERIMENTS.items() if cfg['group'] == args.group]
        if not to_run:
            print(f'Groupe {args.group} inconnu.')
            sys.exit(1)
    elif args.all:
        to_run = list(EXPERIMENTS.keys())
    else:
        parser.print_help()
        print('\nExemple : python steps/06_train_experiments.py --exp g4_e3a')
        sys.exit(0)

    print(f'\nExperiments a lancer : {to_run}')

    all_val_results  = []
    all_test_results = []

    for exp_id in to_run:
        cfg = EXPERIMENTS[exp_id]
        val_m, test_m = run_experiment(exp_id, cfg, args)
        all_val_results.append(val_m)
        if test_m:
            all_test_results.append(test_m)

    if len(to_run) > 1:
        print('\n' + '=' * 65)
        print('RESUME')
        print('=' * 65)
        print(f'{"Exp":8s} {"Nom":38s} {"BLEU4":>6} {"ROUGE":>6} {"CER":>6}')
        print('-' * 65)
        for r in sorted(all_val_results, key=lambda x: x['bleu4'], reverse=True):
            print(f'{r["exp_id"]:8s} {r["exp_name"][:38]:38s} '
                  f'{r["bleu4"]:6.2f} {r["rouge_l"]:6.2f} {r["cer"]:6.1f}')

        summary_path = os.path.join(RESULTS_DIR, 'summary_val.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_val_results, f, ensure_ascii=False, indent=2)
        print(f'\nResume val : {summary_path}')

        if all_test_results:
            summary_test_path = os.path.join(RESULTS_DIR, 'summary_test.json')
            with open(summary_test_path, 'w', encoding='utf-8') as f:
                json.dump(all_test_results, f, ensure_ascii=False, indent=2)
            print(f'Resume test : {summary_test_path}')