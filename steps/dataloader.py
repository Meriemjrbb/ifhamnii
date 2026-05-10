"""
04_build_dataloader_v2.py
Usage : python steps/04_build_dataloader_v2.py

Ce fichier est aussi importe par 08_train_improved.py via :
    from steps.dataloader import build_loaders

CONFIGURATION (modifier ces deux lignes) :
    USE_SEGMENTS   = True   -> charge les segments (02b) | False -> landmarks originaux (02)
    TOKENIZER_KIND = 'bpe'  -> tokenizer BPE (03)        | 'word' -> word-level (03b)

Installation :
    pip install torch sentencepiece numpy
"""

import os, glob, re, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# ── Chemins ───────────────────────────────────────────────────
BASE         = str(Path(__file__).resolve().parent.parent)
LAND_DIR     = os.path.join(BASE, 'landmarks')
SEGMENTS_DIR = os.path.join(BASE, 'segments')
TEXT_DIR     = os.path.join(BASE, 'text_cleaned')
TOK_BPE_DIR    = globals().get('TOK_BPE_DIR',    os.path.join(BASE, 'tokenizer'))
TOK_WORD_DIR = os.path.join(BASE, 'tokenizer_word')
RESULTS      = os.path.join(BASE, 'results')

# ── CONFIGURATION ─────────────────────────────────────────────
# D attendu depuis 02_extract_landmarks_local.py : (33+21+21+49)*3
D_EXPECTED = 372

USE_SEGMENTS   = globals().get('USE_SEGMENTS',   True)
TOKENIZER_KIND = globals().get('TOKENIZER_KIND', 'word')
TOK_BPE_DIR    = globals().get('TOK_BPE_DIR',    os.path.join(BASE, 'tokenizer'))

# ── Nettoyage texte arabe ──────────────────────────────────────
_HARAKAT = re.compile(r'[\u064B-\u0652\u0670\u0655\u0653\u0654]')
_TATWEEL = re.compile(r'\u0640')

def clean(text):
    text = _TATWEEL.sub('', text)
    text = _HARAKAT.sub('', text)
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ── Chargement du Tokenizer ────────────────────────────────────
def _load_tokenizer(kind):
    if kind == 'bpe':
        import sentencepiece as spm
        import glob as _glob
        _candidates = _glob.glob(os.path.join(TOK_BPE_DIR, 'msa_bpe*.model'))
        model_path = _candidates[0] if _candidates else os.path.join(TOK_BPE_DIR, 'msa_bpe2k.model')
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f'BPE non trouve : {model_path}\n'
                f'Executer steps/03_build_tokenizer_bpe.py dabord.'
            )
        sp = spm.SentencePieceProcessor()
        sp.Load(model_path)
        pad_id = 3
        unk_id = sp.unk_id()
        bos_id = sp.bos_id()
        eos_id = sp.eos_id()
        vocab_size = sp.GetPieceSize()
        def encode_text(text):
            c = clean(text)
            return sp.Encode(c, out_type=int), c
        def decode_ids(ids):
            ids = [i for i in ids if i not in (pad_id, bos_id, eos_id)]
            return sp.Decode(ids) if ids else ''
        print(f'Tokenizer BPE charge : VOCAB_SIZE={vocab_size}')

    elif kind == 'word':
        w2i_path = os.path.join(TOK_WORD_DIR, 'word2id.json')
        i2w_path = os.path.join(TOK_WORD_DIR, 'id2word.json')
        for p in [w2i_path, i2w_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f'Word tokenizer non trouve : {p}\n'
                    f'Executer steps/03b_build_tokenizer_word.py dabord.'
                )
        with open(w2i_path, 'r', encoding='utf-8') as f:
            word2id = json.load(f)
        with open(i2w_path, 'r', encoding='utf-8') as f:
            id2word = {int(k): v for k, v in json.load(f).items()}
        unk_id = word2id['<UNK>']
        bos_id = word2id['<SOS>']
        eos_id = word2id['</s>']
        pad_id = word2id['<PAD>']
        vocab_size = len(word2id)
        def encode_text(text):
            c = clean(text)
            return [word2id.get(w, unk_id) for w in c.split()], c
        def decode_ids(ids):
            skip = {unk_id, bos_id, eos_id, pad_id}
            return ' '.join(id2word[i] for i in ids if i in id2word and i not in skip)
        print(f'Tokenizer Word charge : VOCAB_SIZE={vocab_size}')

    else:
        raise ValueError(f"TOKENIZER_KIND doit etre 'bpe' ou 'word', recu : '{kind}'")

    print(f'UNK={unk_id} | BOS={bos_id} | EOS={eos_id} | PAD={pad_id}')
    return encode_text, decode_ids, pad_id, bos_id, eos_id, unk_id, vocab_size


encode_text, decode_ids, PAD_ID, BOS_ID, EOS_ID, UNK_ID, VOCAB_SIZE = \
    _load_tokenizer(globals().get('TOKENIZER_KIND', 'word'))

# ── Augmentation ──────────────────────────────────────────────
def augment_landmarks(X, noise_std=0.02, mask_prob=0.5,
                      stretch_prob=0.3, mask_ratio_max=0.15):
    X_aug = X.copy()
    T     = X_aug.shape[0]
    X_aug += np.random.normal(0, noise_std, X_aug.shape).astype(np.float32)
    if np.random.rand() < mask_prob and T > 4:
        mask_len = max(1, min(int(T * np.random.uniform(0.05, mask_ratio_max)), T // 3))
        start    = np.random.randint(0, max(1, T - mask_len))
        X_aug[start:start + mask_len] = 0.0
    if np.random.rand() < stretch_prob and T > 4:
        factor  = np.random.uniform(0.85, 1.15)
        new_T   = max(2, int(T * factor))
        indices = np.clip(np.linspace(0, T - 1, new_T).astype(int), 0, T - 1)
        X_aug   = X_aug[indices]
    return X_aug

# ── Dataset ───────────────────────────────────────────────────
class Sign2TextDataset(Dataset):
    def __init__(self, use_segments=True, augment=False, indices=None):
        self.augment      = augment
        self.use_segments = use_segments
        self.samples      = []

        if use_segments:
            manifest_path = os.path.join(RESULTS, 'segments_manifest.json')
            if not os.path.exists(manifest_path):
                raise FileNotFoundError(
                    f'Manifeste non trouve : {manifest_path}\n'
                    f'Executer steps/02b_dual_overlap_segmentation.py dabord.'
                )
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            for entry in manifest:
                npz_path = entry['npz_path']
                if os.path.exists(npz_path) and entry.get('text', ''):
                    self.samples.append((entry['seg_id'], npz_path, entry['text']))
        else:
            for npz_path in sorted(glob.glob(os.path.join(LAND_DIR, '*.npz'))):
                vid      = os.path.splitext(os.path.basename(npz_path))[0]
                txt_path = os.path.join(TEXT_DIR, vid + '.txt')
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = clean(f.read())
                    if text:
                        self.samples.append((vid, npz_path, text))

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

        mode = 'SEGMENTS' if use_segments else 'ORIGINAUX'
        aug  = '+AUG' if augment else ''
        print(f'  Dataset [{mode}{aug}][{TOKENIZER_KIND}] : {len(self.samples)} exemples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        vid, npz_path, text = self.samples[i]
        data = np.load(npz_path)
        X    = data['X'].astype(np.float32)
        mask = data['mask'].astype(np.int8)
        ids, clean_txt = encode_text(text)
        if self.augment:
            X_aug = augment_landmarks(X)
            T_new, T_old = X_aug.shape[0], X.shape[0]
            if T_new != T_old:
                idx  = np.clip(np.linspace(0, T_old-1, T_new).astype(int), 0, T_old-1)
                mask = mask[idx]
            X = X_aug
        if len(ids) == 0:
            ids = [UNK_ID]
        return {'vid': vid, 'X': X, 'X_mask': mask, 'y_ids': ids, 'y_raw': clean_txt}

# ── Collate ───────────────────────────────────────────────────
def pad_1d(seqs, pad_id):
    B     = len(seqs)
    L_max = max(len(s) for s in seqs) if seqs else 1
    padded   = torch.full((B, L_max), pad_id, dtype=torch.long)
    pad_mask = torch.ones((B, L_max), dtype=torch.bool)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        pad_mask[i, :len(s)] = False
    return padded, pad_mask

def collate_fn(batch):
    B     = len(batch)
    D     = batch[0]['X'].shape[1]
    if D not in (372, 1629):
        import warnings
        warnings.warn(f'D={D} inattendu (attendu D_EXPECTED={D_EXPECTED}). Verifier les landmarks.')
    T_max = max(it['X'].shape[0] for it in batch)
    X      = torch.zeros((B, T_max, D), dtype=torch.float32)
    X_mask = torch.zeros((B, T_max), dtype=torch.bool)
    vids, y_raw_list = [], []
    for i, item in enumerate(batch):
        T = item['X'].shape[0]
        X[i, :T] = torch.from_numpy(item['X'])
        X_mask[i, :T] = torch.from_numpy(item['X_mask'].astype(np.int64)) > 0
        vids.append(item['vid'])
        y_raw_list.append(item['y_raw'])
    y_inp_list = [[BOS_ID] + it['y_ids'] for it in batch]
    y_tgt_list = [it['y_ids'] + [EOS_ID] for it in batch]
    y_inp, y_pad = pad_1d(y_inp_list, PAD_ID)
    y_tgt, _     = pad_1d(y_tgt_list, PAD_ID)
    return {'vid': vids, 'X': X, 'X_mask': X_mask,
            'y_inp': y_inp, 'y_tgt': y_tgt, 'y_mask': y_pad, 'y_raw': y_raw_list}

# ── Build loaders ─────────────────────────────────────────────
def build_loaders(batch_size=8, num_workers=0, use_segments=None):
    """
    Retourne (train_loader, val_loader, test_loader_or_None).
    num_workers=0 recommande sur Windows pour eviter les erreurs multiprocessing.
    """
    _use_seg   = use_segments if use_segments is not None else USE_SEGMENTS
    split_path = os.path.join(RESULTS, 'split.json')

    if os.path.exists(split_path):
        with open(split_path, 'r') as f:
            split = json.load(f)
        train_ids = set(split['train'])
        val_ids   = set(split['val'])
        test_ids  = set(split.get('test', []))

        full_train_ds = Sign2TextDataset(use_segments=_use_seg, augment=True)
        full_eval_ds  = Sign2TextDataset(use_segments=_use_seg, augment=False)

        def _ids_for(ds, id_set):
            return [i for i, (vid, _, _) in enumerate(ds.samples)
                    if vid.rsplit('_seg', 1)[0] in id_set]

        train_idx = _ids_for(full_train_ds, train_ids)
        val_idx   = _ids_for(full_eval_ds,  val_ids)
        test_idx  = _ids_for(full_eval_ds,  test_ids)

        train_ds = Subset(full_train_ds, train_idx)
        val_ds   = Subset(full_eval_ds,  val_idx)
        test_ds  = Subset(full_eval_ds,  test_idx) if test_idx else None

        print(f'Split charge depuis split.json')
        print(f'  train_ids={len(train_ids)} | val_ids={len(val_ids)} | test_ids={len(test_ids)}')

    else:
        print('split.json non trouve -> split 80/20 automatique')
        if _use_seg:
            manifest_path = os.path.join(RESULTS, 'segments_manifest.json')
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            unique_vids = sorted(set(e['src_vid'] for e in manifest))
            np.random.seed(42)
            np.random.shuffle(unique_vids)
            n_tr = int(0.8 * len(unique_vids))
            train_vids_set = set(unique_vids[:n_tr])
            val_vids_set   = set(unique_vids[n_tr:])
            full_train_ds  = Sign2TextDataset(use_segments=True, augment=True)
            full_eval_ds   = Sign2TextDataset(use_segments=True, augment=False)
            train_idx = [i for i, (vid, _, _) in enumerate(full_train_ds.samples)
                         if vid.rsplit('_seg', 1)[0] in train_vids_set]
            val_idx   = [i for i, (vid, _, _) in enumerate(full_eval_ds.samples)
                         if vid.rsplit('_seg', 1)[0] in val_vids_set]
            train_ds  = Subset(full_train_ds, train_idx)
            val_ds    = Subset(full_eval_ds,  val_idx)
        else:
            full_ds = Sign2TextDataset(use_segments=False, augment=False)
            n = len(full_ds)
            idx = np.arange(n)
            np.random.seed(42)
            np.random.shuffle(idx)
            n_train  = int(0.8 * n)
            train_ds = Subset(Sign2TextDataset(use_segments=False, augment=True),  idx[:n_train])
            val_ds   = Subset(Sign2TextDataset(use_segments=False, augment=False), idx[n_train:])
        test_ds = None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=collate_fn, pin_memory=True) \
                   if test_ds else None

    print(f'\nTrain : {len(train_ds)} exemples | {len(train_loader)} batches')
    print(f'Val   : {len(val_ds)} exemples   | {len(val_loader)} batches')
    if test_loader:
        print(f'Test  : {len(test_ds)} exemples  | {len(test_loader)} batches')
    else:
        print(f'Test  : non disponible')

    config = {
        'use_segments'  : _use_seg,
        'tokenizer_kind': TOKENIZER_KIND,
        'vocab_size'    : int(VOCAB_SIZE),
        'pad_id': int(PAD_ID), 'bos_id': int(BOS_ID), 'eos_id': int(EOS_ID),
        'batch_size': batch_size,
        'n_train': len(train_ds), 'n_val': len(val_ds),
        'n_test' : len(test_ds) if test_ds else 0
    }
    with open(os.path.join(RESULTS, 'dataloader_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print('Config sauvegardee.')

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    print(f'USE_SEGMENTS   = {USE_SEGMENTS}')
    print(f'TOKENIZER_KIND = {TOKENIZER_KIND}')
    print()

    # num_workers=0 obligatoire sur Windows
    train_loader, val_loader, test_loader = build_loaders(batch_size=8, num_workers=0)

    b = next(iter(train_loader))
    print(f'\nTest batch : X={tuple(b["X"].shape)} | y_inp={tuple(b["y_inp"].shape)}')
    print(f'Exemple texte  : "{b["y_raw"][0]}"')
    print(f'Exemple decode : "{decode_ids(b["y_inp"][0].tolist())}"')
