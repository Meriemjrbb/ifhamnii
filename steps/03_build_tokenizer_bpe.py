"""
03_build_tokenizer_bpe.py
Emplacement : C:\\Users\\MSI\\PFA_Sign2Text\\steps\\03_build_tokenizer_bpe.py
Usage       : python steps/03_build_tokenizer_bpe.py

Installation préalable :
    pip install sentencepiece
"""

import os, re, glob, json
from pathlib import Path
import sentencepiece as spm

BASE = str(Path(__file__).resolve().parent.parent)
TEXTS        = os.path.join(BASE, 'text_cleaned')
TOK_DIR      = os.path.join(BASE, 'tokenizer')
os.makedirs(os.path.join(BASE, 'tokenizer_4k'), exist_ok=True)
RESULTS      = os.path.join(BASE, 'results')
CORPUS_FILE  = os.path.join(TOK_DIR, 'corpus.txt')
MODEL_PREFIX = os.path.join(BASE, 'tokenizer_4k', 'msa_bpe4k')

VOCAB_SIZE   = 4000

os.makedirs(TOK_DIR, exist_ok=True)

_HARAKAT = re.compile(r'[\u064B-\u0652\u0670\u0655\u0653\u0654]')
_TATWEEL = re.compile(r'\u0640')

def clean(text):
    text = _TATWEEL.sub('', text)
    text = _HARAKAT.sub('', text)
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

if __name__ == '__main__':
    # Construire le corpus
    txt_files = sorted(glob.glob(os.path.join(TEXTS, '*.txt')))
    print(f'{len(txt_files)} fichiers texte trouvés')

    lines = []
    for p in txt_files:
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            c = clean(f.read())
        if c:
            lines.append(c)

    with open(CORPUS_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'Corpus écrit : {len(lines)} lignes → {CORPUS_FILE}')

    # Entraîner SentencePiece BPE
    spm.SentencePieceTrainer.Train(
        input          = CORPUS_FILE,
        model_prefix   = MODEL_PREFIX,
        vocab_size     = VOCAB_SIZE,
        model_type     = 'bpe',
        pad_id         = 3,
        unk_id         = 0,
        bos_id         = 1,
        eos_id         = 2,
        character_coverage = 1.0,
        input_sentence_size = 5000000
    )
    print(f'Tokenizer sauvegardé : {MODEL_PREFIX}.model')

    # Vérification
    sp = spm.SentencePieceProcessor()
    sp.Load(MODEL_PREFIX + '.model')
    test = 'السلام عليكم كيف حالك'
    ids  = sp.Encode(test, out_type=int)
    dec  = sp.Decode(ids)
    print(f'\nTest : "{test}"')
    print(f'  IDs    : {ids}')
    print(f'  Decode : "{dec}"')
    print(f'  Vocab  : {sp.GetPieceSize()}')

