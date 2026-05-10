"""
prepare_arabert_embeddings.py
Emplacement : PFA_Sign2Text/steps/prepare_arabert_embeddings.py

Extrait les embeddings AraBERT alignes sur le vocabulaire du word tokenizer.

Probleme : AraBERT a son propre vocabulaire (WordPiece, ~64k mots).
Notre word tokenizer a 8622 mots differents.
On ne peut pas utiliser directement la matrice AraBERT car les IDs ne correspondent pas.

Solution :
  Pour chaque mot du word tokenizer :
    1. Tokeniser le mot avec le tokenizer AraBERT
    2. Extraire les vecteurs d'embedding AraBERT pour ses sous-tokens
    3. Faire la moyenne de ces vecteurs
    4. Stocker dans notre matrice (8622 x 768)

Resultat : embeddings/arabert_aligned.pt
  → matrice FloatTensor (8622, 768) utilisable directement dans g4_e3a/b/c

Usage :
    cd PFA_Sign2Text
    pip install transformers
    python steps/prepare_arabert_embeddings.py

Temps : ~2-3 minutes
"""

import os, json, torch
from pathlib import Path

BASE         = str(Path(__file__).resolve().parent.parent)
TOK_WORD_DIR = os.path.join(BASE, 'tokenizer_word')
EMB_DIR      = os.path.join(BASE, 'embeddings')
os.makedirs(EMB_DIR, exist_ok=True)

OUT_PATH = os.path.join(EMB_DIR, 'arabert_aligned.pt')

print('=' * 60)
print('prepare_arabert_embeddings.py')
print('=' * 60)

# ── Charger le word tokenizer ─────────────────────────────────
w2i_path = os.path.join(TOK_WORD_DIR, 'word2id.json')
with open(w2i_path, 'r', encoding='utf-8') as f:
    word2id = json.load(f)

VOCAB_SIZE = len(word2id)
print(f'  Word tokenizer : {VOCAB_SIZE} mots')

# ── Charger AraBERT ───────────────────────────────────────────
print(f'  Chargement AraBERT...')
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    print('  ERREUR : pip install transformers requis')
    exit(1)

MODEL_NAME  = 'aubmindlab/bert-base-arabertv2'
arabert_tok = AutoTokenizer.from_pretrained(MODEL_NAME)
arabert_mod = AutoModel.from_pretrained(MODEL_NAME)
arabert_mod.eval()

# Matrice d'embedding AraBERT : (64000, 768)
arabert_emb = arabert_mod.embeddings.word_embeddings.weight.data
EMB_DIM     = arabert_emb.shape[1]   # 768
print(f'  AraBERT vocab  : {arabert_emb.shape[0]} tokens, dim={EMB_DIM}')

# ── Construire la matrice alignee ─────────────────────────────
print(f'\n  Construction matrice alignee ({VOCAB_SIZE} x {EMB_DIM})...')
matrix = torch.zeros(VOCAB_SIZE, EMB_DIM)

n_found    = 0
n_partial  = 0
n_missing  = 0

special_tokens = {'<PAD>', '<UNK>', '<SOS>', '</s>'}

for word, idx in word2id.items():
    if idx >= VOCAB_SIZE:
        continue

    if word in special_tokens:
        # Garder zeros pour les tokens speciaux
        n_missing += 1
        continue

    # Tokeniser le mot avec AraBERT
    token_ids = arabert_tok.encode(word, add_special_tokens=False)

    if not token_ids:
        n_missing += 1
        continue

    # Extraire les vecteurs et faire la moyenne
    vecs = arabert_emb[token_ids]   # (n_subtokens, 768)
    matrix[idx] = vecs.mean(dim=0)

    if len(token_ids) == 1:
        n_found += 1
    else:
        n_partial += 1

print(f'  Mots trouves exactement   : {n_found}')
print(f'  Mots divises en sous-tokens: {n_partial}')
print(f'  Mots non trouves (zeros)  : {n_missing}')
print(f'  Couverture                : {100*(n_found+n_partial)/VOCAB_SIZE:.1f}%')

# ── Sauvegarder ───────────────────────────────────────────────
torch.save(matrix, OUT_PATH)
print(f'\n  Sauvegarde : {OUT_PATH}')
print(f'  Shape      : {tuple(matrix.shape)}')
print(f'  Taille     : {os.path.getsize(OUT_PATH)/1e6:.1f} MB')

# Nettoyage memoire
del arabert_mod, arabert_emb
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print('\nTermine. Utiliser dans 06_train_experiments.py : emb_kind=arabert')
