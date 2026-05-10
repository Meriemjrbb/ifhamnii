"""
03b_build_tokenizer_word.py
Usage : python steps/03b_build_tokenizer_word.py

Construit un tokenizer mot-a-mot (word-level) sur les transcriptions arabes.

DIFFERENCE vs BPE (03_build_tokenizer_bpe.py) :
  BPE        : decoupe les mots en sous-mots  -> vocab fixe (ex: 2000)
  Word-Level : garde chaque MOT entier        -> vocab = nb mots uniques

AVANTAGES Word-Level pour ce projet :
  + Sequences plus courtes (1 mot = 1 token, pas 2-3 sous-mots)
  + Chaque token = un vrai mot arabe, plus interpretable
  + Meilleur pour les petits datasets (<10 000 phrases)

INCONVENIENTS :
  - Mots rares -> <UNK>
  - Ne gere pas les variantes morphologiques
"""

import os, re, glob, json
from pathlib import Path
from collections import Counter

BASE         = str(Path(__file__).resolve().parent.parent)
TEXTS        = os.path.join(BASE, 'text_cleaned')
TOK_WORD_DIR = os.path.join(BASE, 'tokenizer_word')
RESULTS      = os.path.join(BASE, 'results')

os.makedirs(TOK_WORD_DIR, exist_ok=True)
os.makedirs(RESULTS,      exist_ok=True)

MIN_FREQ = 1  # Garder tous les mots (corpus petit)

# ── Tokens speciaux (meme IDs que BPE pour la compatibilite) ──
# UNK=0, BOS=1, EOS=2, PAD=3
SPECIAL_TOKENS = ['<UNK>', '<SOS>', '</s>', '<PAD>']

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


if __name__ == '__main__':
    # ── ETAPE 1 : Lire toutes les transcriptions ───────────────
    txt_files = sorted(glob.glob(os.path.join(TEXTS, '*.txt')))
    print(f'{len(txt_files)} fichiers texte trouves')

    sentences = []
    for path in txt_files:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        text = clean(text)
        if text:
            sentences.append(text)

    print(f'Phrases non vides : {len(sentences)}')
    print(f'\nExemples :')
    for s in sentences[:3]:
        print(f'  {s}')

    # ── ETAPE 2 : Compter les mots ─────────────────────────────
    counter = Counter()
    for s in sentences:
        counter.update(s.split())

    print(f'\nMots uniques dans le corpus : {len(counter)}')
    print(f'Top 10 mots les plus frequents :')
    for mot, freq in counter.most_common(10):
        print(f'  {mot:20s} : {freq}')

    # ── ETAPE 3 : Filtrer et construire le vocabulaire ─────────
    words_kept    = sorted([w for w, c in counter.items() if c >= MIN_FREQ])
    words_removed = [w for w, c in counter.items() if c < MIN_FREQ]

    print(f'\nMIN_FREQ = {MIN_FREQ}')
    print(f'Mots gardes  : {len(words_kept)}')
    print(f'Mots enleves : {len(words_removed)}')

    vocab  = SPECIAL_TOKENS + words_kept
    word2id = {w: i for i, w in enumerate(vocab)}
    id2word = {i: w for w, i in word2id.items()}

    UNK_ID = word2id['<UNK>']  # 0
    BOS_ID = word2id['<SOS>']  # 1
    EOS_ID = word2id['</s>']   # 2
    PAD_ID = word2id['<PAD>']  # 3

    print(f'\nVocabulaire final : {len(vocab)} tokens')
    print(f'UNK={UNK_ID} | BOS={BOS_ID} | EOS={EOS_ID} | PAD={PAD_ID}')

    # ── ETAPE 4 : Statistiques sur les longueurs ───────────────
    def encode_word(text):
        c = clean(text)
        return [word2id.get(w, UNK_ID) for w in c.split()], c

    def decode_word(ids):
        skip = {UNK_ID, BOS_ID, EOS_ID, PAD_ID}
        return ' '.join(id2word[i] for i in ids if i in id2word and i not in skip)

    lengths   = []
    unk_total = 0
    tok_total = 0
    for s in sentences:
        ids, _ = encode_word(s)
        lengths.append(len(ids))
        unk_total += ids.count(UNK_ID)
        tok_total += len(ids)

    avg_len  = sum(lengths) / len(lengths)
    max_len  = max(lengths)
    unk_rate = unk_total / tok_total * 100 if tok_total else 0

    print(f'\nStatistiques Word-Level :')
    print(f'  Longueur moyenne : {avg_len:.1f} tokens/phrase')
    print(f'  Longueur max     : {max_len} tokens')
    print(f'  Taux <UNK>       : {unk_rate:.2f}%')
    print(f'\n  -> Utiliser MAX_TGT_LEN = {max_len + 5} dans le dataloader')

    # Comparaison avec BPE si disponible
    bpe_model = os.path.join(BASE, 'tokenizer', 'msa_bpe2k.model')
    if os.path.exists(bpe_model):
        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.Load(bpe_model)
            bpe_lengths = [len(sp.Encode(s, out_type=int)) for s in sentences]
            print(f'\nComparaison BPE vs Word-Level :')
            print(f'  BPE  : vocab={sp.GetPieceSize()} | max_len={max(bpe_lengths)} tokens')
            print(f'  Word : vocab={len(vocab)}        | max_len={max_len} tokens')
            if max_len < max(bpe_lengths):
                print('  -> Word-Level a des sequences PLUS COURTES : avantage entrainement')
            else:
                print('  -> BPE a des sequences plus courtes')
        except Exception:
            pass

    # Test encodage/decodage
    test_phrase = sentences[0] if sentences else 'السلام عليكم'
    ids, cleaned = encode_word(test_phrase)
    decoded      = decode_word(ids)
    print(f'\nTest encodage/decodage :')
    print(f'  Original : {test_phrase}')
    print(f'  Nettoye  : {cleaned}')
    print(f'  IDs      : {ids[:8]}... (total={len(ids)})')
    print(f'  Decode   : {decoded}')
    print(f'  Reconstruction OK : {cleaned == decoded}')

    # ── ETAPE 5 : Sauvegarde ───────────────────────────────────
    # word_vocab.txt
    vocab_path = os.path.join(TOK_WORD_DIR, 'word_vocab.txt')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(vocab))

    # word2id.json
    w2i_path = os.path.join(TOK_WORD_DIR, 'word2id.json')
    with open(w2i_path, 'w', encoding='utf-8') as f:
        json.dump(word2id, f, ensure_ascii=False, indent=2)

    # id2word.json (cles en string pour JSON)
    i2w_path = os.path.join(TOK_WORD_DIR, 'id2word.json')
    with open(i2w_path, 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in id2word.items()}, f, ensure_ascii=False, indent=2)

    # word_tokenizer_info.json
    info = {
        'type'         : 'word',
        'vocab_size'   : len(vocab),
        'min_frequency': MIN_FREQ,
        'special_tokens': {
            'unk': '<UNK>', 'unk_id': UNK_ID,
            'bos': '<SOS>', 'bos_id': BOS_ID,
            'eos': '</s>',  'eos_id': EOS_ID,
            'pad': '<PAD>', 'pad_id': PAD_ID,
        },
        'stats': {
            'n_sentences'    : len(sentences),
            'avg_len_tokens' : round(avg_len, 2),
            'max_len_tokens' : max_len,
            'unk_rate_pct'   : round(unk_rate, 4),
        }
    }
    info_path = os.path.join(TOK_WORD_DIR, 'word_tokenizer_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f'\nTokenizer Word-Level sauvegarde :')
    print(f'  {vocab_path}')
    print(f'  {w2i_path}')
    print(f'  {i2w_path}')
    print(f'  {info_path}')
    print(f'\nPour utiliser ce tokenizer dans le dataloader :')
    print(f"  TOKENIZER_KIND = 'word'")
