"""
05_build_transformer_encoder_decoder.py
Emplacement : PFA_Sign2Text/steps/05_build_transformer_encoder_decoder.py

Construit le modele Transformer anti-overfitting (VERSION CORRIGEE).
Peut etre importe par 06_train_transformer.py via :
    from steps.model import build_model

CORRECTIONS vs version initiale :
  1. D_MODEL=128, DROPOUT=0.3, LABEL_SMOOTHING=0.2, WEIGHT_DECAY=1e-2
  2. CNN temporel avec PermuteAndLayerNorm (fix du bug LayerNorm sur Conv1d)
  3. Chaque classe definie UNE SEULE FOIS

Usage autonome (smoke test) :
    cd PFA_Sign2Text
    python steps/05_build_transformer_encoder_decoder.py

Installation :
    pip install torch
"""

import os, sys, math
from pathlib import Path
import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────
# CHEMINS
# ──────────────────────────────────────────────────────────────
BASE = str(Path(__file__).resolve().parent.parent)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ──────────────────────────────────────────────────────────────
# HYPERPARAMETRES
# ──────────────────────────────────────────────────────────────
# Ces valeurs corrigent l'overfitting observe :
#   Train loss -> 2, Val loss -> 7, BLEU monte puis descend
#
# CHANGEMENTS vs version initiale :
#   D_MODEL  : 256  -> 128  (modele 4x plus petit)
#   DROPOUT  : 0.1  -> 0.3  (regularisation 3x plus forte)
#   LABEL_SM : 0.0  -> 0.2  (empeche sur-confiance)
#   WEIGHT_D : 1e-4 -> 1e-2 (penalise les gros poids)
#   LR       : 2e-3 -> 3e-4 (evite la divergence)

D_MODEL         = 128    # Dimension interne (reduit de 256)
N_HEAD          = 4      # Tetes d'attention (128/4 = 32 dims/tete)
NUM_ENC         = 2      # Couches encodeur (reduit de 3)
NUM_DEC         = 2      # Couches decodeur (reduit de 3)
D_FF            = 512    # Feedforward (= 4 * D_MODEL)
DROPOUT         = 0.3    # Dropout fort contre overfitting
LR              = 3e-4   # Learning rate standard Transformer
WEIGHT_DECAY    = 1e-2   # L2 regularisation forte
LABEL_SMOOTHING = 0.2    # Empeche le modele d'etre sur-confiant

# Ces valeurs sont normalement importees depuis 04_build_dataloader_v2.py
# Elles servent de secours si ce script est execute seul
VOCAB_SIZE_DEFAULT = 4000
PAD_ID_DEFAULT     = 3
BOS_ID_DEFAULT     = 1
EOS_ID_DEFAULT     = 2


# ──────────────────────────────────────────────────────────────
# COMPOSANTS DU MODELE
# ──────────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    Encodage positionnel sinusoidal.
    PE[t, 2i]   = sin(t / 10000^(2i/d))
    PE[t, 2i+1] = cos(t / 10000^(2i/d))
    Ajoute l'information de position a chaque vecteur d'entree.
    """
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # register_buffer : sauvegarde dans le modele, pas mis a jour par optimizer
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        # x : (B, T, D) -> ajoute pe[:, :T, :]
        return x + self.pe[:, :x.size(1), :]


class PermuteAndLayerNorm(nn.Module):
    """
    Applique LayerNorm sur un tensor (B, C, T) en format CNN.

    POURQUOI cette classe existe :
      Conv1d travaille en (B, C, T)
      LayerNorm standard attend (B, T, C)
      => PermuteAndLayerNorm gere les transpositions automatiquement
         pour pouvoir l'utiliser dans nn.Sequential sans erreur de forme.

    Flux :
      (B, C, T) -> transpose -> (B, T, C) -> LayerNorm -> (B, T, C) -> transpose -> (B, C, T)
    """
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)        # LayerNorm sur dim C
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        return x


class VideoEncoder(nn.Module):
    """
    Architecture hybride CNN + Transformer pour encoder les sequences de landmarks.

    Flux :
      (B, T, D_in)
        -> Projection lineaire   : (B, T, D_model)
        -> CNN temporel 2 couches : capture les patterns locaux (3-10 frames)
        -> Connexion residuelle   : h = h + CNN(h)
        -> Positional Encoding
        -> TransformerEncoder     : capture les dependances globales
        -> LayerNorm
      -> (B, T, D_model) memoire

    AVANTAGE vs Transformer seul :
      Le CNN gere les patterns LOCAUX (un geste = 3-10 frames consecutives)
      Le Transformer gere les patterns GLOBAUX (relations entre gestes)
      -> Division du travail = meilleure generalisation avec peu de donnees
    """
    def __init__(self, d_in, d_model, nhead, num_layers, d_ff, dropout):
        super().__init__()

        # Projection des landmarks vers la dimension du Transformer
        self.proj = nn.Linear(d_in, d_model)

        # CNN Temporel : travaille en format (B, C, T)
        # kernel_size=3, padding=1 -> conserve la longueur T
        # PermuteAndLayerNorm : gere les transpositions pour LayerNorm
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            PermuteAndLayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
        )

        self.pe      = SinusoidalPositionalEncoding(d_model)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.ln      = nn.LayerNorm(d_model)

    def forward(self, X, X_mask_bool):
        """
        X          : (B, T, D_in)  landmarks video
        X_mask_bool: (B, T) bool   True=frame valide, False=padding
        Retourne   : (memory (B,T,D_model), src_key_padding_mask (B,T))
        """
        # Convention PyTorch : src_key_padding_mask True = IGNORE
        # Notre convention   : X_mask_bool True = VALIDE  -> on inverse
        src_key_padding_mask = ~X_mask_bool

        # 1. Projection : (B, T, D_in) -> (B, T, D_model)
        h = self.proj(X)

        # 2. CNN temporel avec connexion residuelle
        h_cnn = self.temporal_cnn(h.transpose(1, 2))   # (B, D, T)
        h_cnn = h_cnn.transpose(1, 2)                   # (B, T, D)
        h     = h + h_cnn                               # connexion residuelle

        # 3. Positional encoding + Transformer
        h   = self.pe(h)
        mem = self.encoder(h, src_key_padding_mask=src_key_padding_mask)

        return self.ln(mem), src_key_padding_mask


def causal_mask(sz, device):
    """
    Masque triangulaire superieur (sz, sz).
    True = position a masquer (futur) -> le decodeur ne peut pas voir les tokens futurs.
    """
    return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)


class TextDecoder(nn.Module):
    """
    Decodeur Transformer : genere le texte token par token
    en s'appuyant sur la memoire video.

    Architecture :
      Embedding + Positional Encoding
      N x (Masked Self-Attn + Cross-Attn video + FeedForward)
      LayerNorm + Projection vocabulaire
    """
    def __init__(self, vocab_size, d_model, nhead, num_layers, d_ff, dropout, pad_id):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pe      = SinusoidalPositionalEncoding(d_model)
        dec_layer    = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.ln      = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, y_inp, mem, mem_key_padding_mask, y_pad_mask):
        """
        y_inp              : (B, L) IDs avec BOS au debut
        mem                : (B, T, D_model) memoire encodeur
        mem_key_padding_mask : (B, T) bool True=padding video
        y_pad_mask         : (B, L) bool True=padding texte
        Retourne : logits (B, L, vocab_size)
        """
        B, L     = y_inp.shape
        y        = self.pe(self.embed(y_inp))   # (B, L, D)
        tgt_mask = causal_mask(L, y.device)      # (L, L)
        h = self.decoder(
            tgt=y, memory=mem,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=y_pad_mask,
            memory_key_padding_mask=mem_key_padding_mask
        )
        return self.lm_head(self.ln(h))          # (B, L, vocab_size)


class Sign2TextTransformer(nn.Module):
    """
    Modele complet : Video -> Encodeur CNN+Transformer -> Decodeur -> Texte
    """
    def __init__(self, d_in, vocab_size, d_model, nhead,
                 num_enc, num_dec, d_ff, dropout, pad_id):
        super().__init__()
        self.encoder = VideoEncoder(d_in, d_model, nhead, num_enc, d_ff, dropout)
        self.decoder = TextDecoder(vocab_size, d_model, nhead, num_dec, d_ff, dropout, pad_id)
        self.pad_id  = pad_id

    def forward(self, batch):
        """
        Prend un batch du DataLoader.
        Retourne (logits (B,L,V), y_tgt (B,L)).
        """
        X      = batch['X'].to(device)
        X_mask = batch['X_mask'].to(device)
        y_inp  = batch['y_inp'].to(device)
        y_tgt  = batch['y_tgt'].to(device)
        y_mask = batch['y_mask'].to(device)
        mem, mem_kpm = self.encoder(X, X_mask)
        logits = self.decoder(y_inp, mem, mem_kpm, y_mask)
        return logits, y_tgt


class SequenceLoss(nn.Module):
    """
    CrossEntropy avec :
      ignore_index = pad_id    : ne pas calculer la loss sur le padding
      label_smoothing = 0.2   : evite la sur-confiance (anti-overfitting)
    """
    def __init__(self, pad_id, label_smoothing=0.2):
        super().__init__()
        self.crit = nn.CrossEntropyLoss(
            ignore_index=pad_id,
            label_smoothing=label_smoothing
        )

    def forward(self, logits, y_tgt):
        B, L, V = logits.shape
        return self.crit(logits.view(B * L, V), y_tgt.view(B * L))


def token_accuracy(logits, y_tgt, pad_id):
    """Pourcentage de tokens corrects (hors padding). Indicateur rapide."""
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        mask = (y_tgt != pad_id)
        return (pred[mask] == y_tgt[mask]).sum().item() / max(1, mask.sum().item())


# ──────────────────────────────────────────────────────────────
# FONCTION DE CONSTRUCTION (importable)
# ──────────────────────────────────────────────────────────────

def build_model(d_in, vocab_size, pad_id,
                d_model=D_MODEL, nhead=N_HEAD,
                num_enc=NUM_ENC, num_dec=NUM_DEC,
                d_ff=D_FF, dropout=DROPOUT,
                label_smoothing=LABEL_SMOOTHING,
                lr=LR, weight_decay=WEIGHT_DECAY):
    """
    Construit et retourne (model, criterion, optimizer).
    Usage depuis 06_train_transformer.py :
        from steps.model import build_model
        model, criterion, optimizer = build_model(d_in=372, vocab_size=VOCAB_SIZE, pad_id=PAD_ID)
    """
    model = Sign2TextTransformer(
        d_in=d_in, vocab_size=vocab_size,
        d_model=d_model, nhead=nhead,
        num_enc=num_enc, num_dec=num_dec,
        d_ff=d_ff, dropout=dropout, pad_id=pad_id
    ).to(device)

    criterion = SequenceLoss(pad_id, label_smoothing=label_smoothing).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    return model, criterion, optimizer


# ──────────────────────────────────────────────────────────────
# POINT D'ENTREE : smoke test autonome
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 60)
    print('05_build_transformer_encoder_decoder.py — Smoke Test')
    print('=' * 60)
    print(f'  device = {device}')
    if device.type == 'cpu':
        print('  ATTENTION : GPU recommande pour l entrainement.')

    # Essayer de charger les variables du dataloader
    try:
        sys.path.insert(0, BASE)
        from steps.dataloader import (
            build_loaders, VOCAB_SIZE, PAD_ID, BOS_ID, EOS_ID, decode_ids
        )
        print(f'  Dataloader importe : VOCAB_SIZE={VOCAB_SIZE}')
        train_loader, val_loader, _ = build_loaders(batch_size=4, num_workers=0)
        sample_batch = next(iter(train_loader))
        D_IN = sample_batch['X'].shape[-1]
        vocab_size = VOCAB_SIZE
        pad_id     = PAD_ID
    except Exception as e:
        print(f'  Dataloader non disponible ({e})')
        print(f'  Utilisation de valeurs fictives pour le smoke test.')
        # Batch synthetique
        B, T, D_IN = 2, 50, 372
        L          = 8
        vocab_size = VOCAB_SIZE_DEFAULT
        pad_id     = PAD_ID_DEFAULT
        sample_batch = {
            'X'     : torch.randn(B, T, D_IN),
            'X_mask': torch.ones(B, T, dtype=torch.bool),
            'y_inp' : torch.randint(4, vocab_size, (B, L)),
            'y_tgt' : torch.randint(4, vocab_size, (B, L)),
            'y_mask': torch.zeros(B, L, dtype=torch.bool),
        }

    print(f'\n  Hyperparametres :')
    print(f'    D_MODEL={D_MODEL} | N_HEAD={N_HEAD} | dims/tete={D_MODEL//N_HEAD}')
    print(f'    NUM_ENC={NUM_ENC} | NUM_DEC={NUM_DEC} | D_FF={D_FF}')
    print(f'    DROPOUT={DROPOUT} | LABEL_SMOOTHING={LABEL_SMOOTHING}')
    print(f'    LR={LR} | WEIGHT_DECAY={WEIGHT_DECAY}')
    print(f'    D_IN={D_IN} | VOCAB_SIZE={vocab_size}')

    model, criterion, optimizer = build_model(
        d_in=D_IN, vocab_size=vocab_size, pad_id=pad_id
    )

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n  Parametres totaux      : {total_p/1e6:.2f}M')
    print(f'  Parametres entrainables : {trainable_p/1e6:.2f}M')

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, y_tgt = model(sample_batch)
        loss = criterion(logits, y_tgt).item()
        acc  = token_accuracy(logits, y_tgt, pad_id)

    print(f'\n  Smoke test OK !')
    print(f'    logits.shape = {tuple(logits.shape)}  <- (B, L, vocab_size)')
    print(f'    loss = {loss:.4f}')
    print(f'    acc  = {acc:.4f}')

    # La loss initiale devrait etre proche de log(VOCAB_SIZE)
    expected = math.log(vocab_size)
    print(f'\n  Loss aleatoire theorique = log({vocab_size}) = {expected:.3f}')
    if loss < expected * 2:
        print('  Loss initiale OK (dans la plage attendue)')
    else:
        print('  ATTENTION : loss trop elevee, verifier le modele')

    print('\nPour lancer l entrainement :')
    print('  python steps/06_train_transformer.py')
