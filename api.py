"""
api.py — FastAPI server for Arabic Sign Language Translation
Best model: g4_e3b — AraBERT progressive unfreeze ep30
  D_IN=372 | D_MODEL=128 | VOCAB_SIZE=8622 | emb_dim=768 (proj 768→128)

PATCHED VERSION
---------------
This file fixes four train/inference inconsistencies that were silently
hurting prediction quality. Each fix is documented next to the relevant
code with a short comment starting with `# FIX —`.

  1. FIX — face landmark indices now match 02_extract_landmarks_local.py
           (NOSE_TIP = [1, 4, 94] instead of NOSE = [1, 2, 98, 327] + CHIN
           truncated by [:49]).
  2. FIX — per-frame spatial normalisation
           (centre on hip midpoint, scale by inter-shoulder distance)
           applied inside extract_landmarks, identical to
           normalize_landmarks_frame() in training.
  3. FIX — step-based downsampling (np.arange(0, n, step)) instead of
           np.interp resampling. Selects real frames, doesn't fabricate
           intermediate ones.
  4. FIX — per-frame validity mask is propagated all the way to the
           encoder. Frames where MediaPipe failed to detect pose AND
           both hands are now correctly masked out, as during training.

Bonus fix: a token-overlap-aware merge replaces the brittle
substring/concat heuristic when stitching per-segment predictions
into a full sentence.
"""

import os, re, json, math, tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import cv2
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
BASE       = r'C:\Users\MSI\PFA_Sign2Text'
CKPT_PATH  = os.path.join(BASE, 'models', 'best', 'best_model.pt')
W2ID_PATH  = os.path.join(BASE, 'tokenizer_word', 'word2id.json')
ID2W_PATH  = os.path.join(BASE, 'tokenizer_word', 'id2word.json')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparams — g4_e3b
D_MODEL    = 128
N_HEAD     = 4
NUM_ENC    = 2
NUM_DEC    = 2
D_FF       = 512
DROPOUT    = 0.0    # 0 at inference
D_IN       = 372
VOCAB_SIZE = 8622
EMB_DIM    = 768    # AraBERT embedding dim before projection 768→128

# Token IDs (word tokenizer)
PAD_ID = 3
BOS_ID = 1
EOS_ID = 2

# Landmarks extraction
TARGET_FPS  = 20

# Segmentation — same as 02b_dual_overlap_segmentation.py
SEGMENT_LEN = 300
STRIDE      = 150
MIN_FRAMES  = 30

# ──────────────────────────────────────────────────────────────
# FIX #1 — Face landmark indices
# Restored to be IDENTICAL to 02_extract_landmarks_local.py.
# Old api.py used NOSE=[1,2,98,327] + CHIN=[152,176,150,136] then
# truncated with [:49], which kept indices 2 and 98 (never seen in
# training) and dropped indices 4 and 94 (used in training).
# ──────────────────────────────────────────────────────────────
LIPS_OUTER    = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
LIPS_INNER    = [78, 13, 312, 308, 14, 87, 88, 95, 82, 310]
LEFT_EYE      = [33, 159, 145, 133]
RIGHT_EYE     = [362, 386, 374, 263]
LEFT_EYEBROW  = [70, 107, 55, 46]
RIGHT_EYEBROW = [300, 336, 285, 276]
NOSE_TIP      = [1, 4, 94]                         # ← matches training

FACE_IDX      = list(dict.fromkeys(LIPS_OUTER + LIPS_INNER +
                                   LEFT_EYE + RIGHT_EYE +
                                   LEFT_EYEBROW + RIGHT_EYEBROW +
                                   NOSE_TIP))      # 49 unique indices
MAX_FACE_IDX  = max(FACE_IDX)
assert len(FACE_IDX) == 49, f"FACE_IDX must have 49 entries, got {len(FACE_IDX)}"

# ──────────────────────────────────────────────────────────────
# TEXT CLEANING
# ──────────────────────────────────────────────────────────────
_HARAKAT = re.compile(r'[\u064B-\u0652\u0670\u0655\u0653\u0654]')
_TATWEEL = re.compile(r'\u0640')

def clean_arabic(text: str) -> str:
    text = _TATWEEL.sub('', text)
    text = _HARAKAT.sub('', text)
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# ──────────────────────────────────────────────────────────────
# ARCHITECTURE — IDENTICAL to 06_train_experiments.py
# ──────────────────────────────────────────────────────────────
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class PermuteAndLayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class VideoEncoder(nn.Module):
    def __init__(self, d_in, d_model, nhead, num_layers, d_ff, dropout):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.cnn  = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            PermuteAndLayerNorm(d_model), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
        )
        self.pe      = SinusoidalPositionalEncoding(d_model)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.ln      = nn.LayerNorm(d_model)

    def forward(self, X, X_mask_bool):
        pad_mask = ~X_mask_bool
        h = self.proj(X)
        h = h + self.cnn(h.transpose(1, 2)).transpose(1, 2)
        h   = self.pe(h)
        mem = self.encoder(h, src_key_padding_mask=pad_mask)
        return self.ln(mem), pad_mask


class TextDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers,
                 d_ff, dropout, pad_id, emb_dim=None):
        super().__init__()
        # emb_dim=768 for g4_e3b (AraBERT) — projection 768→128
        # emb_dim=None or =d_model for scratch
        actual_emb_dim = emb_dim if emb_dim and emb_dim != d_model else d_model
        self.embed    = nn.Embedding(vocab_size, actual_emb_dim, padding_idx=pad_id)
        self.emb_proj = nn.Linear(actual_emb_dim, d_model, bias=False) \
                        if actual_emb_dim != d_model else None
        self.pe       = SinusoidalPositionalEncoding(d_model)
        dec_layer     = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True)
        self.decoder  = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.ln       = nn.LayerNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, y_inp, mem, mem_kpm, y_pad_mask):
        L   = y_inp.shape[1]
        emb = self.embed(y_inp)
        if self.emb_proj is not None:
            emb = self.emb_proj(emb)
        y        = self.pe(emb)
        tgt_mask = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=y.device), diagonal=1)
        h = self.decoder(tgt=y, memory=mem, tgt_mask=tgt_mask,
                         tgt_key_padding_mask=y_pad_mask,
                         memory_key_padding_mask=mem_kpm)
        return self.lm_head(self.ln(h))


class Sign2TextTransformer(nn.Module):
    def __init__(self, d_in, vocab_size, d_model, nhead,
                 num_enc, num_dec, d_ff, dropout, pad_id, emb_dim=None):
        super().__init__()
        self.encoder = VideoEncoder(d_in, d_model, nhead, num_enc, d_ff, dropout)
        self.decoder = TextDecoder(vocab_size, d_model, nhead, num_dec,
                                   d_ff, dropout, pad_id, emb_dim=emb_dim)
        self.pad_id  = pad_id

# ──────────────────────────────────────────────────────────────
# TOKENIZER LOADING
# ──────────────────────────────────────────────────────────────
print("Loading word tokenizer...")
with open(W2ID_PATH, encoding='utf-8') as f:
    word2id = json.load(f)
with open(ID2W_PATH, encoding='utf-8') as f:
    id2word = json.load(f)

def decode_ids(ids: list) -> str:
    skip = {BOS_ID, PAD_ID, EOS_ID}
    return ' '.join(
        id2word.get(str(i), '') for i in ids
        if i not in skip and id2word.get(str(i), '')
    )

# ──────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────
print("Loading g4_e3b model...")
ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
state = ckpt['model_state']

# Auto-detect emb_dim from the checkpoint
emb_dim_ckpt = None
if 'decoder.embed.weight' in state:
    emb_dim_ckpt = state['decoder.embed.weight'].shape[1]
    print(f"  detected emb_dim: {emb_dim_ckpt}")

model = Sign2TextTransformer(
    d_in=D_IN, vocab_size=VOCAB_SIZE,
    d_model=D_MODEL, nhead=N_HEAD,
    num_enc=NUM_ENC, num_dec=NUM_DEC,
    d_ff=D_FF, dropout=DROPOUT,
    pad_id=PAD_ID, emb_dim=emb_dim_ckpt
).to(DEVICE)

model.load_state_dict(state, strict=True)
model.eval()
print(f"Model loaded on {DEVICE} | "
      f"Epoch {ckpt.get('epoch','?')} | "
      f"Val BLEU-4 {ckpt.get('val_bleu4', ckpt.get('val_metrics',{}).get('bleu4','?')):.2f}")

# ──────────────────────────────────────────────────────────────
# FIX #3 — Step-based downsampling (matches training).
# Old api.py used np.interp across all 372 channels, which fabricated
# intermediate frames the model never saw at train time.
# ──────────────────────────────────────────────────────────────
def downsample_indices(n, src_fps, tgt_fps):
    if not src_fps or src_fps <= 0 or not tgt_fps or tgt_fps <= 0:
        return np.arange(n)
    step = max(1, int(round(src_fps / tgt_fps)))
    return np.arange(0, n, step)

# ──────────────────────────────────────────────────────────────
# FIX #2 — Per-frame spatial normalisation (matches training).
# Old api.py fed raw MediaPipe coordinates (range ~[0, 1]) directly
# to the encoder, while the model was trained on landmarks centred on
# the hip midpoint and scaled by the inter-shoulder distance.
# ──────────────────────────────────────────────────────────────
def normalize_landmarks_frame(vec):
    pose_dim = 33 * 3
    pose     = vec[:pose_dim].reshape(-1, 3) if vec.shape[0] >= pose_dim else None

    def safe_point(arr, i):
        if arr is None or i >= arr.shape[0]:
            return None
        p = arr[i]
        return None if np.allclose(p, 0) else p

    if pose is not None and pose.shape[0] >= 25:
        lhip = safe_point(pose, 23)
        rhip = safe_point(pose, 24)
        lsho = safe_point(pose, 11)
        rsho = safe_point(pose, 12)

        if lhip is not None and rhip is not None:
            center = (lhip + rhip) / 2.0
        elif lsho is not None and rsho is not None:
            center = (lsho + rsho) / 2.0
        else:
            center = np.zeros(3, dtype=np.float32)

        if lsho is not None and rsho is not None:
            scale = np.linalg.norm(lsho[:2] - rsho[:2]) + 1e-6
        elif lhip is not None and rhip is not None:
            scale = np.linalg.norm(lhip[:2] - rhip[:2]) + 1e-6
        else:
            scale = 1.0
    else:
        center = np.zeros(3, dtype=np.float32)
        scale  = 1.0

    out = vec.copy().reshape(-1, 3)
    out = (out - center) / scale
    return out.reshape(-1)

# ──────────────────────────────────────────────────────────────
# EXTRACT LANDMARKS (MediaPipe Holistic)
# Returns (X, valid_mask) — see FIX #4 below.
# ──────────────────────────────────────────────────────────────
mp_holistic = mp.solutions.holistic

def extract_landmarks(video_path: str):
    """
    Extract MediaPipe Holistic landmarks from a video.

    Returns
    -------
    X         : float32 (T, 372)  — normalised landmark vectors
    valid_mask: bool    (T,)      — True iff pose OR a hand was detected
                                    in the original frame (FIX #4)

    Pipeline matches 02_extract_landmarks_local.py exactly:
        detect → step-based downsample → per-frame spatial normalise.
    """
    cap     = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    print(f"  source FPS: {src_fps:.1f}")

    frames, valid = [], []
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)

            pose = (np.array([[lm.x, lm.y, lm.z]
                    for lm in res.pose_landmarks.landmark]).flatten()
                    if res.pose_landmarks else np.zeros(33 * 3))

            lh = (np.array([[lm.x, lm.y, lm.z]
                  for lm in res.left_hand_landmarks.landmark]).flatten()
                  if res.left_hand_landmarks else np.zeros(21 * 3))

            rh = (np.array([[lm.x, lm.y, lm.z]
                  for lm in res.right_hand_landmarks.landmark]).flatten()
                  if res.right_hand_landmarks else np.zeros(21 * 3))

            if res.face_landmarks:
                all_face = np.array([[lm.x, lm.y, lm.z]
                                     for lm in res.face_landmarks.landmark],
                                    dtype=np.float32)
                if all_face.shape[0] > MAX_FACE_IDX:
                    face = all_face[FACE_IDX].flatten()
                else:
                    # Partial-detection fallback (matches training)
                    face = np.zeros(len(FACE_IDX) * 3, dtype=np.float32)
                    safe = [i for i in FACE_IDX if i < all_face.shape[0]]
                    face[:len(safe) * 3] = all_face[safe].flatten()
            else:
                face = np.zeros(len(FACE_IDX) * 3)

            frames.append(np.concatenate([pose, lh, rh, face]).astype(np.float32))

            # FIX #4 — track per-frame validity
            valid.append(int(
                (res.pose_landmarks       is not None) or
                (res.left_hand_landmarks  is not None) or
                (res.right_hand_landmarks is not None)
            ))

    cap.release()

    if not frames:
        return (np.zeros((0, D_IN), dtype=np.float32),
                np.zeros((0,),     dtype=bool))

    arr   = np.stack(frames, axis=0)            # (T_orig, 372)
    valid = np.array(valid, dtype=np.int8)      # (T_orig,)

    # 1. Step-based downsampling to TARGET_FPS (FIX #3)
    idx   = downsample_indices(arr.shape[0], src_fps, TARGET_FPS)
    arr   = arr[idx]
    valid = valid[idx]

    # 2. Per-frame spatial normalisation (FIX #2)
    arr = np.stack([normalize_landmarks_frame(v) for v in arr], axis=0)
    arr = arr.astype(np.float32)

    valid_mask = (valid > 0)
    print(f"  {len(frames)} raw frames → {len(arr)} frames @ {TARGET_FPS} FPS "
          f"(valid: {int(valid_mask.sum())}/{len(valid_mask)})")
    return arr, valid_mask


# ──────────────────────────────────────────────────────────────
# SEGMENTATION — overlap windows of SEGMENT_LEN frames.
# Mask is now (real_frame AND valid) so the encoder ignores frames
# where MediaPipe failed to detect anything (FIX #4).
# ──────────────────────────────────────────────────────────────
def split_segments(landmarks: np.ndarray, valid_mask: np.ndarray):
    T = len(landmarks)
    segments = []

    if T <= SEGMENT_LEN:
        pad  = np.zeros((SEGMENT_LEN - T, D_IN), dtype=np.float32)
        seg  = np.concatenate([landmarks, pad], axis=0)
        mask = np.zeros(SEGMENT_LEN, dtype=bool)
        mask[:T] = valid_mask
        segments.append((seg, mask))
    else:
        for start in range(0, T, STRIDE):
            end      = min(start + SEGMENT_LEN, T)
            n_frames = end - start
            if n_frames < MIN_FRAMES:
                break
            seg_x = landmarks[start:end]
            seg_v = valid_mask[start:end]
            mask  = np.zeros(SEGMENT_LEN, dtype=bool)
            mask[:n_frames] = seg_v
            if n_frames < SEGMENT_LEN:
                pad   = np.zeros((SEGMENT_LEN - n_frames, D_IN), dtype=np.float32)
                seg_x = np.concatenate([seg_x, pad], axis=0)
            segments.append((seg_x.astype(np.float32), mask))

    return segments

# ──────────────────────────────────────────────────────────────
# INFERENCE — greedy decoding (best speed/quality trade-off, group 5)
# ──────────────────────────────────────────────────────────────
def predict_segment(seg: np.ndarray, mask: np.ndarray, max_len=80) -> str:
    X      = torch.tensor(seg).unsqueeze(0).to(DEVICE)
    X_mask = torch.tensor(mask).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        mem, mem_kpm = model.encoder(X, X_mask)
        generated    = [BOS_ID]

        for _ in range(max_len):
            y_inp  = torch.tensor([generated], device=DEVICE)
            y_mask = torch.zeros(1, len(generated),
                                 dtype=torch.bool, device=DEVICE)
            logits = model.decoder(y_inp, mem, mem_kpm, y_mask)
            nxt    = logits[0, -1, :].argmax().item()
            if nxt == EOS_ID:
                break
            generated.append(nxt)

    return decode_ids(generated[1:])


# ──────────────────────────────────────────────────────────────
# Bonus FIX — overlap-aware merge of per-segment predictions.
# The training segmentation gives consecutive segments overlapping
# ground-truth text (TEXT_OVERLAP=0.4 in 02b_dual_overlap_segmentation.py),
# so consecutive predictions tend to share words at the boundary.
# We exploit this by detecting token-level suffix↔prefix overlap and
# only appending the non-overlapping tail.
# ──────────────────────────────────────────────────────────────
def _is_contiguous_subseq(needle, haystack):
    n, h = len(needle), len(haystack)
    if n == 0 or n > h:
        return False
    for i in range(h - n + 1):
        if haystack[i:i + n] == needle:
            return True
    return False


def _merge_overlapping(segments_text):
    # 1. dedupe identical predictions (keep first occurrence)
    seen, deduped = set(), []
    for txt in segments_text:
        norm = ' '.join(txt.split())
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append(txt)

    # 2. drop any segment that is a STRICT contiguous subsequence
    #    (i.e. shorter than) some other kept segment
    keep = []
    for i, txt in enumerate(deduped):
        toks_i = txt.split()
        is_strict_sub = any(
            i != j
            and len(deduped[j].split()) > len(toks_i)
            and _is_contiguous_subseq(toks_i, deduped[j].split())
            for j in range(len(deduped))
        )
        if not is_strict_sub:
            keep.append(txt)

    # 3. forward token-level suffix↔prefix merge
    merged = []
    for txt in keep:
        toks = txt.split()
        if not toks:
            continue
        if not merged:
            merged = list(toks)
            continue
        max_k = min(len(merged), len(toks))
        overlap = 0
        for k in range(max_k, 0, -1):
            if merged[-k:] == toks[:k]:
                overlap = k
                break
        merged.extend(toks[overlap:])

    # 4. collapse consecutive duplicate tokens
    cleaned = []
    for t in merged:
        if not cleaned or cleaned[-1] != t:
            cleaned.append(t)
    return ' '.join(cleaned)


def predict(landmarks: np.ndarray, valid_mask: np.ndarray) -> str:
    segments = split_segments(landmarks, valid_mask)
    print(f"  {len(segments)} segment(s) to process...")

    per_segment = []
    for i, (seg, mask) in enumerate(segments):
        text = predict_segment(seg, mask).strip()
        print(f"  Segment {i+1}/{len(segments)} : {text}")
        if text:
            per_segment.append(text)

    if not per_segment:
        return ""

    final = _merge_overlapping(per_segment)
    print(f"  Final result : {final}")
    return final

# ──────────────────────────────────────────────────────────────
# FASTAPI
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="إفهمني API",
    description="Translation of Tunisian Arabic Sign Language → Arabic",
    version="2.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    val_bleu4 = ckpt.get('val_bleu4',
                ckpt.get('val_metrics', {}).get('bleu4', 0))
    return {
        "status"   : "ok",
        "model"    : "g4_e3b — AraBERT progressive unfreeze ep30",
        "device"   : str(DEVICE),
        "epoch"    : int(ckpt.get('epoch', 0)),
        "val_bleu4": float(val_bleu4),
        "d_in"     : D_IN,
        "vocab"    : VOCAB_SIZE,
        "emb_dim"  : emb_dim_ckpt,
    }

@app.post("/predict")
async def predict_route(video: UploadFile = File(...)):
    suffix = Path(video.filename).suffix or '.mp4'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    try:
        print(f"\n[PREDICT] File: {video.filename}")
        landmarks, valid_mask = extract_landmarks(tmp_path)

        if len(landmarks) == 0:
            return JSONResponse(status_code=422,
                                content={"error": "No landmarks detected"})

        text = predict(landmarks, valid_mask)
        return {
            "text"        : text,
            "frames"      : int(len(landmarks)),
            "valid_frames": int(valid_mask.sum()),
            "model"       : "g4_e3b",
        }

    except Exception as e:
        import traceback
        print(f"[ERROR] {e}\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        try: os.unlink(tmp_path)
        except: pass

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
