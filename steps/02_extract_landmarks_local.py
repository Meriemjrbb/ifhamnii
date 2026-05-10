# ============================================================
# 02_extract_landmarks_local.py
# Extraction des landmarks MediaPipe — VERSION LOCALE
# Sélection LÉGÈRE du visage : ~51 points anatomiques essentiels
# (bouche, yeux, sourcils, nez) — sans mâchoire ni iris
#
# Prérequis :
#   pip install mediapipe opencv-python numpy pandas tqdm
#
# Usage :
#   python 02_extract_landmarks_local.py
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp

# ──────────────────────────────────────────────────────────────
# CHEMINS DU PROJET
# ──────────────────────────────────────────────────────────────
BASE      = r"C:\Users\MSI\PFA_Sign2Text"
VIDEOS    = os.path.join(BASE, "videos")
LANDMARKS = os.path.join(BASE, "landmarks")
RESULTS   = os.path.join(BASE, "results")

os.makedirs(LANDMARKS, exist_ok=True)
os.makedirs(RESULTS,   exist_ok=True)

# ──────────────────────────────────────────────────────────────
# HYPERPARAMÈTRES
# ──────────────────────────────────────────────────────────────
TARGET_FPS       = 20
MODEL_COMPLEXITY = 1
REFINE_FACE      = False  # False suffit — on n'utilise pas les iris
STANDARDIZE      = False
T_MAX            = None   # ex: 600 si RAM insuffisante

# ──────────────────────────────────────────────────────────────
# SÉLECTION LÉGÈRE DU VISAGE — 51 points essentiels
# Tous dans les 468 de base → REFINE_FACE=False suffit
# ──────────────────────────────────────────────────────────────

# BOUCHE — contour extérieur (20 pts) + 10 pts intérieurs clés
# → ouverture, forme des lèvres, phonèmes visuels
LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
              291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
LIPS_INNER = [78, 13, 312, 308, 14, 87, 88, 95, 82, 310]

# YEUX — 4 points par œil (coin interne, haut, bas, coin externe)
# → ouverture / fermeture / clignements
LEFT_EYE  = [33, 159, 145, 133]
RIGHT_EYE = [362, 386, 374, 263]

# SOURCILS — 4 points par sourcil
# → CRITIQUE pour LST : questions (levés) et négation (froncés)
LEFT_EYEBROW  = [70, 107, 55, 46]
RIGHT_EYEBROW = [300, 336, 285, 276]

# NEZ — 3 points (pointe + base gauche + base droite)
# → ancrage spatial central
NOSE_TIP = [1, 4, 94]

# Assemblage et déduplication (ordre conservé)
FACE_IDX = list(dict.fromkeys(
    LIPS_OUTER    +
    LIPS_INNER    +
    LEFT_EYE      + RIGHT_EYE     +
    LEFT_EYEBROW  + RIGHT_EYEBROW +
    NOSE_TIP
))

FACE_POINTS_KEEP = len(FACE_IDX)                            # 49 points
MAX_FACE_IDX     = max(FACE_IDX)                            # dans les 468 de base
D_TOTAL          = (33 + 21 + 21 + FACE_POINTS_KEEP) * 3   # = 372

print("=" * 60)
print("CONFIGURATION — visage léger")
print("=" * 60)
print(f"  BASE             : {BASE}")
print(f"  TARGET_FPS       : {TARGET_FPS}")
print(f"  MODEL_COMPLEXITY : {MODEL_COMPLEXITY}")
print(f"  REFINE_FACE      : {REFINE_FACE}")
print(f"  FACE_POINTS      : {FACE_POINTS_KEEP} points")
print(f"    → Bouche       : {len(LIPS_OUTER) + len(LIPS_INNER)} pts  (extérieur + intérieur)")
print(f"    → Yeux         : {len(LEFT_EYE) + len(RIGHT_EYE)} pts")
print(f"    → Sourcils     : {len(LEFT_EYEBROW) + len(RIGHT_EYEBROW)} pts")
print(f"    → Nez          : {len(NOSE_TIP)} pts")
print(f"  D total          : (33+21+21+{FACE_POINTS_KEEP})×3 = {D_TOTAL}")
print("=" * 60)

mp_holistic = mp.solutions.holistic

# ──────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ──────────────────────────────────────────────────────────────

def downsample_indices(n, src_fps, tgt_fps):
    """Indices des frames à garder pour ramener src_fps → tgt_fps."""
    if not src_fps or src_fps <= 0 or not tgt_fps or tgt_fps <= 0:
        return np.arange(n)
    step = max(1, int(round(src_fps / tgt_fps)))
    return np.arange(0, n, step)


def normalize_landmarks_frame(vec):
    """
    Normalisation spatiale d'une frame :
      1) Centrage sur le milieu hanches (ou épaules si hanches absentes)
      2) Mise à l'échelle par la distance inter-épaules
    → Coordonnées relatives au corps, indépendantes de la distance caméra.
    """
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
# EXTRACTION D'UNE VIDÉO → .NPZ
# ──────────────────────────────────────────────────────────────

def extract_video_npz(video_path, out_npz_path,
                      target_fps=TARGET_FPS,
                      model_complexity=MODEL_COMPLEXITY,
                      refine_face=REFINE_FACE,
                      standardize=STANDARDIZE,
                      t_max=T_MAX):
    """
    Lit une vidéo, extrait les landmarks MediaPipe frame par frame,
    normalise et sauvegarde en .npz.

    Format de sortie .npz :
        X       : float32 (T, D)  — landmarks normalisés
                  D = (33 + 21 + 21 + 49) × 3 = 372
        mask    : int8    (T,)    — 1 si frame valide, 0 sinon
        src_fps : float32         — FPS original
        tgt_fps : float32         — FPS cible
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"ok": False, "reason": "cannot_open"}

    src_fps      = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=refine_face
    )

    seq, valid = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holistic.process(rgb)

        parts = []

        # ── 1. POSE — 33 pts × 3 = 99 features ──────────────
        if res.pose_landmarks is not None:
            pose_pts = np.array(
                [(lm.x, lm.y, lm.z) for lm in res.pose_landmarks.landmark],
                dtype=np.float32
            )
        else:
            pose_pts = np.zeros((33, 3), dtype=np.float32)
        parts.append(pose_pts.reshape(-1))

        # ── 2. MAIN GAUCHE — 21 pts × 3 = 63 features ───────
        if res.left_hand_landmarks is not None:
            lhand = np.array(
                [(lm.x, lm.y, lm.z) for lm in res.left_hand_landmarks.landmark],
                dtype=np.float32
            )
        else:
            lhand = np.zeros((21, 3), dtype=np.float32)
        parts.append(lhand.reshape(-1))

        # ── 3. MAIN DROITE — 21 pts × 3 = 63 features ───────
        if res.right_hand_landmarks is not None:
            rhand = np.array(
                [(lm.x, lm.y, lm.z) for lm in res.right_hand_landmarks.landmark],
                dtype=np.float32
            )
        else:
            rhand = np.zeros((21, 3), dtype=np.float32)
        parts.append(rhand.reshape(-1))

        # ── 4. VISAGE LÉGER — 49 pts × 3 = 147 features ─────
        if res.face_landmarks is not None:
            face_all = np.array(
                [(lm.x, lm.y, lm.z) for lm in res.face_landmarks.landmark],
                dtype=np.float32
            )
            if face_all.shape[0] > MAX_FACE_IDX:
                face_sel = face_all[FACE_IDX]
            else:
                # Fallback : indices accessibles seulement
                safe_idx = [i for i in FACE_IDX if i < face_all.shape[0]]
                face_sel = np.zeros((FACE_POINTS_KEEP, 3), dtype=np.float32)
                face_sel[:len(safe_idx)] = face_all[safe_idx]
        else:
            face_sel = np.zeros((FACE_POINTS_KEEP, 3), dtype=np.float32)
        parts.append(face_sel.reshape(-1))

        # ── Concaténation → vecteur (D,) = 99+63+63+147 = 372 ──
        vec = np.concatenate(parts, axis=0)
        seq.append(vec)

        is_valid = int(
            (res.pose_landmarks       is not None) or
            (res.left_hand_landmarks  is not None) or
            (res.right_hand_landmarks is not None)
        )
        valid.append(is_valid)

    cap.release()
    holistic.close()

    if len(seq) == 0:
        return {"ok": False, "reason": "no_frames"}

    seq   = np.stack(seq, axis=0)
    valid = np.array(valid, dtype=np.int8)

    # Réduction FPS
    idx   = downsample_indices(seq.shape[0], src_fps, target_fps)
    seq   = seq[idx]
    valid = valid[idx]

    # Troncature optionnelle
    if t_max is not None and seq.shape[0] > t_max:
        seq   = seq[:t_max]
        valid = valid[:t_max]

    # Normalisation spatiale frame par frame
    seq = np.stack([normalize_landmarks_frame(v) for v in seq], axis=0)

    # Standardisation globale (désactivée par défaut)
    if standardize:
        mean = seq.mean(axis=0, keepdims=True)
        std  = seq.std(axis=0,  keepdims=True) + 1e-6
        seq  = (seq - mean) / std

    mask = (valid > 0).astype(np.int8)

    np.savez_compressed(
        out_npz_path,
        X       = seq.astype(np.float32),
        mask    = mask.astype(np.int8),
        src_fps = np.float32(src_fps),
        tgt_fps = np.float32(target_fps)
    )

    return {
        "ok"         : True,
        "frames_raw" : int(total_frames),
        "frames_used": int(seq.shape[0]),
        "src_fps"    : float(src_fps),
        "tgt_fps"    : int(target_fps),
        "D"          : int(seq.shape[1])
    }


# ──────────────────────────────────────────────────────────────
# TRAITEMENT DE TOUTES LES VIDÉOS
# ──────────────────────────────────────────────────────────────

def run_extraction():
    videos = sorted([f for f in os.listdir(VIDEOS) if f.lower().endswith(".mp4")])
    if not videos:
        print(f"❌ Aucune vidéo .mp4 trouvée dans : {VIDEOS}")
        return None

    print(f"\n🎬 {len(videos)} vidéos trouvées → extraction en cours...\n")
    rows = []

    for v in tqdm(videos, desc="Extraction landmarks"):
        vid   = os.path.splitext(v)[0]
        vpath = os.path.join(VIDEOS, v)
        opath = os.path.join(LANDMARKS, vid + ".npz")

        if os.path.exists(opath):
            try:
                data = np.load(opath)
                info = {
                    "ok"         : True,
                    "frames_raw" : None,
                    "frames_used": int(data["X"].shape[0]),
                    "src_fps"    : float(data["src_fps"]),
                    "tgt_fps"    : int(data["tgt_fps"]),
                    "D"          : int(data["X"].shape[1])
                }
            except Exception as e:
                info = {"ok": False, "reason": f"cannot_load: {e}"}
        else:
            info = extract_video_npz(vpath, opath)

        rows.append({
            "video_id"   : vid,
            "video_path" : vpath,
            "npz_path"   : opath,
            "ok"         : info.get("ok", False),
            "frames_used": info.get("frames_used"),
            "src_fps"    : info.get("src_fps"),
            "tgt_fps"    : info.get("tgt_fps"),
            "D"          : info.get("D"),
            "reason"     : info.get("reason", "")
        })

    df = pd.DataFrame(rows).sort_values("video_id").reset_index(drop=True)

    csv_path = os.path.join(RESULTS, "metadata_landmarks.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")

    print("\n" + "=" * 60)
    print("RÉSULTATS")
    print("=" * 60)
    print(f"  ✅ Succès  : {df['ok'].sum()} / {len(df)}")
    print(f"  ❌ Échecs  : {(~df['ok']).sum()}")

    d_vals = df['D'].dropna().unique().tolist()
    print(f"  D unique   : {d_vals}  (attendu : [{D_TOTAL}])")
    if d_vals and d_vals[0] != D_TOTAL:
        print(f"  ⚠️  D inattendu — vérifier FACE_IDX ou REFINE_FACE")

    print(f"  CSV        → {csv_path}")

    if (~df['ok']).any():
        print("\n  Vidéos en échec :")
        for _, r in df[~df['ok']].iterrows():
            print(f"    - {r['video_id']} : {r['reason']}")

    print("=" * 60)
    return df


# ──────────────────────────────────────────────────────────────
# VÉRIFICATION D'UN .NPZ (contrôle qualité)
# ──────────────────────────────────────────────────────────────

def verify_sample():
    import random
    npzs = [f for f in os.listdir(LANDMARKS) if f.endswith(".npz")]
    if not npzs:
        print("❌ Aucun .npz trouvé pour la vérification.")
        return

    name = random.choice(npzs)
    path = os.path.join(LANDMARKS, name)
    data = np.load(path)
    X, mask = data["X"], data["mask"]

    print(f"\n🔎 Vérification : {name}")
    print(f"   X.shape        = {X.shape}  (T frames × D features)")
    print(f"   D              = {X.shape[1]}  (attendu : {D_TOTAL})")
    print(f"   Frames valides : {int(mask.sum())} / {len(mask)}  ({100*mask.mean():.1f}%)")
    print(f"   X  min={X.min():.3f} | max={X.max():.3f} | mean={X.mean():.4f} | std={X.std():.3f}")

    ok = True
    if abs(X.mean()) < 1.0:
        print("   ✅ Normalisation OK (mean ≈ 0)")
    else:
        print("   ⚠️  Mean loin de 0 — vérifier normalize_landmarks_frame()") ; ok = False

    if X.shape[1] == D_TOTAL:
        print(f"   ✅ Dimension D correcte ({D_TOTAL})")
    else:
        print(f"   ⚠️  D inattendu : {X.shape[1]}  (attendu {D_TOTAL})") ; ok = False

    if mask.mean() >= 0.5:
        print(f"   ✅ Ratio frames valides OK ({100*mask.mean():.1f}%)")
    else:
        print(f"   ⚠️  Moins de 50% de frames valides — vérifier qualité vidéo") ; ok = False

    if ok:
        print("\n   ✅ Tout est bon — tu peux passer à l'étape suivante (dataloader)")


# ──────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = run_extraction()
    if df is not None and df["ok"].any():
        verify_sample()
