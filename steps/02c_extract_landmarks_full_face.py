"""
02c_extract_landmarks_full_face.py
Emplacement : PFA_Sign2Text/steps/02c_extract_landmarks_full_face.py

Extraction des landmarks MediaPipe avec les 468 points du visage complets.
Sauvegarde dans landmarks_full/ (ne touche pas landmarks/ existant).

Layout du vecteur de sortie (D=1629) :
  cols   0.. 98  : pose        (33 pts x 3)
  cols  99..161  : main gauche (21 pts x 3)
  cols 162..224  : main droite (21 pts x 3)
  cols 225..1628 : visage complet (468 pts x 3)

Usage :
    cd PFA_Sign2Text
    python steps/02c_extract_landmarks_full_face.py

Prerequis :
    pip install mediapipe opencv-python numpy pandas tqdm
"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import mediapipe as mp

# ──────────────────────────────────────────────────────────────
# CHEMINS
# ──────────────────────────────────────────────────────────────
BASE         = str(Path(__file__).resolve().parent.parent)
VIDEOS       = os.path.join(BASE, 'videos')
LANDMARKS    = os.path.join(BASE, 'landmarks_full')   # nouveau dossier
RESULTS      = os.path.join(BASE, 'results')

os.makedirs(LANDMARKS, exist_ok=True)
os.makedirs(RESULTS,   exist_ok=True)

# ──────────────────────────────────────────────────────────────
# HYPERPARAMETRES
# ──────────────────────────────────────────────────────────────
TARGET_FPS       = 20
MODEL_COMPLEXITY = 1
REFINE_FACE      = False   # 468 pts de base, sans iris
STANDARDIZE      = False
T_MAX            = None

# Dimensions
N_FACE   = 468
D_TOTAL  = (33 + 21 + 21 + N_FACE) * 3   # = 1629

print('=' * 60)
print('02c_extract_landmarks_full_face.py')
print('=' * 60)
print(f'  VIDEOS      : {VIDEOS}')
print(f'  LANDMARKS   : {LANDMARKS}')
print(f'  TARGET_FPS  : {TARGET_FPS}')
print(f'  FACE_POINTS : {N_FACE} (tous les points MediaPipe de base)')
print(f'  D_TOTAL     : {D_TOTAL}')
print('=' * 60)

mp_holistic = mp.solutions.holistic


# ──────────────────────────────────────────────────────────────
# UTILITAIRES
# ──────────────────────────────────────────────────────────────

def downsample_indices(n, src_fps, tgt_fps):
    if not src_fps or src_fps <= 0 or not tgt_fps or tgt_fps <= 0:
        return np.arange(n)
    step = max(1, int(round(src_fps / tgt_fps)))
    return np.arange(0, n, step)


def normalize_landmarks_frame(vec):
    """
    Normalisation spatiale : centrage sur hanches/epaules,
    mise a l'echelle par distance inter-epaules.
    Identique a 02_extract_landmarks_local.py.
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
# EXTRACTION D'UNE VIDEO
# ──────────────────────────────────────────────────────────────

def extract_video_npz(video_path, out_npz_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'ok': False, 'reason': 'cannot_open'}

    src_fps      = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    holistic = mp_holistic.Holistic(
        static_image_mode    = False,
        model_complexity     = MODEL_COMPLEXITY,
        smooth_landmarks     = True,
        enable_segmentation  = False,
        refine_face_landmarks= REFINE_FACE
    )

    seq, valid = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holistic.process(rgb)
        parts = []

        # ── Pose (33 pts) ─────────────────────────────────────
        if res.pose_landmarks is not None:
            pose_pts = np.array(
                [(lm.x, lm.y, lm.z) for lm in res.pose_landmarks.landmark],
                dtype=np.float32)
        else:
            pose_pts = np.zeros((33, 3), dtype=np.float32)
        parts.append(pose_pts.reshape(-1))

        # ── Main gauche (21 pts) ──────────────────────────────
        if res.left_hand_landmarks is not None:
            lhand = np.array(
                [(lm.x, lm.y, lm.z) for lm in res.left_hand_landmarks.landmark],
                dtype=np.float32)
        else:
            lhand = np.zeros((21, 3), dtype=np.float32)
        parts.append(lhand.reshape(-1))

        # ── Main droite (21 pts) ──────────────────────────────
        if res.right_hand_landmarks is not None:
            rhand = np.array(
                [(lm.x, lm.y, lm.z) for lm in res.right_hand_landmarks.landmark],
                dtype=np.float32)
        else:
            rhand = np.zeros((21, 3), dtype=np.float32)
        parts.append(rhand.reshape(-1))

        # ── Visage complet (468 pts) ──────────────────────────
        if res.face_landmarks is not None:
            face_all = np.array(
                [(lm.x, lm.y, lm.z) for lm in res.face_landmarks.landmark],
                dtype=np.float32)
            if face_all.shape[0] >= N_FACE:
                face_sel = face_all[:N_FACE]
            else:
                face_sel = np.zeros((N_FACE, 3), dtype=np.float32)
                face_sel[:face_all.shape[0]] = face_all
        else:
            face_sel = np.zeros((N_FACE, 3), dtype=np.float32)
        parts.append(face_sel.reshape(-1))

        # ── Concatenation → (D=1629,) ─────────────────────────
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
        return {'ok': False, 'reason': 'no_frames'}

    seq   = np.stack(seq, axis=0)
    valid = np.array(valid, dtype=np.int8)

    # Reduction FPS
    idx   = downsample_indices(seq.shape[0], src_fps, TARGET_FPS)
    seq   = seq[idx]
    valid = valid[idx]

    # Troncature optionnelle
    if T_MAX is not None and seq.shape[0] > T_MAX:
        seq   = seq[:T_MAX]
        valid = valid[:T_MAX]

    # Normalisation spatiale frame par frame
    seq = np.stack([normalize_landmarks_frame(v) for v in seq], axis=0)

    if STANDARDIZE:
        mean = seq.mean(axis=0, keepdims=True)
        std  = seq.std(axis=0,  keepdims=True) + 1e-6
        seq  = (seq - mean) / std

    mask = (valid > 0).astype(np.int8)

    np.savez_compressed(
        out_npz_path,
        X       = seq.astype(np.float32),
        mask    = mask.astype(np.int8),
        src_fps = np.float32(src_fps),
        tgt_fps = np.float32(TARGET_FPS)
    )

    return {
        'ok'         : True,
        'frames_raw' : int(total_frames),
        'frames_used': int(seq.shape[0]),
        'src_fps'    : float(src_fps),
        'tgt_fps'    : int(TARGET_FPS),
        'D'          : int(seq.shape[1])
    }


# ──────────────────────────────────────────────────────────────
# TRAITEMENT DE TOUTES LES VIDEOS
# ──────────────────────────────────────────────────────────────

def run_extraction():
    videos = sorted([f for f in os.listdir(VIDEOS) if f.lower().endswith('.mp4')])
    if not videos:
        print(f'Aucune video .mp4 trouvee dans : {VIDEOS}')
        return None

    print(f'\n{len(videos)} videos trouvees → extraction en cours...\n')
    rows = []

    for v in tqdm(videos, desc='Extraction full face'):
        vid   = os.path.splitext(v)[0]
        vpath = os.path.join(VIDEOS, v)
        opath = os.path.join(LANDMARKS, vid + '.npz')

        if os.path.exists(opath):
            try:
                data = np.load(opath)
                info = {
                    'ok'         : True,
                    'frames_raw' : None,
                    'frames_used': int(data['X'].shape[0]),
                    'src_fps'    : float(data['src_fps']),
                    'tgt_fps'    : int(data['tgt_fps']),
                    'D'          : int(data['X'].shape[1])
                }
                if info['D'] != D_TOTAL:
                    print(f'  Re-extraction {vid} : D={info["D"]} != {D_TOTAL}')
                    info = extract_video_npz(vpath, opath)
            except Exception as e:
                info = extract_video_npz(vpath, opath)
        else:
            info = extract_video_npz(vpath, opath)

        rows.append({
            'video_id'   : vid,
            'ok'         : info.get('ok', False),
            'frames_used': info.get('frames_used'),
            'D'          : info.get('D'),
            'reason'     : info.get('reason', '')
        })

    df = pd.DataFrame(rows).sort_values('video_id').reset_index(drop=True)

    csv_path = os.path.join(RESULTS, 'metadata_landmarks_full.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')

    print('\n' + '=' * 60)
    print('RESULTATS')
    print('=' * 60)
    print(f'  Succes  : {df["ok"].sum()} / {len(df)}')
    print(f'  Echecs  : {(~df["ok"]).sum()}')
    d_vals = df['D'].dropna().unique().tolist()
    print(f'  D unique: {d_vals}  (attendu: [{D_TOTAL}])')
    print(f'  CSV     : {csv_path}')

    if (~df['ok']).any():
        print('\n  Videos en echec :')
        for _, r in df[~df['ok']].iterrows():
            print(f'    - {r["video_id"]} : {r["reason"]}')
    print('=' * 60)
    return df


# ──────────────────────────────────────────────────────────────
# VERIFICATION D'UN NPZ
# ──────────────────────────────────────────────────────────────

def verify_sample():
    import random
    npzs = [f for f in os.listdir(LANDMARKS) if f.endswith('.npz')]
    if not npzs:
        print('Aucun .npz trouve.')
        return

    name = random.choice(npzs)
    path = os.path.join(LANDMARKS, name)
    data = np.load(path)
    X, mask = data['X'], data['mask']

    print(f'\nVerification : {name}')
    print(f'  X.shape        = {X.shape}')
    print(f'  D              = {X.shape[1]}  (attendu : {D_TOTAL})')
    print(f'  Frames valides : {int(mask.sum())} / {len(mask)}')
    print(f'  X stats        : min={X.min():.3f} max={X.max():.3f} mean={X.mean():.4f}')

    if X.shape[1] == D_TOTAL:
        print(f'  OK — dimension correcte ({D_TOTAL})')
    else:
        print(f'  ERREUR — dimension inattendue : {X.shape[1]}')


# ──────────────────────────────────────────────────────────────
# POINT D'ENTREE
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    df = run_extraction()
    if df is not None and df['ok'].any():
        verify_sample()
        print('\nExtraction terminee.')
        print(f'Prochaine etape : python steps/02d_segment_full_face.py')
