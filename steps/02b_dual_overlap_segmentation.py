"""
02b_dual_overlap_segmentation.py
Usage : python steps/02b_dual_overlap_segmentation.py
"""

import os, json, re, glob
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

BASE         = str(Path(__file__).resolve().parent.parent)
LANDMARKS    = os.path.join(BASE, 'landmarks')
TEXTS        = os.path.join(BASE, 'text_cleaned')
SEGMENTS_DIR = os.path.join(BASE, 'segments')
RESULTS      = os.path.join(BASE, 'results')

os.makedirs(SEGMENTS_DIR, exist_ok=True)
os.makedirs(RESULTS,      exist_ok=True)

# D attendu depuis l'extraction (02_extract_landmarks_local.py)
# D = (33+21+21+49)*3 = 372
D_EXPECTED = 372

SEGMENT_LEN   = 300
VIDEO_OVERLAP = 0.5
TEXT_OVERLAP  = 0.4
MIN_WORDS     = 2
MIN_FRAMES    = 30
STRIDE        = int(SEGMENT_LEN * (1 - VIDEO_OVERLAP))

_HARAKAT = re.compile(r'[\u064B-\u0652\u0670\u0655\u0653\u0654]')
_TATWEEL = re.compile(r'\u0640')

def clean(text):
    text = _TATWEEL.sub('', text)
    text = _HARAKAT.sub('', text)
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def assign_text_to_segment(words, frame_start, frame_end, T_total, text_overlap=0.4):
    N = len(words)
    if N == 0 or T_total == 0:
        return ''
    pos_start = frame_start / T_total
    pos_end   = frame_end   / T_total
    win_start = max(0.0, pos_start - text_overlap / 2)
    win_end   = min(1.0, pos_end   + text_overlap / 2)
    selected  = [w for k, w in enumerate(words)
                 if win_start <= (k + 0.5) / N <= win_end]
    return ' '.join(selected)

def segment_video(vid, npz_path, txt_path):
    try:
        data = np.load(npz_path)
        X    = data['X'].astype(np.float32)
        mask = data['mask'].astype(np.int8)
    except Exception:
        return []

    T_total = X.shape[0]
    D_actual = X.shape[1]
    if D_actual != D_EXPECTED:
        import warnings
        warnings.warn(f'{vid}: D={D_actual} inattendu (attendu {D_EXPECTED}). Vérifier 02_extract_landmarks_local.py')

    try:
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_text = f.read()
        cleaned_text = clean(raw_text)
        words        = cleaned_text.split()
    except Exception:
        return []

    if len(words) < MIN_WORDS:
        return []

    segments_meta = []

    if T_total <= SEGMENT_LEN:
        seg_id   = f'{vid}_seg000'
        out_path = os.path.join(SEGMENTS_DIR, seg_id + '.npz')
        np.savez_compressed(out_path, X=X, mask=mask,
                            text=np.array(cleaned_text), src_vid=np.array(vid))
        segments_meta.append({
            'seg_id': seg_id, 'src_vid': vid,
            'frame_start': 0, 'frame_end': T_total,
            'n_frames': T_total, 'text': cleaned_text,
            'n_words': len(words), 'npz_path': out_path
        })
        return segments_meta

    for seg_idx, frame_start in enumerate(range(0, T_total, STRIDE)):
        frame_end = min(frame_start + SEGMENT_LEN, T_total)
        n_frames  = frame_end - frame_start
        if n_frames < MIN_FRAMES:
            break

        seg_text  = assign_text_to_segment(words, frame_start, frame_end, T_total, TEXT_OVERLAP)
        seg_words = seg_text.split()
        if len(seg_words) < MIN_WORDS:
            continue

        X_seg    = X[frame_start:frame_end]
        mask_seg = mask[frame_start:frame_end]
        seg_id   = f'{vid}_seg{seg_idx:03d}'
        out_path = os.path.join(SEGMENTS_DIR, seg_id + '.npz')

        np.savez_compressed(out_path, X=X_seg, mask=mask_seg,
                            text=np.array(seg_text), src_vid=np.array(vid))
        segments_meta.append({
            'seg_id': seg_id, 'src_vid': vid,
            'frame_start': int(frame_start), 'frame_end': int(frame_end),
            'n_frames': int(n_frames), 'text': seg_text,
            'n_words': len(seg_words), 'npz_path': out_path
        })

    return segments_meta


if __name__ == '__main__':
    npz_files    = sorted(glob.glob(os.path.join(LANDMARKS, '*.npz')))
    all_segments = []
    skipped      = []

    for npz_path in tqdm(npz_files, desc='Segmentation'):
        vid      = os.path.splitext(os.path.basename(npz_path))[0]
        txt_path = os.path.join(TEXTS, vid + '.txt')
        if not os.path.exists(txt_path):
            skipped.append(vid)
            continue
        segs = segment_video(vid, npz_path, txt_path)
        if segs:
            all_segments.extend(segs)
        else:
            skipped.append(vid)

    manifest_path = os.path.join(RESULTS, 'segments_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(all_segments, f, ensure_ascii=False, indent=2)

    pd.DataFrame(all_segments).to_csv(
        os.path.join(RESULTS, 'segments_manifest.csv'), index=False)

    print(f'\nSegments crees : {len(all_segments)}')
    print(f'Ignores        : {len(skipped)}')
    print(f'Facteur        : x{len(all_segments)/max(1,len(npz_files)-len(skipped)):.1f}')
    print(f'Manifeste      : {manifest_path}')