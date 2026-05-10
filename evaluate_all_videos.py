"""
evaluate_all_videos.py
======================
Adapted version of evaluate_demo_candidates.py — runs end-to-end inference
on the FULL dataset (train + val + test) by default and keeps every video
whose sentence-level BLEU-4 is at least 90.

Key differences vs the original script
--------------------------------------
1. --split defaults to "all" (train + val + test = 665 videos).
2. Each result row records which split the video came from, so demo
   candidates can be filtered by origin if you want to avoid showing
   train videos at the defense.
3. Resumable. The CSV is written incrementally (one row per video,
   flushed to disk immediately). If the run is interrupted (Ctrl-C,
   power cut, OOM), just re-run and the script picks up from the last
   completed video. Use --restart to ignore existing progress.
4. ETA estimate during the loop, plus a per-split summary at the end.

Compatible with both the patched api.py (extract_landmarks returns
(X, valid_mask), predict takes (X, valid_mask)) and the original
(extract_landmarks returns X, predict takes X).

Usage
-----
    # Default: full dataset, threshold 90, resumable
    python evaluate_all_videos.py

    # Lower threshold to get more candidates
    python evaluate_all_videos.py --threshold 80

    # Force a clean run (ignore previous CSV)
    python evaluate_all_videos.py --restart

    # Limit to a specific split if you want to test on val/test only
    python evaluate_all_videos.py --split test
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Sacrebleu — same metric as training (char-level BLEU-4)
# ---------------------------------------------------------------------------
try:
    from sacrebleu.metrics import BLEU as SacreBLEU
except ImportError:
    sys.exit("sacrebleu is required.  Install with:  pip install sacrebleu")

# ---------------------------------------------------------------------------
# Import the inference module.  This has the side-effect of loading the
# model and tokenizer — slow but only happens once.
# ---------------------------------------------------------------------------
print("Loading model and tokenizer (importing api.py)...")
import api  # noqa: E402
print("Model loaded.\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
CSV_FIELDS = ["video_id", "split", "bleu4", "reference", "prediction",
              "n_frames", "valid_frames", "elapsed_s", "error"]


def read_ground_truth(text_dir: Path, vid: str) -> str:
    txt_path = text_dir / f"{vid}.txt"
    if not txt_path.is_file():
        return ""
    raw = txt_path.read_text(encoding="utf-8", errors="ignore")
    return api.clean_arabic(raw)


def load_split_map(split_path: Path, requested_split: str):
    """Return a list of (video_id, split_name) tuples."""
    if not split_path.is_file():
        sys.exit(f"split.json not found at {split_path}")
    split = json.loads(split_path.read_text(encoding="utf-8"))

    if requested_split == "all":
        wanted = ["train", "val", "test"]
    else:
        if requested_split not in split:
            sys.exit(f"Split '{requested_split}' not found. "
                     f"Available: {list(split.keys())}")
        wanted = [requested_split]

    pairs = []
    for s in wanted:
        for vid in split.get(s, []):
            pairs.append((vid, s))
    return pairs


def evaluate_video(video_path: Path, reference: str,
                   bleu_metric: SacreBLEU) -> dict:
    """Run the full pipeline on one video and return a result row."""
    t0 = time.time()
    # Patched api.py returns (X, valid_mask); original returns just X.
    out = api.extract_landmarks(str(video_path))
    if isinstance(out, tuple):
        landmarks, valid_mask = out
    else:
        landmarks = out
        valid_mask = None

    if landmarks.size == 0:
        return {"prediction": "", "bleu4": 0.0, "n_frames": 0,
                "valid_frames": 0,
                "elapsed_s": round(time.time() - t0, 2),
                "error": "no_landmarks"}

    if valid_mask is not None:
        prediction = api.predict(landmarks, valid_mask)
    else:
        prediction = api.predict(landmarks)

    bleu = bleu_metric.sentence_score(prediction, [reference]).score

    return {
        "prediction": prediction,
        "bleu4": round(float(bleu), 2),
        "n_frames": int(landmarks.shape[0]),
        "valid_frames": int(valid_mask.sum()) if valid_mask is not None else -1,
        "elapsed_s": round(time.time() - t0, 2),
        "error": "",
    }


def load_existing_results(csv_path: Path):
    """Read an existing CSV (if any) so we can skip already-done videos."""
    if not csv_path.is_file():
        return {}, []
    done = {}
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                r["bleu4"] = float(r.get("bleu4", 0) or 0)
            except ValueError:
                r["bleu4"] = 0.0
            for k in ("n_frames", "valid_frames", "elapsed_s"):
                try:
                    r[k] = float(r[k]) if k == "elapsed_s" else int(r[k])
                except (KeyError, ValueError, TypeError):
                    r[k] = 0
            done[r["video_id"]] = r
            rows.append(r)
    return done, rows


def fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=api.BASE,
                        help="Project root (default: api.BASE = %(default)s)")
    parser.add_argument("--split", default="all",
                        choices=["train", "val", "test", "all"],
                        help="Which split to evaluate (default: %(default)s)")
    parser.add_argument("--threshold", type=float, default=90.0,
                        help="Minimum BLEU-4 to keep a video (default: %(default)s)")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Stop after N videos (useful for a quick test)")
    parser.add_argument("--video-dir", default=None,
                        help="Override path to the videos folder")
    parser.add_argument("--text-dir", default=None,
                        help="Override path to the cleaned-texts folder")
    parser.add_argument("--output-dir", default="demo_results_all",
                        help="Where to save CSV/JSON outputs (default: %(default)s)")
    parser.add_argument("--restart", action="store_true",
                        help="Ignore existing CSV and re-evaluate everything")
    parser.add_argument("--verbose", action="store_true",
                        help="Print reference/prediction for every video")
    args = parser.parse_args()

    base = Path(args.base)
    video_dir = Path(args.video_dir) if args.video_dir else base / "videos"
    if args.text_dir:
        text_dir = Path(args.text_dir)
    else:
        text_dir = base / "text_cleaned"
        if not text_dir.is_dir():
            text_dir = base / "texts"
    split_path = base / "results" / "split.json"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "all_results.csv"

    print(f"Project root : {base}")
    print(f"Videos dir   : {video_dir}")
    print(f"Texts dir    : {text_dir}")
    print(f"Split file   : {split_path}")
    print(f"Split        : {args.split}")
    print(f"Threshold    : BLEU-4 >= {args.threshold}")
    print(f"Output dir   : {output_dir}")
    print(f"CSV (state)  : {csv_path}\n")

    pairs = load_split_map(split_path, args.split)
    if args.max_videos is not None:
        pairs = pairs[:args.max_videos]

    # --- Resumability -------------------------------------------------------
    if args.restart and csv_path.is_file():
        csv_path.unlink()
        print("--restart: removed existing CSV.")
    done_map, existing_rows = load_existing_results(csv_path)
    if done_map:
        print(f"Resuming: {len(done_map)} video(s) already done.")

    # Open CSV in append mode; write header if file is fresh.
    new_file = not csv_path.is_file()
    csv_fh = csv_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
    if new_file:
        writer.writeheader()
        csv_fh.flush()

    todo = [(vid, sp) for (vid, sp) in pairs if vid not in done_map]
    print(f"{len(pairs)} video(s) total | "
          f"{len(done_map)} done | {len(todo)} to evaluate.\n")

    if not todo:
        print("Nothing to do. Generating final outputs from existing CSV...")

    bleu_metric = SacreBLEU(max_ngram_order=4, tokenize="char")
    started = time.time()
    completed_now = 0

    # --- Main loop ----------------------------------------------------------
    for i, (vid, split_name) in enumerate(todo, start=1):
        video_path = video_dir / f"{vid}.mp4"
        reference = read_ground_truth(text_dir, vid)

        if not video_path.is_file():
            print(f"[{i}/{len(todo)}] {vid} ({split_name}): SKIP — video not found")
            continue
        if not reference:
            print(f"[{i}/{len(todo)}] {vid} ({split_name}): SKIP — no reference")
            continue

        try:
            row = evaluate_video(video_path, reference, bleu_metric)
        except Exception as e:
            print(f"[{i}/{len(todo)}] {vid} ({split_name}): ERROR — {e}")
            row = {"prediction": "", "bleu4": 0.0, "n_frames": 0,
                   "valid_frames": 0, "elapsed_s": 0.0, "error": str(e)}

        row.update({"video_id": vid, "split": split_name,
                    "reference": reference})

        # Write immediately so progress survives interruption
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
        csv_fh.flush()

        completed_now += 1
        elapsed = time.time() - started
        avg = elapsed / completed_now
        eta = avg * (len(todo) - completed_now)

        marker = "OK ✓" if row["bleu4"] >= args.threshold else "    "
        print(f"[{i}/{len(todo)}] {vid} ({split_name}) "
              f"BLEU={row['bleu4']:6.2f} {marker}  "
              f"avg {avg:.1f}s/vid  ETA {fmt_eta(eta)}")
        if args.verbose:
            print(f"    REF: {row['reference']}")
            print(f"    HYP: {row['prediction']}")

    csv_fh.close()
    total_elapsed = time.time() - started
    print("\n" + "=" * 72)
    print(f"This run: {completed_now} new video(s) in {fmt_eta(total_elapsed)}")

    # --- Re-load the full CSV and write final JSONs ------------------------
    _, all_rows = load_existing_results(csv_path)
    all_rows.sort(key=lambda r: r["bleu4"], reverse=True)
    selected = [r for r in all_rows if r["bleu4"] >= args.threshold]

    json_all = output_dir / "all_results.json"
    json_all.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    json_sel = output_dir / "demo_candidates.json"
    json_sel.write_text(json.dumps(selected, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"All results CSV  -> {csv_path}")
    print(f"All results JSON -> {json_all}")
    print(f"Selected JSON    -> {json_sel}")

    # --- Per-split summary -------------------------------------------------
    print("\n" + "=" * 72)
    print(f"DEMO CANDIDATES (BLEU-4 >= {args.threshold})")
    print("=" * 72)
    by_split = {"train": [], "val": [], "test": [], "": []}
    for r in selected:
        by_split.setdefault(r.get("split", ""), []).append(r)

    total_by_split = {"train": 0, "val": 0, "test": 0, "": 0}
    for r in all_rows:
        total_by_split[r.get("split", "")] = total_by_split.get(
            r.get("split", ""), 0) + 1

    print(f"{'split':<8} {'kept':>6} / {'total':<6}  {'rate':>6}")
    print("-" * 32)
    for s in ("train", "val", "test"):
        kept = len(by_split.get(s, []))
        total = total_by_split.get(s, 0)
        rate = (100.0 * kept / total) if total else 0.0
        print(f"{s:<8} {kept:>6} / {total:<6}  {rate:>5.1f}%")
    print("-" * 32)
    kept_all = len(selected)
    print(f"{'TOTAL':<8} {kept_all:>6} / {len(all_rows):<6}  "
          f"{100.0 * kept_all / max(1, len(all_rows)):>5.1f}%")

    # --- Pretty list of selected videos ------------------------------------
    if not selected:
        print("\nNo candidates passed the threshold. Try --threshold 80 to "
              "see borderline cases, or check the api.py sanity block below.")
    else:
        print(f"\nTop candidates (showing all {len(selected)}, sorted by BLEU desc):")
        for r in selected:
            tag = f"[{r.get('split', '?')}]"
            print(f"\n  • {r['video_id']} {tag}  BLEU-4 = {r['bleu4']:.2f}")
            print(f"    REF: {r['reference']}")
            print(f"    HYP: {r['prediction']}")

    # --- api.py sanity check -----------------------------------------------
    print("\n" + "=" * 72)
    print("api.py SANITY CHECK")
    print("=" * 72)
    expected_face = (
        [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
         291, 375, 321, 405, 314, 17, 84, 181, 91, 146] +
        [78, 13, 312, 308, 14, 87, 88, 95, 82, 310] +
        [33, 159, 145, 133, 362, 386, 374, 263] +
        [70, 107, 55, 46, 300, 336, 285, 276] +
        [1, 4, 94]
    )
    api_face = getattr(api, "FACE_IDX", None)
    face_ok = api_face == expected_face
    norm_ok = hasattr(api, "normalize_landmarks_frame")
    down_ok = hasattr(api, "downsample_indices")
    sig_ok  = "valid_mask" in api.predict.__code__.co_varnames

    def status(ok):
        return "OK" if ok else "MISMATCH — patch not applied"

    print(f"  FACE_IDX == training indices               : {status(face_ok)}")
    print(f"  normalize_landmarks_frame() is defined     : {status(norm_ok)}")
    print(f"  downsample_indices() is defined (no interp): {status(down_ok)}")
    print(f"  predict(landmarks, valid_mask) signature   : {status(sig_ok)}")
    if not all([face_ok, norm_ok, down_ok, sig_ok]):
        print("\n  Note: this run used an UNPATCHED api.py. BLEU scores are "
              "lower than they would be with the patched version.")
    else:
        print("\n  api.py looks fully patched.")
    print()


if __name__ == "__main__":
    main()
