import cv2
import os
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm


def compute_motion_score(clip_path, blur_kernel=5):
    """Compute motion score for a clip"""
    frames = sorted([f for f in os.listdir(clip_path) if f.endswith('.jpg')])

    if len(frames) < 2:
        return 0.0

    diffs = []
    prev_frame = None

    for frame_file in frames:
        frame = cv2.imread(os.path.join(clip_path, frame_file), cv2.IMREAD_GRAYSCALE)
        frame = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)

        if prev_frame is not None:
            diff = cv2.absdiff(frame, prev_frame)
            diffs.append(diff.mean())

        prev_frame = frame

    return np.mean(diffs) if diffs else 0.0


def autosuggest_labels(clips_root, out_csv, blur_kernel=5, threshold=0.5):
    """Generate suggested labels based on motion scores"""
    clips = sorted(os.listdir(clips_root))
    scores = []

    print(f"Computing motion scores for {len(clips)} clips...")

    for clip_name in tqdm(clips):
        clip_path = os.path.join(clips_root, clip_name)
        if os.path.isdir(clip_path):
            score = compute_motion_score(clip_path, blur_kernel)
            scores.append((clip_name, score))

    # Normalize scores to 0-1
    if scores:
        max_score = max(s[1] for s in scores)
        min_score = min(s[1] for s in scores)
        range_score = max_score - min_score if max_score > min_score else 1

        data = []
        for clip_name, score in scores:
            norm_score = (score - min_score) / range_score
            suggested_label = 1 if norm_score > threshold else 0
            data.append({
                'clip': clip_name,
                'score': norm_score,
                'suggested_label': suggested_label
            })

        df = pd.DataFrame(data)
        df.to_csv(out_csv, index=False)

        print(f"Saved {len(data)} suggestions to {out_csv}")
        print(f"Suggested {sum(d['suggested_label'] for d in data)} movement clips")
        print(f"Threshold: {threshold} (adjust with --threshold)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_root", default="dataset/clips", help="Clips directory")
    parser.add_argument("--out_csv", default="dataset/suggestions.csv", help="Output CSV")
    parser.add_argument("--blur", type=int, default=5, help="Gaussian blur kernel")
    parser.add_argument("--threshold", type=float, default=0.5, help="Motion threshold")

    args = parser.parse_args()
    autosuggest_labels(args.clips_root, args.out_csv, args.blur, args.threshold)