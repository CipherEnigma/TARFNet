import cv2
import os
import argparse
from tqdm import tqdm


def extract_frames(video_path, out_dir, target_fps=10):
    """Extract frames from video at target FPS"""
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame skip to achieve target FPS
    skip_frames = max(1, int(original_fps / target_fps))

    print(f"Original FPS: {original_fps}, Target FPS: {target_fps}")
    print(f"Will extract every {skip_frames} frames")

    frame_count = 0
    saved_count = 0

    pbar = tqdm(total=total_frames // skip_frames, desc="Extracting frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames == 0:
            filename = f"{saved_count:06d}.jpg"
            cv2.imwrite(os.path.join(out_dir, filename), frame)
            saved_count += 1
            pbar.update(1)

        frame_count += 1

    cap.release()
    pbar.close()
    print(f"Extracted {saved_count} frames to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out_dir", default="dataset/frames", help="Output directory")
    parser.add_argument("--target_fps", type=int , default=10, help="Target FPS")

    args = parser.parse_args()
    extract_frames(args.video, args.out_dir, args.target_fps)