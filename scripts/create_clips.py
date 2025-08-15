import os
import shutil
import argparse
from tqdm import tqdm


def create_clips(frame_dir, clip_dir, clip_len=16, stride=4):

    os.makedirs(clip_dir, exist_ok=True)


    frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    print(f"Found {len(frames)} frames")

    clip_count = 0


    for start_idx in tqdm(range(0, len(frames) - clip_len + 1, stride), desc="Creating clips"):
        clip_name = f"clip_{clip_count:06d}"
        clip_path = os.path.join(clip_dir, clip_name)
        os.makedirs(clip_path, exist_ok=True)


        for i in range(clip_len):
            src_frame = os.path.join(frame_dir, frames[start_idx + i])
            dst_frame = os.path.join(clip_path, f"{i:02d}.jpg")
            shutil.copy2(src_frame, dst_frame)

        clip_count += 1

    print(f"Created {clip_count} clips in {clip_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_dir", default="dataset/frames", help="Frame directory")
    parser.add_argument("--clip_dir", default="dataset/clips", help="Clip output directory")
    parser.add_argument("--clip_len", type=int, default=16, help="Frames per clip")
    parser.add_argument("--stride", type=int, default=4, help="Stride between clips")

    args = parser.parse_args()
    create_clips(args.frame_dir, args.clip_dir, args.clip_len, args.stride)