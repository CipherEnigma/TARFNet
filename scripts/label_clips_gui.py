import cv2
import os
import pandas as pd
import argparse


def play_clip(clip_path, fps=10):
    """Play a clip and return user input"""
    frames = sorted([f for f in os.listdir(clip_path) if f.endswith('.jpg')])

    print(f"\nPlaying {os.path.basename(clip_path)} ({len(frames)} frames)")
    print("Press: 1=movement, 0=no movement, s=skip, q=quit")

    # Play clip once
    for frame_file in frames:
        frame = cv2.imread(os.path.join(clip_path, frame_file))
        cv2.imshow('Clip Labeler', frame)

        key = cv2.waitKey(1000 // fps) & 0xFF
        if key in [ord('1'), ord('0'), ord('s'), ord('q')]:
            cv2.destroyAllWindows()
            return chr(key)

    # Wait for input after clip ends
    cv2.waitKey(0)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

    return chr(key) if key in [ord('1'), ord('0'), ord('s'), ord('q')] else 's'


def label_clips_gui(clips_root, out_csv, start_from=0):
    """GUI for labeling clips"""
    clips = sorted(os.listdir(clips_root))

    # Load existing labels if available
    labels = {}
    if os.path.exists(out_csv):
        df = pd.read_csv(out_csv)
        labels = dict(zip(df['clip'], df['label']))
        print(f"Loaded {len(labels)} existing labels")

    labeled_count = 0

    for i, clip_name in enumerate(clips[start_from:], start_from):
        clip_path = os.path.join(clips_root, clip_name)

        if not os.path.isdir(clip_path):
            continue

        if clip_name in labels:
            print(f"Skipping {clip_name} (already labeled)")
            continue

        print(f"\n[{i + 1}/{len(clips)}]")

        try:
            key = play_clip(clip_path)

            if key == '1':
                labels[clip_name] = 1
                labeled_count += 1
                print("Labeled: MOVEMENT")
            elif key == '0':
                labels[clip_name] = 0
                labeled_count += 1
                print("Labeled: NO MOVEMENT")
            elif key == 's':
                print("Skipped")
                continue
            elif key == 'q':
                print("Quitting...")
                break

        except Exception as e:
            print(f"Error with {clip_name}: {e}")
            continue

        # Save every 10 labels
        if labeled_count % 10 == 0:
            save_labels(labels, out_csv)

    # Final save
    save_labels(labels, out_csv)
    print(f"\nLabeling complete! Total labels: {len(labels)}")


def save_labels(labels, out_csv):
    """Save labels to CSV"""
    data = [{'clip': clip, 'label': label} for clip, label in labels.items()]
    df = pd.DataFrame(data)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(labels)} labels to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_root", default="dataset/clips", help="Clips directory")
    parser.add_argument("--out_csv", default="dataset/labels.csv", help="Output CSV")
    parser.add_argument("--start_from", type=int, default=0, help="Start from clip index")

    args = parser.parse_args()

    try:
        label_clips_gui(args.clips_root, args.out_csv, args.start_from)
    except Exception as e:
        print(f"GUI error (no display?): {e}")
        print("Use remote desktop or label clips manually by opening frames in image viewer")