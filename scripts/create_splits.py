import pandas as pd
import argparse
from sklearn.model_selection import train_test_split


def create_splits(labels_csv, out_csv, val_frac=0.15, test_frac=0.15):

    df = pd.read_csv(labels_csv)


    df = df.dropna(subset=['label'])

    print(f"Total labeled clips: {len(df)}")
    print(f"Movement clips: {sum(df['label'] == 1)}")
    print(f"No movement clips: {sum(df['label'] == 0)}")


    train_clips, temp_clips, train_labels, temp_labels = train_test_split(
        df['clip'], df['label'],
        test_size=(val_frac + test_frac),
        stratify=df['label'],
        random_state=42
    )


    val_clips, test_clips, val_labels, test_labels = train_test_split(
        temp_clips, temp_labels,
        test_size=(test_frac / (val_frac + test_frac)),
        stratify=temp_labels,
        random_state=42
    )


    splits_data = []

    for clip, label in zip(train_clips, train_labels):
        splits_data.append({'clip': clip, 'split': 'train', 'label': label})

    for clip, label in zip(val_clips, val_labels):
        splits_data.append({'clip': clip, 'split': 'val', 'label': label})

    for clip, label in zip(test_clips, test_labels):
        splits_data.append({'clip': clip, 'split': 'test', 'label': label})

    splits_df = pd.DataFrame(splits_data)
    splits_df.to_csv(out_csv, index=False)

    print(f"\nSplit summary:")
    print(splits_df.groupby(['split', 'label']).size().unstack(fill_value=0))
    print(f"\nSaved splits to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_csv", default="dataset/labels.csv", help="Input labels CSV")
    parser.add_argument("--out_csv", default="dataset/splits.csv", help="Output splits CSV")
    parser.add_argument("--val_frac", type=float, default=0.15, help="Validation fraction")
    parser.add_argument("--test_frac", type=float, default=0.15, help="Test fraction")

    args = parser.parse_args()
    create_splits(args.labels_csv, args.out_csv, args.val_frac, args.test_frac)