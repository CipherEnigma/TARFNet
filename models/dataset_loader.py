import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


class SleepMovementDataset(Dataset):


    def __init__(self, clips_dir, splits_csv, split='train', transform=None, num_frames=8):
        self.clips_dir = clips_dir
        self.num_frames = num_frames
        self.transform = transform

        # Load splits
        splits_df = pd.read_csv(splits_csv)
        self.data = splits_df[splits_df['split'] == split].reset_index(drop=True)

        print(f"{split} set: {len(self.data)} clips")
        print(f"Movement clips: {sum(self.data['label'] == 1)}")
        print(f"No movement clips: {sum(self.data['label'] == 0)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        clip_name = row['clip']
        label = row['label']

        # Load clip frames
        clip_path = os.path.join(self.clips_dir, clip_name)
        frames = []

        for i in range(self.num_frames):
            frame_path = os.path.join(clip_path, f"{i:02d}.jpg")

            if os.path.exists(frame_path):
                image = Image.open(frame_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                frames.append(image)
            else:
                # If frame doesn't exist, duplicate last frame
                if frames:
                    frames.append(frames[-1])
                else:

                    dummy_frame = torch.zeros(3, 224, 224)
                    frames.append(dummy_frame)


        video_tensor = torch.stack(frames)

        return video_tensor, torch.tensor(label, dtype=torch.long)


def get_transforms():
    """Get data transforms for training and validation"""

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def create_dataloaders(clips_dir, splits_csv, batch_size=8, num_workers=2, num_frames=8):
    """Create train, validation, and test dataloaders"""

    train_transform, val_transform = get_transforms()

    # Create datasets
    train_dataset = SleepMovementDataset(
        clips_dir, splits_csv, split='train',
        transform=train_transform, num_frames=num_frames
    )

    val_dataset = SleepMovementDataset(
        clips_dir, splits_csv, split='val',
        transform=val_transform, num_frames=num_frames
    )

    test_dataset = SleepMovementDataset(
        clips_dir, splits_csv, split='test',
        transform=val_transform, num_frames=num_frames
    )


    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader



if __name__ == "__main__":

    clips_dir = "dataset/clips"
    splits_csv = "dataset/splits.csv"

    if os.path.exists(clips_dir) and os.path.exists(splits_csv):
        train_loader, val_loader, test_loader = create_dataloaders(
            clips_dir, splits_csv, batch_size=4
        )

        print("Dataset loading test:")


        for videos, labels in train_loader:
            print(f"Video batch shape: {videos.shape}")
            print(f"Labels batch shape: {labels.shape}")
            print(f"Labels: {labels}")
            break
    else:
        print("Dataset files not found. Run data preparation first.")