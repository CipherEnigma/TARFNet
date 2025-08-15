import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for video sequences"""

    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        attn_weights = self.attention(x)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)


        weighted_x = x * attn_weights
        attended_features = torch.sum(weighted_x, dim=1)  # (batch, hidden_dim)

        return attended_features, attn_weights


class ResidualBlock(nn.Module):
    """Residual block for feature learning"""

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # for skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)

        return out


class TARFNet(nn.Module):
    """Temporal Attention Residual Fusion Network"""

    def __init__(self, num_frames=8, num_classes=2):
        super(TARFNet, self).__init__()
        self.num_frames = num_frames


        self.feature_extractor = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),


            ResidualBlock(64, 64),
            ResidualBlock(64, 128),
            nn.MaxPool2d(2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256),
            nn.MaxPool2d(2),
            ResidualBlock(256, 256),
            ResidualBlock(256, 512),


            nn.AdaptiveAvgPool2d((1, 1))
        )


        self.temporal_attention = TemporalAttention(512)


        self.fusion_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, num_frames, channels, height, width)
        batch_size = x.size(0)

        #
        frame_features = []
        for i in range(self.num_frames):
            frame = x[:, i]  # (batch_size, channels, height, width)
            features = self.feature_extractor(frame)
            features = features.view(batch_size, -1)  # Flatten
            frame_features.append(features)


        temporal_features = torch.stack(frame_features, dim=1)


        attended_features, attention_weights = self.temporal_attention(temporal_features)


        output = self.fusion_layer(attended_features)

        return output, attention_weights


def create_tarfnet(num_frames=8, num_classes=2):

    return TARFNet(num_frames=num_frames, num_classes=num_classes)



def model_summary(model, input_shape):
    """Print model summary"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"TARFNet Model Summary:")
    print(f"Input shape: {input_shape}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")



if __name__ == "__main__":

    model = create_tarfnet(num_frames=8, num_classes=2)


    batch_size = 4
    num_frames = 8
    channels = 3
    height, width = 224, 224

    dummy_input = torch.randn(batch_size, num_frames, channels, height, width)

    # Forward pass
    output, attention_weights = model(dummy_input)

    print("Model test successful!")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")


    model_summary(model, (num_frames, channels, height, width))