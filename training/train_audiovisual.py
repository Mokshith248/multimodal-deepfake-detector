import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import numpy as np


DATA_DIR = "datasets/processed/audiovisual"
MODEL_PATH = "models/audiovisual_deepfake_model.pth"

BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 1e-4

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class AudioVisualDataset(Dataset):

    def __init__(self, split):

        self.files = []

        split_dir = os.path.join(
            DATA_DIR,
            split
        )

        for filename in os.listdir(split_dir):

            if filename.endswith(".pt"):

                self.files.append(
                    os.path.join(
                        split_dir,
                        filename
                    )
                )

        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        sample = torch.load(
            self.files[index],
            map_location="cpu"
        )

        frames = sample["frames"].float() / 255.0

        mel = sample["mel"].float()

        label = torch.tensor(
        sample["label"],
        dtype=torch.long
        )

        frames = frames.permute(
            0, 3, 1, 2
        )

        frames = (
            frames - torch.tensor(
                [0.485, 0.456, 0.406]
            ).view(1, 3, 1, 1)
        ) / torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(1, 3, 1, 1)

        mel = mel.unsqueeze(0)

        mel = mel.repeat(
            3, 1, 1
        )

        mel = (
            mel - 0.5
        ) / 0.5

        return frames, mel, label


class VisualEncoder(nn.Module):

    def __init__(self):

        super().__init__()
        model = models.efficientnet_b0(
            weights=None,
            num_classes=2
        )

        checkpoint = torch.load(
            "models/image_deepfake_model.pth",
            map_location="cpu"
        )

        model.load_state_dict(
            checkpoint
        )

        model.classifier = nn.Identity()

        self.cnn = model

        for parameter in self.cnn.parameters():
            parameter.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):

        batch_size, sequence_length, channels, height, width = x.shape

        x = x.reshape(
            batch_size * sequence_length,
            channels,
            height,
            width
        )

        with torch.no_grad():

            features = self.cnn(x)

        features = features.reshape(
            batch_size,
            sequence_length,
            1280
        )

        output, _ = self.lstm(
            features
        )

        return output[:, -1, :]


class AudioEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                128,
                256,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

        checkpoint = torch.load(
            "models/audio_deepfake_model.pth",
            map_location="cpu"
        )

        full_state = checkpoint

        feature_state = {
            key.replace(
                "features.",
                ""
            ): value

            for key, value in full_state.items()

            if key.startswith("features.")
        }

        self.features.load_state_dict(
            feature_state
        )

        for parameter in self.features.parameters():
            parameter.requires_grad = False

    def forward(self, x):

        with torch.no_grad():

            x = self.features(x)

        x = x.view(
            x.size(0),
            -1
        )

        return x


class AudioVisualFusion(nn.Module):

    def __init__(self):

        super().__init__()

        self.visual_encoder = VisualEncoder()

        self.audio_encoder = AudioEncoder()

        self.classifier = nn.Sequential(

            nn.Linear(
                256 + 256,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                256,
                2
            )
        )

    def forward(
        self,
        video,
        audio
    ):

        visual_features = self.visual_encoder(
            video
        )

        audio_features = self.audio_encoder(
            audio
        )

        fused_features = torch.cat(
            [
                visual_features,
                audio_features
            ],
            dim=1
        )

        return self.classifier(
            fused_features
        )


def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for video, audio, labels in loader:

            video = video.to(device)

            audio = audio.to(device)

            labels = labels.to(device)

            outputs = model(
                video,
                audio
            )

            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = (
        correct / total
        if total > 0
        else 0
    )

    average_loss = (
        total_loss / len(loader)
        if len(loader) > 0
        else 0
    )

    return average_loss, accuracy


print("AUDIO-VISUAL FUSION TRAINING")
print("----------------------------")
print("Device:", device)

train_dataset = AudioVisualDataset("train")
dev_dataset = AudioVisualDataset("dev")
test_dataset = AudioVisualDataset("test")

print("Training samples:", len(train_dataset))
print("Development samples:", len(dev_dataset))
print("Test samples:", len(test_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

model = AudioVisualFusion().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE
)

best_accuracy = 0.0

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for video, audio, labels in train_loader:

        video = video.to(device)

        audio = audio.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(
            video,
            audio
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    train_loss = (
        total_loss / len(train_loader)
    )

    dev_loss, dev_accuracy = evaluate(
        model,
        dev_loader,
        criterion
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Dev Loss: {dev_loss:.4f} "
        f"Dev Accuracy: {dev_accuracy:.4f}"
    )

    if dev_accuracy > best_accuracy:

        best_accuracy = dev_accuracy

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print("Best model saved.")


print()
print("Loading best model...")

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

test_loss, test_accuracy = evaluate(
    model,
    test_loader,
    criterion
)

print()
print("AUDIO-VISUAL FUSION TRAINING COMPLETED")
print("---------------------------------------")
print(
    "Best Development Accuracy:",
    round(best_accuracy, 4)
)
print(
    "Final Test Accuracy:",
    round(test_accuracy, 4)
)
print(
    "Model saved:",
    MODEL_PATH
)