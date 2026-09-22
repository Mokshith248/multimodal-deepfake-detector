import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_DIR = "datasets/processed/audiovisual/test"
MODEL_PATH = "models/audiovisual_deepfake_model.pth"
IMAGE_MODEL_PATH = "models/image_deepfake_model.pth"


class AudioVisualDataset(Dataset):

    def __init__(self, directory):

        self.files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".pt")
        ]

        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        data = torch.load(
            self.files[index],
            map_location="cpu"
        )

        frames = data["frames"]
        mel = data["mel"]
        label = int(data["label"])

        if frames.dtype == torch.uint8:

            frames = frames.float() / 255.0

        if frames.ndim == 4:

            frames = frames.permute(
                0, 3, 1, 2
            )

        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(1, 3, 1, 1)

        frames = (frames - mean) / std

        if mel.ndim == 2:

            mel = mel.unsqueeze(0)

        if mel.shape[0] == 1:

            mel = mel.repeat(3, 1, 1)

        mel = (mel - 0.5) / 0.5

        return frames, mel, label


class VisualEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        model = models.efficientnet_b0(
            weights=None,
            num_classes=2
        )

        checkpoint = torch.load(
            IMAGE_MODEL_PATH,
            map_location="cpu"
        )

        model.load_state_dict(checkpoint)

        model.classifier = nn.Identity()

        self.cnn = model

        self.lstm = nn.LSTM(
            1280,
            256,
            batch_first=True
        )

    def forward(self, x):

        batch, time, channels, height, width = x.shape

        x = x.view(
            batch * time,
            channels,
            height,
            width
        )

        features = self.cnn(x)

        features = features.view(
            batch,
            time,
            1280
        )

        _, (hidden, _) = self.lstm(features)

        return hidden[-1]


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

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.classifier = nn.Sequential(

            nn.Dropout(0.3),

            nn.Identity(),

            nn.Linear(
                256,
                2
            )
        )

    def forward(self, x):

        x = self.features(x)

        return x.flatten(1)


class AudioVisualModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.visual_encoder = VisualEncoder()

        self.audio_encoder = AudioEncoder()

        self.classifier = nn.Sequential(

            nn.Linear(
                512,
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
        frames,
        mel
    ):

        visual_features = self.visual_encoder(
            frames
        )

        audio_features = self.audio_encoder(
            mel
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


print()
print("AUDIO-VISUAL MODEL EVALUATION")
print("==============================")
print("Device:", DEVICE)


dataset = AudioVisualDataset(
    TEST_DIR
)

print("Test samples:", len(dataset))


loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False
)


model = AudioVisualModel()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint
)

model.to(DEVICE)

model.eval()

print()
print("Model loaded successfully.")
print("Starting evaluation...")
print()


y_true = []
y_pred = []


with torch.no_grad():

    for batch_number, (
        frames,
        mel,
        labels
    ) in enumerate(loader):

        frames = frames.to(DEVICE)

        mel = mel.to(DEVICE)

        outputs = model(
            frames,
            mel
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        y_true.extend(
            labels.tolist()
        )

        y_pred.extend(
            predictions.cpu().tolist()
        )

        if (batch_number + 1) % 20 == 0:

            print(
                f"Processed "
                f"{min((batch_number + 1) * 2, len(dataset))}"
                f"/{len(dataset)} samples"
            )


accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_true,
    y_pred
)


print()
print("================================")
print("FINAL AUDIO-VISUAL RESULTS")
print("================================")

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1 Score : {f1 * 100:.2f}%"
)

print()
print("Confusion Matrix:")
print(cm)

print()
print("Classification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "FAKE",
            "REAL"
        ],
        zero_division=0
    )
)

print()
print("Evaluation completed successfully.")