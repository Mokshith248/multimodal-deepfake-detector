from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

DATA_DIR = Path("datasets/processed/video")
MODEL_DIR = Path("models")

IMAGE_SIZE = 224
SEQUENCE_LENGTH = 8
BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class VideoFrameDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        for label, class_name in enumerate(["fake", "real"]):

            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                continue

            files = sorted(
                [
                    file for file in class_dir.iterdir()
                    if file.suffix.lower()
                    in {".jpg", ".jpeg", ".png"}
                ]
            )

            for i in range(0, len(files), SEQUENCE_LENGTH):

                sequence = files[i:i + SEQUENCE_LENGTH]

                if len(sequence) == SEQUENCE_LENGTH:
                    self.samples.append(
                        (sequence, label)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        files, label = self.samples[index]

        frames = []

        for file in files:

            image = Image.open(file).convert("RGB")

            if self.transform:
                image = self.transform(image)

            frames.append(image)

        frames = torch.stack(frames)

        return frames, torch.tensor(
            label,
            dtype=torch.long
        )


class CNNLSTM(nn.Module):

    def __init__(self, num_classes=2):

        super().__init__()

        cnn = models.efficientnet_b0(
            weights=None
        )

        feature_size = cnn.classifier[1].in_features

        cnn.classifier = nn.Identity()

        self.cnn = cnn

        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )

        self.classifier = nn.Linear(
            256,
            num_classes
        )

    def forward(self, x):

        batch_size, sequence_length, channels, height, width = x.shape

        x = x.view(
            batch_size * sequence_length,
            channels,
            height,
            width
        )

        features = self.cnn(x)

        features = features.view(
            batch_size,
            sequence_length,
            -1
        )

        output, _ = self.lstm(features)

        last_output = output[:, -1, :]

        return self.classifier(last_output)


def main():

    print("VIDEO DEEPFAKE TRAINING")
    print(f"Device: {DEVICE}")

    if not DATA_DIR.exists():

        print(
            f"Dataset folder not found: {DATA_DIR}"
        )

        return

    transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = VideoFrameDataset(
        DATA_DIR,
        transform
    )

    print(
        f"Video sequences found: {len(dataset)}"
    )

    if len(dataset) < 2:

        print("\nNot enough video sequences for training.")

        print(
            "Add processed video frames to:"
        )

        print(
            "datasets/processed/video/real"
        )

        print(
            "datasets/processed/video/fake"
        )

        return

    train_size = int(
        0.8 * len(dataset)
    )

    val_size = len(dataset) - train_size

    if train_size == 0 or val_size == 0:

        print(
            "\nDataset is too small for "
            "train/validation split."
        )

        return

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = CNNLSTM()

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    best_accuracy = 0.0

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for frames, labels in train_loader:

            frames = frames.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(frames)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            predictions = outputs.argmax(
                dim=1
            )

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

        train_accuracy = (
            100 * correct / total
        )

        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for frames, labels in val_loader:

                frames = frames.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(frames)

                predictions = outputs.argmax(
                    dim=1
                )

                val_total += labels.size(0)

                val_correct += (
                    predictions == labels
                ).sum().item()

        val_accuracy = (
            100 * val_correct / val_total
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: "
            f"{running_loss / len(train_loader):.4f} | "
            f"Train Accuracy: "
            f"{train_accuracy:.2f}% | "
            f"Validation Accuracy: "
            f"{val_accuracy:.2f}%"
        )

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                MODEL_DIR /
                "video_deepfake_model.pth"
            )

            print("Best video model saved.")

    print(
        "\nVIDEO MODEL TRAINING COMPLETED"
    )

    print(
        f"Best validation accuracy: "
        f"{best_accuracy:.2f}%"
    )


if __name__ == "__main__":
    main()