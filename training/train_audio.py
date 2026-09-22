from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

DATA_DIR = Path("datasets/processed/audio")
MODEL_DIR = Path("models")

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class AudioDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        for label, class_name in enumerate(["fake", "real"]):

            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                continue

            files = [
                file
                for file in class_dir.iterdir()
                if file.is_file()
                and file.suffix.lower() == ".png"
            ]

            for file in files:
                self.samples.append(
                    (file, label)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        file, label = self.samples[index]

        image = Image.open(file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(
            label,
            dtype=torch.long
        )


class AudioCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3, 32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                128, 256,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.3),

            nn.Linear(
                256,
                2
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


def main():

    print("AUDIO DEEPFAKE TRAINING")
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
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    dataset = AudioDataset(
        DATA_DIR,
        transform
    )

    print(
        f"Audio samples found: {len(dataset)}"
    )

    if len(dataset) < 2:

        print(
            "\nNot enough audio samples for training."
        )

        print(
            "Add processed audio files to:"
        )

        print(
            "datasets/processed/audio/real"
        )

        print(
            "datasets/processed/audio/fake"
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

    train_dataset, val_dataset = random_split(
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

    model = AudioCNN()

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

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

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

            for images, labels in val_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

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
                "audio_deepfake_model.pth"
            )

            print(
                "Best audio model saved."
            )

    print(
        "\nAUDIO MODEL TRAINING COMPLETED"
    )

    print(
        f"Best validation accuracy: "
        f"{best_accuracy:.2f}%"
    )


if __name__ == "__main__":
    main()