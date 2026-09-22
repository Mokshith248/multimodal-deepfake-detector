from pathlib import Path
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

DATA_DIR = Path("datasets/processed/image")
MODEL_DIR = Path("models")

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.0001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["fake", "real"]


class VideoLevelImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_video_id(filename):
    name = Path(filename).stem

    match = re.match(r"(.+)_frame_\d+$", name)

    if match:
        return match.group(1)

    return name


def create_video_level_split():
    all_samples = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = DATA_DIR / class_name

        if not class_dir.exists():
            continue

        for image_path in class_dir.glob("*.jpg"):
            video_id = get_video_id(image_path.name)

            all_samples.append(
                (video_id, image_path, class_index)
            )

    videos = {}

    for video_id, image_path, label in all_samples:
        key = (label, video_id)

        if key not in videos:
            videos[key] = []

        videos[key].append(image_path)

    real_videos = sorted(
        [key for key in videos if key[0] == 1],
        key=lambda x: x[1]
    )

    fake_videos = sorted(
        [key for key in videos if key[0] == 0],
        key=lambda x: x[1]
    )

    generator = torch.Generator().manual_seed(42)

    real_order = torch.randperm(
        len(real_videos),
        generator=generator
    ).tolist()

    fake_order = torch.randperm(
        len(fake_videos),
        generator=generator
    ).tolist()

    real_videos = [
        real_videos[i] for i in real_order
    ]

    fake_videos = [
        fake_videos[i] for i in fake_order
    ]

    real_split = int(0.8 * len(real_videos))
    fake_split = int(0.8 * len(fake_videos))

    train_videos = (
        real_videos[:real_split] +
        fake_videos[:fake_split]
    )

    val_videos = (
        real_videos[real_split:] +
        fake_videos[fake_split:]
    )

    train_samples = []
    val_samples = []

    for label, video_id in train_videos:
        for image_path in videos[(label, video_id)]:
            train_samples.append(
                (image_path, label)
            )

    for label, video_id in val_videos:
        for image_path in videos[(label, video_id)]:
            val_samples.append(
                (image_path, label)
            )

    print("\nVIDEO-LEVEL DATASET SPLIT")
    print("--------------------------")

    print(f"Total videos: {len(real_videos) + len(fake_videos)}")
    print(f"Training videos: {len(train_videos)}")
    print(f"Validation videos: {len(val_videos)}")

    print(f"\nTraining images: {len(train_samples)}")
    print(f"Validation images: {len(val_samples)}")

    print(
        f"\nTraining real videos: {real_split}"
    )
    print(
        f"Training fake videos: {fake_split}"
    )

    print(
        f"Validation real videos: "
        f"{len(real_videos) - real_split}"
    )

    print(
        f"Validation fake videos: "
        f"{len(fake_videos) - fake_split}"
    )

    return train_samples, val_samples


def create_model():
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )

    input_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        input_features,
        2
    )

    return model


def main():

    print("IMAGE DEEPFAKE TRAINING")
    print("=======================")
    print(f"Device: {DEVICE}")

    if not DATA_DIR.exists():
        print(f"Dataset folder not found: {DATA_DIR}")
        return

    train_samples, val_samples = create_video_level_split()

    if len(train_samples) == 0 or len(val_samples) == 0:
        print("\nDataset is too small.")
        return

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = VideoLevelImageDataset(
        train_samples,
        transform=train_transform
    )

    val_dataset = VideoLevelImageDataset(
        val_samples,
        transform=val_transform
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

    print("\nLoading EfficientNet-B0...")

    model = create_model()
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
                "image_deepfake_model.pth"
            )

            print("Best model saved.")

    print(
        "\nIMAGE MODEL TRAINING COMPLETED"
    )

    print(
        f"Best validation accuracy: "
        f"{best_accuracy:.2f}%"
    )


if __name__ == "__main__":
    main()