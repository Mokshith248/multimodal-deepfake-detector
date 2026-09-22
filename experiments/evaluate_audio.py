from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from PIL import Image

DATA_DIR = Path("datasets/processed/audio")
MODEL_PATH = Path("models/audio_deepfake_model.pth")

IMAGE_SIZE = 224

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

        return image, label


class AudioCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                128,
                256,
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


def calculate_metrics(
    predictions,
    labels
):

    tp_fake = 0
    fn_fake = 0
    fp_fake = 0

    tp_real = 0
    fn_real = 0
    fp_real = 0

    for prediction, label in zip(
        predictions,
        labels
    ):

        if label == 0:

            if prediction == 0:
                tp_fake += 1
            else:
                fn_fake += 1

        elif label == 1:

            if prediction == 1:
                tp_real += 1
            else:
                fn_real += 1

    fp_fake = fn_real
    fp_real = fn_fake

    fake_precision = (
        tp_fake / (tp_fake + fp_fake)
        if (tp_fake + fp_fake) > 0
        else 0
    )

    fake_recall = (
        tp_fake / (tp_fake + fn_fake)
        if (tp_fake + fn_fake) > 0
        else 0
    )

    fake_f1 = (
        2 * fake_precision * fake_recall /
        (fake_precision + fake_recall)
        if (fake_precision + fake_recall) > 0
        else 0
    )

    real_precision = (
        tp_real / (tp_real + fp_real)
        if (tp_real + fp_real) > 0
        else 0
    )

    real_recall = (
        tp_real / (tp_real + fn_real)
        if (tp_real + fn_real) > 0
        else 0
    )

    real_f1 = (
        2 * real_precision * real_recall /
        (real_precision + real_recall)
        if (real_precision + real_recall) > 0
        else 0
    )

    return (
        fake_precision,
        fake_recall,
        fake_f1,
        real_precision,
        real_recall,
        real_f1
    )


def main():

    print("AUDIO MODEL EVALUATION")
    print("======================")

    print(f"Device: {DEVICE}")

    if not DATA_DIR.exists():

        print(
            f"\nDataset not found: {DATA_DIR}"
        )

        return

    if not MODEL_PATH.exists():

        print(
            f"\nModel not found: {MODEL_PATH}"
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
        f"Total dataset samples: "
        f"{len(dataset)}"
    )

    train_size = int(
        0.8 * len(dataset)
    )

    val_size = len(dataset) - train_size

    _, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    model = AudioCNN()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint
    )

    model = model.to(DEVICE)

    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():

        for index in range(
            len(val_dataset)
        ):

            image, label = val_dataset[index]

            image = image.unsqueeze(0)

            image = image.to(DEVICE)

            output = model(image)

            prediction = output.argmax(
                dim=1
            ).item()

            predictions.append(
                prediction
            )

            labels.append(
                label
            )

    print(
        f"\nPredictions generated: "
        f"{len(predictions)}"
    )

    correct = sum(
        prediction == label
        for prediction, label
        in zip(predictions, labels)
    )

    total = len(labels)

    accuracy = (
        100 * correct / total
        if total > 0
        else 0
    )

    (
        fake_precision,
        fake_recall,
        fake_f1,
        real_precision,
        real_recall,
        real_f1
    ) = calculate_metrics(
        predictions,
        labels
    )

    fake_fake = sum(
        1
        for prediction, label
        in zip(predictions, labels)
        if label == 0 and prediction == 0
    )

    fake_real = sum(
        1
        for prediction, label
        in zip(predictions, labels)
        if label == 0 and prediction == 1
    )

    real_fake = sum(
        1
        for prediction, label
        in zip(predictions, labels)
        if label == 1 and prediction == 0
    )

    real_real = sum(
        1
        for prediction, label
        in zip(predictions, labels)
        if label == 1 and prediction == 1
    )

    print("\nFINAL AUDIO RESULTS")
    print("===================")

    print(
        f"Accuracy : {accuracy:.2f}%"
    )

    print("\nFAKE CLASS")
    print("----------")

    print(
        f"Precision: "
        f"{fake_precision * 100:.2f}%"
    )

    print(
        f"Recall   : "
        f"{fake_recall * 100:.2f}%"
    )

    print(
        f"F1 Score : "
        f"{fake_f1 * 100:.2f}%"
    )

    print("\nREAL CLASS")
    print("----------")

    print(
        f"Precision: "
        f"{real_precision * 100:.2f}%"
    )

    print(
        f"Recall   : "
        f"{real_recall * 100:.2f}%"
    )

    print(
        f"F1 Score : "
        f"{real_f1 * 100:.2f}%"
    )

    print("\nConfusion Matrix")
    print("----------------")

    print(
        f"[[{fake_fake:4d} {fake_real:4d}]"
    )

    print(
        f" [{real_fake:4d} {real_real:4d}]]"
    )

    print("\nMatrix format:")
    print(
        "[[Fake→Fake, Fake→Real]"
    )
    print(
        " [Real→Fake, Real→Real]]"
    )

    print(
        f"\nTotal evaluated samples: "
        f"{total}"
    )

    print(
        f"Correct predictions: "
        f"{correct}"
    )

    print(
        f"Incorrect predictions: "
        f"{total - correct}"
    )


if __name__ == "__main__":
    main()