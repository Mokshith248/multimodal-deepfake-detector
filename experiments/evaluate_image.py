from pathlib import Path
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

DATA_DIR = Path("datasets/processed/image")
MODEL_PATH = Path("models/image_deepfake_model.pth")

IMAGE_SIZE = 224
BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = ["fake", "real"]


class ImageDataset(Dataset):

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, label


def get_video_id(filename):

    name = Path(filename).stem

    match = re.match(
        r"(.+)_frame_\d+$",
        name
    )

    if match:
        return match.group(1)

    return name


def create_validation_split():

    videos = {}

    for label, class_name in enumerate(CLASS_NAMES):

        class_dir = DATA_DIR / class_name

        for image_path in class_dir.glob("*.jpg"):

            video_id = get_video_id(
                image_path.name
            )

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
        real_videos[i]
        for i in real_order
    ]

    fake_videos = [
        fake_videos[i]
        for i in fake_order
    ]

    real_split = int(
        0.8 * len(real_videos)
    )

    fake_split = int(
        0.8 * len(fake_videos)
    )

    validation_videos = (
        real_videos[real_split:]
        + fake_videos[fake_split:]
    )

    validation_samples = []

    for label, video_id in validation_videos:

        for image_path in videos[
            (label, video_id)
        ]:

            validation_samples.append(
                (image_path, label)
            )

    print("IMAGE VALIDATION SET")
    print("====================")

    print(
        f"Validation videos: "
        f"{len(validation_videos)}"
    )

    print(
        f"Validation images: "
        f"{len(validation_samples)}"
    )

    print(
        f"Validation real videos: "
        f"{len(real_videos) - real_split}"
    )

    print(
        f"Validation fake videos: "
        f"{len(fake_videos) - fake_split}"
    )

    return validation_samples


def create_model():

    model = models.efficientnet_b0(
        weights=None
    )

    input_features = (
        model.classifier[1].in_features
    )

    model.classifier[1] = nn.Linear(
        input_features,
        2
    )

    return model


def calculate_metrics(labels, predictions):

    tp_fake = 0
    fp_fake = 0
    fn_fake = 0

    tp_real = 0
    fp_real = 0
    fn_real = 0

    correct = 0

    for actual, predicted in zip(
        labels,
        predictions
    ):

        if actual == predicted:
            correct += 1

        if actual == 0 and predicted == 0:
            tp_fake += 1

        if actual == 1 and predicted == 0:
            fp_fake += 1

        if actual == 0 and predicted == 1:
            fn_fake += 1

        if actual == 1 and predicted == 1:
            tp_real += 1

        if actual == 0 and predicted == 1:
            fp_real += 1

        if actual == 1 and predicted == 0:
            fn_real += 1

    total = len(labels)

    accuracy = correct / total

    fake_precision = (
        tp_fake /
        (tp_fake + fp_fake)
        if (tp_fake + fp_fake) > 0
        else 0
    )

    fake_recall = (
        tp_fake /
        (tp_fake + fn_fake)
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
        tp_real /
        (tp_real + fp_real)
        if (tp_real + fp_real) > 0
        else 0
    )

    real_recall = (
        tp_real /
        (tp_real + fn_real)
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
        accuracy,
        fake_precision,
        fake_recall,
        fake_f1,
        real_precision,
        real_recall,
        real_f1,
        tp_fake,
        fp_fake,
        fn_fake,
        tp_real,
        fp_real,
        fn_real
    )


def main():

    print("\nIMAGE MODEL EVALUATION")
    print("======================")

    print(
        f"Device: {DEVICE}"
    )

    samples = create_validation_split()

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

    dataset = ImageDataset(
        samples,
        transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = create_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint
    )

    model.to(DEVICE)
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            ).cpu().tolist()

            all_predictions.extend(
                predictions
            )

            all_labels.extend(
                labels.tolist()
            )

    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )

    (
        accuracy,
        fake_precision,
        fake_recall,
        fake_f1,
        real_precision,
        real_recall,
        real_f1,
        tp_fake,
        fp_fake,
        fn_fake,
        tp_real,
        fp_real,
        fn_real
    ) = metrics

    print("\nFINAL IMAGE RESULTS")
    print("===================")

    print(
        f"Accuracy : {accuracy * 100:.2f}%"
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
        "[["
        f"{tp_fake:4d}"
        f" {fn_fake:4d}"
        "]]"
    )

    print(
        "[["
        f"{fp_fake:4d}"
        f" {tp_real:4d}"
        "]]"
    )

    print("\nMatrix format:")
    print("[[Fake→Fake, Fake→Real]")
    print(" [Real→Fake, Real→Real]]")


if __name__ == "__main__":
    main()