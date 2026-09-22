from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from collections import defaultdict

DATA_DIR = Path("datasets/processed/video")
MODEL_PATH = Path("models/video_deepfake_model.pth")

IMAGE_SIZE = 224
SEQUENCE_LENGTH = 8
BATCH_SIZE = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class VideoFrameDataset(Dataset):

    def __init__(self, sequences, transform):
        self.sequences = sequences
        self.transform = transform

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):

        frame_paths, label = self.sequences[index]

        frames = []

        for path in frame_paths:

            image = Image.open(path).convert("RGB")

            image = self.transform(image)

            frames.append(image)

        frames = torch.stack(frames)

        return frames, label


class VideoModel(nn.Module):

    def __init__(self):

        super().__init__()

        cnn = models.efficientnet_b0(
            weights=None
        )

        input_features = (
            cnn.classifier[1].in_features
        )

        cnn.classifier = nn.Identity()

        self.cnn = cnn

        self.lstm = nn.LSTM(
            input_features,
            256,
            batch_first=True
        )

        self.classifier = nn.Linear(
            256,
            2
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
            -1
        )

        _, (hidden, _) = self.lstm(
            features
        )

        output = self.classifier(
            hidden[-1]
        )

        return output


def create_sequences():

    sequences = []

    for label, class_name in enumerate(
        ["fake", "real"]
    ):

        class_dir = DATA_DIR / class_name

        if not class_dir.exists():
            continue

        frames = sorted(
            class_dir.glob("*.jpg")
        )

        grouped = defaultdict(list)

        for frame in frames:

            name = frame.stem

            if "_frame_" in name:

                video_id = name.rsplit(
                    "_frame_",
                    1
                )[0]

            else:

                video_id = name

            grouped[video_id].append(
                frame
            )

        for video_id in sorted(grouped):

            video_frames = grouped[
                video_id
            ]

            video_frames = sorted(
                video_frames
            )

            if len(video_frames) < SEQUENCE_LENGTH:
                continue

            for start in range(
                0,
                len(video_frames)
                - SEQUENCE_LENGTH
                + 1,
                SEQUENCE_LENGTH
            ):

                sequence = video_frames[
                    start:
                    start + SEQUENCE_LENGTH
                ]

                sequences.append(
                    (sequence, label)
                )

    return sequences


def split_by_video(sequences):

    video_groups = defaultdict(list)

    for frames, label in sequences:

        first_frame = frames[0]

        name = first_frame.stem

        if "_frame_" in name:

            video_id = name.rsplit(
                "_frame_",
                1
            )[0]

        else:

            video_id = name

        video_groups[
            (label, video_id)
        ].append(
            (frames, label)
        )

    real_videos = sorted(
        [
            key for key in video_groups
            if key[0] == 1
        ]
    )

    fake_videos = sorted(
        [
            key for key in video_groups
            if key[0] == 0
        ]
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

    validation_keys = (
        real_videos[real_split:]
        + fake_videos[fake_split:]
    )

    validation = []

    for key in validation_keys:

        validation.extend(
            video_groups[key]
        )

    print("\nVIDEO VALIDATION SET")
    print("====================")

    print(
        f"Validation videos: "
        f"{len(validation_keys)}"
    )

    print(
        f"Validation sequences: "
        f"{len(validation)}"
    )

    print(
        f"Validation real videos: "
        f"{len(real_videos) - real_split}"
    )

    print(
        f"Validation fake videos: "
        f"{len(fake_videos) - fake_split}"
    )

    return validation


def calculate_metrics(
    labels,
    predictions
):

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
        tp_fake / (tp_fake + fp_fake)
        if tp_fake + fp_fake > 0
        else 0
    )

    fake_recall = (
        tp_fake / (tp_fake + fn_fake)
        if tp_fake + fn_fake > 0
        else 0
    )

    fake_f1 = (
        2 * fake_precision * fake_recall /
        (fake_precision + fake_recall)
        if fake_precision + fake_recall > 0
        else 0
    )

    real_precision = (
        tp_real / (tp_real + fp_real)
        if tp_real + fp_real > 0
        else 0
    )

    real_recall = (
        tp_real / (tp_real + fn_real)
        if tp_real + fn_real > 0
        else 0
    )

    real_f1 = (
        2 * real_precision * real_recall /
        (real_precision + real_recall)
        if real_precision + real_recall > 0
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
        fn_fake,
        fp_real,
        tp_real
    )


def main():

    print("\nVIDEO MODEL EVALUATION")
    print("======================")

    print(
        f"Device: {DEVICE}"
    )

    sequences = create_sequences()

    print(
        f"Total sequences: "
        f"{len(sequences)}"
    )

    validation = split_by_video(
        sequences
    )

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
        validation,
        transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = VideoModel()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint
    )

    model.to(DEVICE)
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for frames, labels in loader:

            frames = frames.to(DEVICE)

            outputs = model(frames)

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
        fn_fake,
        fp_real,
        tp_real
    ) = metrics

    print("\nFINAL VIDEO RESULTS")
    print("===================")

    print(
        f"Accuracy : "
        f"{accuracy * 100:.2f}%"
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
        f"[[{tp_fake:4d} {fn_fake:4d}]"
    )

    print(
        f" [{fp_real:4d} {tp_real:4d}]]"
    )

    print(
        "\nMatrix format:"
    )

    print(
        "[[Fake→Fake, Fake→Real]"
    )

    print(
        " [Real→Fake, Real→Real]]"
    )


if __name__ == "__main__":
    main()