from pathlib import Path
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from facenet_pytorch import MTCNN

MODEL_PATH = Path("models/video_deepfake_model.pth")
IMAGE_SIZE = 224
SEQUENCE_LENGTH = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["fake", "real"]

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

face_detector = MTCNN(
    image_size=224,
    margin=20,
    keep_all=False,
    post_process=False,
    device="cpu"
)


class VideoModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = models.efficientnet_b0(weights=None)
        feature_size = self.cnn.classifier[1].in_features
        self.cnn.classifier = nn.Identity()

        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )

        self.classifier = nn.Linear(256, 2)

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

        output = output[:, -1, :]

        return self.classifier(output)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Train the video model first."
        )

    model = VideoModel()

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()

    return model


def extract_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Unable to open video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError("Video contains no frames.")

    frame_indices = torch.linspace(
        0,
        total_frames - 1,
        SEQUENCE_LENGTH
    ).long().tolist()

    frames = []

    for index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(frame)

        face = face_detector(image)

        if face is not None:
            face = face.permute(1, 2, 0).byte().numpy()
            face = Image.fromarray(face)

            tensor = transform(face)
            frames.append(tensor)

    cap.release()

    if len(frames) == 0:
        raise ValueError("No face detected in the video.")

    while len(frames) < SEQUENCE_LENGTH:
        frames.append(frames[-1].clone())

    return torch.stack(frames[:SEQUENCE_LENGTH])


def predict(video_path):
    model = load_model()

    frames = extract_frames(video_path)

    frames = frames.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(frames)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    label = CLASS_NAMES[predicted.item()]

    return {
        "prediction": label,
        "confidence": float(confidence.item() * 100)
    }


def detect_video(video_path):
    return predict(video_path)


if __name__ == "__main__":
    print("VIDEO DEEPFAKE DETECTOR")

    try:
        model = load_model()

        print("Video model loaded successfully.")
        print(f"Device: {DEVICE}")

        video_path = input(
            "\nEnter video path: "
        ).strip()

        result = predict(video_path)

        print(
            f"\nPrediction: "
            f"{result['prediction'].upper()}"
        )

        print(
            f"Confidence: "
            f"{result['confidence']:.2f}%"
        )

    except Exception as e:
        print(f"\nError: {e}")