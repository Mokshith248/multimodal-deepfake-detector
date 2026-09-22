from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from facenet_pytorch import MTCNN

MODEL_PATH = Path("models/image_deepfake_model.pth")

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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


def create_model():
    model = models.efficientnet_b0(
        weights=None
    )

    input_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        input_features,
        2
    )

    return model


def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Train the image model first."
        )

    model = create_model()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)
    model.eval()

    return model


def predict(image_path):

    model = load_model()

    image = Image.open(image_path).convert("RGB")

    face = face_detector(image)

    if face is None:
        raise ValueError(
            "No face detected in the image."
        )

    face = face.permute(1, 2, 0).byte().numpy()

    face = Image.fromarray(face)

    input_tensor = transform(face)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(DEVICE)

    with torch.no_grad():

        output = model(input_tensor)

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
def detect_image(image_path):
    return predict(image_path)


if __name__ == "__main__":

    print("IMAGE DEEPFAKE DETECTOR")

    image_path = input(
        "Enter image path: "
    ).strip()

    try:

        result = predict(image_path)

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