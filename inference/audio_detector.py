import os
import torch
import torch.nn as nn
import librosa
import numpy as np
from PIL import Image
from torchvision import transforms

MODEL_PATH = "models/audio_deepfake_model.pth"


class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AudioCNN().to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )
    model.eval()
else:
    model = None


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


def create_mel_spectrogram(audio_path):
    audio, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    target_length = 16000 * 4

    if len(audio) < target_length:
        audio = np.pad(
            audio,
            (0, target_length - len(audio))
        )
    else:
        audio = audio[:target_length]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=128
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mel_db = (
        (mel_db - mel_db.min())
        / (mel_db.max() - mel_db.min() + 1e-8)
    )

    mel_image = (mel_db * 255).astype(np.uint8)

    image = Image.fromarray(mel_image)
    image = image.resize((224, 224))
    image = image.convert("RGB")

    return image


def detect_audio(audio_path):

    if model is None:
        return {
            "error": "Audio model has not been trained yet."
        }

    if not os.path.exists(audio_path):
        return {
            "error": "Audio file not found."
        }

    try:
        image = create_mel_spectrogram(audio_path)

        tensor = transform(image)
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )

        prediction = prediction.item()
        confidence = confidence.item() * 100

        if prediction == 0:
            label = "FAKE"
        else:
            label = "REAL"

        return {
            "prediction": label,
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        return {
            "error": str(e)
        }


if __name__ == "__main__":

    print("AUDIO DEEPFAKE DETECTOR")

    if model is None:
        print("Audio model not found.")
        print("Train the model first using:")
        print("python training\\train_audio.py")
    else:
        print("Audio model loaded successfully.")
        print("Device:", device)

        audio_path = input("Enter audio path: ").strip()

        result = detect_audio(audio_path)

        if "error" in result:
            print("\nError:", result["error"])
        else:
            print(f"\nPrediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.2f}%")