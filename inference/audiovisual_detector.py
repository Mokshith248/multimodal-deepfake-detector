import os
import subprocess
import tempfile

import cv2
import librosa
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from facenet_pytorch import MTCNN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FFMPEG = r"C:\Users\Mokshith\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe"

MODEL_PATH = "models/audiovisual_deepfake_model.pth"


class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        model = models.efficientnet_b0(
            weights=None,
            num_classes=2
        )

        checkpoint = torch.load(
            "models/image_deepfake_model.pth",
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

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Identity(),
            nn.Linear(256, 2)
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
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, frames, mel):
        visual_features = self.visual_encoder(frames)
        audio_features = self.audio_encoder(mel)

        fused_features = torch.cat(
            [visual_features, audio_features],
            dim=1
        )

        return self.classifier(fused_features)


mtcnn = MTCNN(
    image_size=224,
    margin=0,
    keep_all=False,
    device=DEVICE
)


def extract_video_frames(video_path, num_frames=8):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()
        raise ValueError("Could not read video.")

    indexes = np.linspace(
        0,
        total_frames - 1,
        num_frames
    ).astype(int)

    frames = []

    for index in indexes:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index)
        )

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        face = mtcnn(frame)

        if face is None:

            face = cv2.resize(
                frame,
                (224, 224)
            )

            face = torch.tensor(
                face,
                dtype=torch.float32
            ).permute(2, 0, 1) / 255.0

        else:

            face = face / 255.0

        frames.append(face)

    cap.release()

    if not frames:
        raise ValueError(
            "No usable video frames found."
        )

    while len(frames) < num_frames:
        frames.append(
            frames[-1].clone()
        )

    frames = frames[:num_frames]

    frames = torch.stack(frames)

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(1, 3, 1, 1)

    frames = (
        frames - mean
    ) / std

    return frames.unsqueeze(0)


def extract_audio_mel(video_path):

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ) as temp:

        wav_path = temp.name

    try:

        command = [
            FFMPEG,
            "-y",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            wav_path
        ]

        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        audio, _ = librosa.load(
            wav_path,
            sr=16000,
            mono=True
        )

    finally:

        if os.path.exists(wav_path):
            os.remove(wav_path)

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
        sr=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=128
    )

    mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mel = cv2.resize(
        mel,
        (224, 224)
    )

    mel = (
        mel - mel.min()
    ) / (
        mel.max() - mel.min() + 1e-8
    )

    mel = torch.tensor(
        mel,
        dtype=torch.float32
    )

    mel = mel.unsqueeze(0).repeat(
        3, 1, 1
    )

    mel = (
        mel - 0.5
    ) / 0.5

    return mel.unsqueeze(0)


_model = None


def load_model():

    global _model

    if _model is None:

        _model = AudioVisualModel()

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )

        _model.load_state_dict(
            checkpoint
        )

        _model.to(DEVICE)
        _model.eval()

    return _model


def detect_audiovisual(video_path):

    model = load_model()

    frames = extract_video_frames(
        video_path
    )

    mel = extract_audio_mel(
        video_path
    )

    frames = frames.to(DEVICE)
    mel = mel.to(DEVICE)

    with torch.no_grad():

        output = model(
            frames,
            mel
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            prediction
        ].item()

    label = (
        "FAKE"
        if prediction == 0
        else "REAL"
    )

    return {
        "prediction": label,
        "confidence": confidence*100
    }