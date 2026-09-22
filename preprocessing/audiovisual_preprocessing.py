import os
import traceback
import cv2
import subprocess
import tempfile
import json
import torch
import librosa
import numpy as np
import pandas as pd
from PIL import Image
from facenet_pytorch import MTCNN


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "datasets", "av_manifest.csv")

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "processed",
    "audiovisual"
)

LAVDF_ROOT = r"D:\LAV-DF\extracted\LAV-DF"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FRAME_COUNT = 8
IMAGE_SIZE = 224

mtcnn = MTCNN(
    image_size=IMAGE_SIZE,
    margin=20,
    keep_all=False,
    post_process=False,
    device=DEVICE
)


def extract_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"No frames found: {video_path}")

    indices = np.linspace(
        0,
        total_frames - 1,
        FRAME_COUNT
    ).astype(int)

    frames = []

    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(frame)

        face = mtcnn(image)

        if face is None:
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
            image = np.array(image)
        else:
            image = face.permute(1, 2, 0).cpu().numpy()

        image = image.astype(np.uint8)

        frames.append(image)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No usable frames found: {video_path}")

    while len(frames) < FRAME_COUNT:
        frames.append(frames[-1].copy())

    frames = frames[:FRAME_COUNT]

    return np.stack(frames)

def extract_audio_mel(video_path):

    ffmpeg_path = r"C:\Users\Mokshith\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe"

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ) as temp_file:

        temp_wav = temp_file.name

    try:

        command = [
            ffmpeg_path,
            "-y",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            temp_wav
        ]

        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        audio, sr = librosa.load(
            temp_wav,
            sr=16000,
            mono=True
        )

    finally:

        if os.path.exists(temp_wav):
            os.remove(temp_wav)

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
        /
        (mel_db.max() - mel_db.min() + 1e-8)
    )

    mel_db = cv2.resize(
        mel_db,
        (224, 224)
    )

    return mel_db.astype(np.float32)

def process_sample(row):
    relative_path = row["file"]

    video_path = os.path.join(
        LAVDF_ROOT,
        relative_path.replace("/", os.sep)
    )

    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    frames = extract_video_frames(video_path)

    mel = extract_audio_mel(video_path)

    sample = {
        "frames": torch.from_numpy(frames),
        "mel": torch.from_numpy(mel),
        "label": torch.tensor(int(row["label"])),
        "category": row["category"],
        "file": row["file"],
        "modify_video": bool(row["modify_video"]),
        "modify_audio": bool(row["modify_audio"])
    }

    split = row["split"]

    output_split = os.path.join(
        OUTPUT_DIR,
        split
    )

    os.makedirs(output_split, exist_ok=True)

    filename = os.path.basename(
        relative_path
    ).replace(".mp4", ".pt")

    output_path = os.path.join(
        output_split,
        filename
    )

    torch.save(sample, output_path)

    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )

    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(MANIFEST_PATH)

    if args.limit is not None:
        df = df.head(args.limit)

    print("Audio-Visual Preprocessing")
    print("--------------------------")
    print("Device:", DEVICE)
    print("Samples:", len(df))
    print()

    success = 0
    failed = 0

    for index, row in df.iterrows():

        try:
            output_path = process_sample(row)

            success += 1

            print(
                f"[{index + 1}/{len(df)}] "
                f"Processed: {row['file']}"
            )

        except Exception as e:

            failed += 1

            print(
                f"[{index + 1}/{len(df)}] "
                f"FAILED: {row['file']}"
            )

            print("Reason:", repr(e))
            import traceback
            traceback.print_exc()   

    print()
    print("Preprocessing complete.")
    print("Successful:", success)
    print("Failed:", failed)


if __name__ == "__main__":
    main()
