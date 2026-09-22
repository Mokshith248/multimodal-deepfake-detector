from pathlib import Path
import numpy as np
import librosa
from PIL import Image

INPUT_DIR = Path("datasets/audio")
OUTPUT_DIR = Path("datasets/processed/audio")

SAMPLE_RATE = 16000
DURATION = 4
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def process_audio(input_path, output_path):
    try:
        audio, _ = librosa.load(
            input_path,
            sr=SAMPLE_RATE,
            mono=True,
            duration=DURATION
        )

        target_length = SAMPLE_RATE * DURATION

        if len(audio) < target_length:
            audio = np.pad(
                audio,
                (0, target_length - len(audio))
            )
        else:
            audio = audio[:target_length]

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        minimum = mel_db.min()
        maximum = mel_db.max()

        if maximum > minimum:
            mel_normalized = (
                (mel_db - minimum) /
                (maximum - minimum)
            )
        else:
            mel_normalized = np.zeros_like(mel_db)

        image = (mel_normalized * 255).astype(np.uint8)

        image = Image.fromarray(image)
        image = image.resize((224, 224))

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        image.save(output_path)

        return True

    except Exception as e:
        print(f"\nError processing {input_path}: {e}")
        return False


def process_class(class_name):
    input_dir = INPUT_DIR / class_name
    output_dir = OUTPUT_DIR / class_name

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not input_dir.exists():
        print(f"Folder not found: {input_dir}")
        return

    files = [
        file for file in input_dir.iterdir()
        if file.is_file()
        and file.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    print(
        f"\nProcessing {class_name}: "
        f"{len(files)} audio files found"
    )

    processed = 0
    skipped = 0

    for index, file in enumerate(files, start=1):

        output_file = (
            output_dir /
            f"{file.stem}.png"
        )

        if process_audio(file, output_file):
            processed += 1
        else:
            skipped += 1

        print(
            f"\rProgress: {index}/{len(files)} | "
            f"Processed: {processed} | "
            f"Skipped: {skipped}",
            end=""
        )

    print()
    print(
        f"{class_name}: "
        f"{processed} processed, "
        f"{skipped} skipped"
    )


def main():
    print("AUDIO PREPROCESSING STARTED")

    process_class("real")
    process_class("fake")

    print("\nAUDIO PREPROCESSING COMPLETED")


if __name__ == "__main__":
    main()