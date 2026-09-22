from pathlib import Path
import cv2
from PIL import Image
from facenet_pytorch import MTCNN

INPUT_DIR = Path("datasets/video")
OUTPUT_DIR = Path("datasets/processed/video")

FRAME_INTERVAL = 10
IMAGE_SIZE = (224, 224)

mtcnn = MTCNN(
    image_size=224,
    margin=20,
    keep_all=False,
    post_process=False,
    device="cpu"
)

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def process_video(video_path, output_dir):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"\nCould not open: {video_path}")
        return 0

    frame_number = 0
    saved = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_number % FRAME_INTERVAL == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)

            try:
                face = mtcnn(image)

                if face is not None:
                    face = face.permute(1, 2, 0).byte().numpy()
                    face_image = Image.fromarray(face)
                    face_image = face_image.resize(IMAGE_SIZE)

                    output_dir.mkdir(parents=True, exist_ok=True)

                    output_file = (
                        output_dir /
                        f"{video_path.stem}_frame_{frame_number:06d}.jpg"
                    )

                    face_image.save(output_file, quality=95)
                    saved += 1

            except Exception as e:
                print(f"\nError processing frame {frame_number}: {e}")

        frame_number += 1

    cap.release()

    return saved


def process_class(class_name):
    input_dir = INPUT_DIR / class_name
    output_dir = OUTPUT_DIR / class_name

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Folder not found: {input_dir}")
        return

    videos = [
        file for file in input_dir.iterdir()
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    print(f"\nProcessing {class_name}: {len(videos)} videos found")

    total_saved = 0

    for index, video in enumerate(videos, start=1):
        print(f"\nVideo {index}/{len(videos)}: {video.name}")

        saved = process_video(video, output_dir)
        total_saved += saved

        print(f"Frames saved: {saved}")

    print(f"\n{class_name}: {total_saved} face frames saved")


def main():
    print("VIDEO PREPROCESSING STARTED")

    process_class("real")
    process_class("fake")

    print("\nVIDEO PREPROCESSING COMPLETED")


if __name__ == "__main__":
    main()