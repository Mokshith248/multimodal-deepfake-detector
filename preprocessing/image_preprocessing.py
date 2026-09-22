from pathlib import Path
from PIL import Image
from facenet_pytorch import MTCNN

INPUT_DIR = Path("datasets/image")
OUTPUT_DIR = Path("datasets/processed/image")

IMAGE_SIZE = (224, 224)

mtcnn = MTCNN(
    image_size=224,
    margin=20,
    keep_all=False,
    post_process=False,
    device="cpu"
)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def process_image(input_path, output_path):
    try:
        image = Image.open(input_path).convert("RGB")

        face = mtcnn(image)

        if face is None:
            return False

        face = face.permute(1, 2, 0).byte().numpy()
        face_image = Image.fromarray(face)

        face_image = face_image.resize(IMAGE_SIZE)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        face_image.save(output_path, quality=95)

        return True

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def process_class(class_name):
    input_class_dir = INPUT_DIR / class_name
    output_class_dir = OUTPUT_DIR / class_name

    output_class_dir.mkdir(parents=True, exist_ok=True)

    if not input_class_dir.exists():
        print(f"Folder not found: {input_class_dir}")
        return

    files = [
        file for file in input_class_dir.iterdir()
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    processed = 0
    skipped = 0

    print(f"\nProcessing {class_name}: {len(files)} images found")

    for index, file in enumerate(files, start=1):
        output_file = output_class_dir / file.name

        if process_image(file, output_file):
            processed += 1
        else:
            skipped += 1

        print(
            f"\rProgress: {index}/{len(files)} | "
            f"Processed: {processed} | Skipped: {skipped}",
            end=""
        )

    print()
    print(f"{class_name}: {processed} processed, {skipped} skipped")


def main():
    print("IMAGE PREPROCESSING STARTED")

    process_class("real")
    process_class("fake")

    print("\nIMAGE PREPROCESSING COMPLETED")


if __name__ == "__main__":
    main()