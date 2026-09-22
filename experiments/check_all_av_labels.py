import os
import torch
from collections import Counter

BASE_DIR = "datasets/processed/audiovisual"

for split in ["train", "dev", "test"]:

    split_dir = os.path.join(
        BASE_DIR,
        split
    )

    files = [
        os.path.join(split_dir, f)
        for f in os.listdir(split_dir)
        if f.endswith(".pt")
    ]

    files.sort()

    labels = Counter()
    categories = Counter()

    for file in files:

        data = torch.load(
            file,
            map_location="cpu"
        )

        labels[int(data["label"])] += 1

        categories[
            str(data.get("category", "unknown"))
        ] += 1

    print()
    print("=" * 40)
    print(split.upper())
    print("=" * 40)

    print("Total:", len(files))

    print("Labels:")
    for label, count in sorted(labels.items()):
        name = "FAKE" if label == 0 else "REAL"
        print(
            f"  {label} ({name}): {count}"
        )

    print("Categories:")
    for category, count in sorted(categories.items()):
        print(
            f"  {category}: {count}"
        )
        