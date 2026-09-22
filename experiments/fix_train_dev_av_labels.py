import os
import torch

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

    for file in files:

        data = torch.load(
            file,
            map_location="cpu"
        )

        category = data.get(
            "category",
            ""
        )

        if category == "real":
            data["label"] = 1
        else:
            data["label"] = 0

        torch.save(
            data,
            file
        )

    print(
        f"{split}: {len(files)} files fixed"
    )

print()
print("All Audio-Visual labels fixed.")
print("FAKE = 0")
print("REAL = 1")
