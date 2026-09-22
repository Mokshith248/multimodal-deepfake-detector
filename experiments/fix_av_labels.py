import os
import torch

TEST_DIR = "datasets/processed/audiovisual/test"

files = [
    os.path.join(TEST_DIR, f)
    for f in os.listdir(TEST_DIR)
    if f.endswith(".pt")
]

files.sort()

fixed = 0

for file in files:

    data = torch.load(
        file,
        map_location="cpu"
    )

    category = data.get("category", "")

    if category == "real":
        data["label"] = 1
    else:
        data["label"] = 0

    torch.save(
        data,
        file
    )

    fixed += 1

print()
print("AUDIO-VISUAL LABEL FIX")
print("======================")
print(f"Files updated: {fixed}")
print("REAL = 1")
print("FAKE = 0")
print()
print("Label correction completed successfully.")