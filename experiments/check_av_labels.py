import os
import torch
from collections import Counter

TEST_DIR = "datasets/processed/audiovisual/test"

files = [
    os.path.join(TEST_DIR, f)
    for f in os.listdir(TEST_DIR)
    if f.endswith(".pt")
]

files.sort()

print("AUDIO-VISUAL TEST LABEL CHECK")
print("============================")
print("Total files:", len(files))

label_counts = Counter()
category_counts = Counter()
modify_counts = Counter()

for file in files:

    data = torch.load(
        file,
        map_location="cpu"
    )

    label = int(data["label"])

    category = data.get(
        "category",
        "unknown"
    )

    modify_video = data.get(
        "modify_video",
        None
    )

    modify_audio = data.get(
        "modify_audio",
        None
    )

    label_counts[label] += 1
    category_counts[str(category)] += 1

    modify_counts[
        (
            bool(modify_video),
            bool(modify_audio)
        )
    ] += 1


print()
print("LABEL COUNTS")
print("------------")

for label, count in sorted(
    label_counts.items()
):

    name = (
        "FAKE"
        if label == 0
        else "REAL"
        if label == 1
        else "UNKNOWN"
    )

    print(
        f"Label {label} ({name}): {count}"
    )


print()
print("CATEGORY COUNTS")
print("---------------")

for category, count in sorted(
    category_counts.items()
):

    print(
        f"{category}: {count}"
    )


print()
print("MODIFICATION COUNTS")
print("--------------------")

for modifications, count in sorted(
    modify_counts.items()
):

    video, audio = modifications

    print(
        f"Video modified={video}, "
        f"Audio modified={audio}: {count}"
    )


print()
print("EXPECTED TEST DISTRIBUTION")
print("---------------------------")
print("REAL: 100")
print("VIDEO FAKE: 100")
print("AUDIO FAKE: 100")
print("AUDIO + VIDEO FAKE: 100")
print()
print("Expected total FAKE: 300")
print("Expected total REAL: 100")
print("Expected total: 400")