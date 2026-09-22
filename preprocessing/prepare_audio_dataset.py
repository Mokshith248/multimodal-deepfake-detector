from pathlib import Path
import shutil

PROTOCOL = Path(r"D:\ASVspoof2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt")
SOURCE = Path(r"D:\ASVspoof2019\LA\ASVspoof2019_LA_train\flac")

REAL_DIR = Path("datasets/audio/real")
FAKE_DIR = Path("datasets/audio/fake")

LIMIT = 1000

REAL_DIR.mkdir(parents=True, exist_ok=True)
FAKE_DIR.mkdir(parents=True, exist_ok=True)

real_count = 0
fake_count = 0

with PROTOCOL.open("r") as f:
    for line in f:
        parts = line.strip().split()

        if len(parts) < 4:
            continue

        filename = parts[1]
        label = parts[-1]

        source_file = SOURCE / f"{filename}.flac"

        if not source_file.exists():
            continue

        if label == "bonafide" and real_count < LIMIT:
            shutil.copy2(source_file, REAL_DIR / source_file.name)
            real_count += 1

        elif label == "spoof" and fake_count < LIMIT:
            shutil.copy2(source_file, FAKE_DIR / source_file.name)
            fake_count += 1

        if real_count >= LIMIT and fake_count >= LIMIT:
            break

print("AUDIO DATASET PREPARATION COMPLETED")
print(f"Real samples: {real_count}")
print(f"Fake samples: {fake_count}")
print(f"Total samples: {real_count + fake_count}")