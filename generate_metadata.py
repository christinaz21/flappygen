import json
import os
import random

# Set dataset directory
data_dir = "data/flappy_bird/collected"
metadata_path = os.path.join("data/flappy_bird", "metadata.json")

# Get all files in the directory
all_files = sorted([f for f in os.listdir(data_dir) if f.endswith((".png", ".jpg", ".mp4", ".npy"))])  # Add relevant extensions
all_files = [os.path.splitext(f)[0] for f in all_files]

# Shuffle for randomness
random.seed(42)  # Ensures reproducibility
random.shuffle(all_files)

# Split into 90% training, 10% validation
split_idx = int(0.9 * len(all_files))
train_files = all_files[:split_idx]
val_files = all_files[split_idx:]

# Save metadata.json
metadata = {
    "training": train_files,
    "validation": val_files
}

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"✅ Metadata saved: {metadata_path}")
print(f"📂 Training files: {len(train_files)}, Validation files: {len(val_files)}")
