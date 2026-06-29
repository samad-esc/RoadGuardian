import os
import shutil
from pathlib import Path

ROOT = Path("datasets")

MERGED = ROOT / "merged"

# Final class ids
POTHOLE = 0
CRACK = 1
WATER = 2
DEBRIS = 3


def ensure_dirs(split):
    (MERGED / split / "images").mkdir(parents=True, exist_ok=True)
    (MERGED / split / "labels").mkdir(parents=True, exist_ok=True)


for split in ["train", "valid", "test"]:
    ensure_dirs(split)


# ----------------------------
# Helper
# ----------------------------

def copy_image(img_path, dst_name, split):
    shutil.copy(
        img_path,
        MERGED / split / "images" / dst_name
    )


def write_label(dst_name, split, labels):
    with open(MERGED / split / "labels" / dst_name, "w") as f:
        f.writelines(labels)


# ----------------------------
# Merge RDD2022
# ----------------------------

print("Merging RDD2022...")

split_map = {
    "train": "train",
    "val": "valid",
    "test": "test"
}

for src_split, dst_split in split_map.items():

    img_dir = ROOT / "rdd2022" / "RDD_SPLIT" / src_split / "images"
    lbl_dir = ROOT / "rdd2022" / "RDD_SPLIT" / src_split / "labels"

    for img in img_dir.iterdir():

        label = lbl_dir / (img.stem + ".txt")

        if not label.exists():
            continue

        new_lines = []

        with open(label) as f:

            for line in f:

                parts = line.strip().split()

                cls = int(parts[0])

                if cls in [0, 1, 2]:
                    cls = CRACK

                elif cls == 3:
                    cls = POTHOLE

                else:
                    continue

                parts[0] = str(cls)

                new_lines.append(" ".join(parts) + "\n")

        if len(new_lines) == 0:
            continue

        new_img = "rdd_" + img.name
        new_lbl = "rdd_" + label.name

        copy_image(img, new_img, dst_split)
        write_label(new_lbl, dst_split, new_lines)


# ----------------------------
# Generic dataset
# ----------------------------

def merge_simple(dataset, class_id):

    print(f"Merging {dataset}...")

    for split in ["train", "valid", "test"]:

        img_dir = ROOT / dataset / split / "images"
        lbl_dir = ROOT / dataset / split / "labels"

        for img in img_dir.iterdir():

            label = lbl_dir / (img.stem + ".txt")

            if not label.exists():
                continue

            new_lines = []

            with open(label) as f:

                for line in f:

                    p = line.strip().split()

                    p[0] = str(class_id)

                    new_lines.append(" ".join(p) + "\n")

            new_img = dataset + "_" + img.name
            new_lbl = dataset + "_" + label.name

            copy_image(img, new_img, split)
            write_label(new_lbl, split, new_lines)


merge_simple("debris", DEBRIS)
merge_simple("waterlogging", WATER)

print("Done.")