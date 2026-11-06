"""Crop and align manually annotated scans; write images, masks, and metadata."""

import csv
import re
from collections import Counter
from datetime import timedelta

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from config import DATA, DPIS, FLIP, INOCULATION, MIXED, SOURCE_FUNGI, organism_names

def read_image(path, mode):
    with Image.open(path) as image:
        return np.array(image.getchannel("A").convert("1") if mode == "A"
                        else image.convert(mode))


def load_scan(path):
    image = read_image(path / "Layer 0.png", "RGB")
    plate = read_image(path / "plates.png", "1")
    groups = []
    for group in sorted(p for p in path.iterdir() if p.is_dir()):
        files = sorted(group.glob("*.png"),
                       key=lambda p: int(re.search(r"Layer (\d+)", p.name)[1]))
        if len(files) != len(DPIS):
            raise ValueError(f"Expected {len(DPIS)} masks in {group}")
        masks = []
        for file in files:
            mask = ndi.binary_fill_holes(read_image(file, "A"))
            masks.append((mask | masks[-1] if masks else mask) & plate)
        groups.append(np.stack(masks))
    if len(groups) not in (1, 2):
        raise ValueError(f"Expected one or two colony groups in {path}")
    if len(groups) == 1:
        groups.insert(0, np.zeros_like(groups[0]))
    return image, plate, np.stack(groups)


def centers(masks):
    return np.array([ndi.center_of_mass(m) if m.any() else np.array(m.shape) / 2
                     for m in masks])


def rotate(array, angle, flip, mask=False):
    result = ndi.rotate(array, angle, reshape=False, order=0 if mask else 3)
    result[[0, -1]] = 0
    result[:, [0, -1]] = 0
    if mask:
        result = ndi.binary_opening(result)
    return np.rot90(result, 2) if flip else result


def crop_plates(image, plate, masks, name):
    labels, count = ndi.label(ndi.binary_dilation(plate))
    if name in MIXED and count != len(MIXED[name]):
        raise ValueError(f"Unexpected plate count in {name}: {count}")
    for i, region in enumerate(ndi.find_objects(labels)):
        p = labels[region] == i + 1
        im = image[region] * p[..., None]
        m = masks[:, :, region[0], region[1]] * p
        y, x = (centers(m[:, 0])[0] - centers(m[:, 0])[1]).astype(int)
        angle = np.degrees(np.arctan2(y, x))
        im = rotate(im, angle, name in FLIP)
        m = np.array([[rotate(frame, angle, name in FLIP, mask=True)
                       for frame in group] for group in m])
        if centers(m[:, 0])[0, 1] > centers(m[:, 0])[1, 1]:
            m = m[::-1]
        # The reference corrects a swapped annotation on the second MN5-BB plate.
        if name == "MN5-BB-0" and i == 1:
            im, m = im[:, ::-1], m[::-1, :, :, ::-1]
        yield i, im, m


def preprocess():
    scans = sorted(p for p in (DATA / "raw/images").iterdir() if p.is_dir())
    scans.sort(key=lambda p: "4-0" in p.name or " 3-0" in p.name)
    if not scans:
        raise ValueError("No scans found in data/raw/images")
    for folder in ("images", "masks"):
        (DATA / folder).mkdir(exist_ok=True)
    replicates, rows = Counter(), []
    for scan in scans:
        fungus = next((code for source, code in SOURCE_FUNGI.items()
                       if source in scan.name.lower()), "None")
        bacterium = next((b for b in ("BA", "BB", "BC", "BD")
                          if b in scan.name), "None")
        image, plate, masks = load_scan(scan)
        for i, image_crop, colony_masks in crop_plates(image, plate, masks, scan.name):
            b = MIXED[scan.name][i] if scan.name in MIXED else bacterium
            if fungus == b == "None":
                continue
            replicates[fungus, b] += 1
            replicate = replicates[fungus, b]
            for j, dpi in enumerate(DPIS):
                day = (INOCULATION + timedelta(days=dpi)).isoformat()
                stem = f"{fungus}_{b}_{replicate}_{day}"
                # Crops are ordered left/right: fungus=63, bacterium=191.
                mask = colony_masks[0, j] * 63 + colony_masks[1, j] * 191
                mask[mask > 250] = 255
                if "None" in (fungus, b):
                    mask[mask > 0] = 255
                Image.fromarray(image_crop).save(DATA / "images" / f"{stem}.jpg")
                Image.fromarray(mask.astype(np.uint8)).save(DATA / "masks" / f"{stem}.png")
                rows.append(dict(fungus=fungus, bacterium=b, replicate=replicate,
                                 date=day, dpi=dpi, scan=scan.name, plate_index=i,
                                 **organism_names(fungus, b)))
        print(f"Processed {scan.name}", flush=True)
    with (DATA / "metadata.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} image/mask pairs from {sum(replicates.values())} plates.")


if __name__ == "__main__":
    preprocess()
