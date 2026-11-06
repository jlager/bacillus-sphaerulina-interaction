"""Extract colony area and horizontal growth from processed masks."""

import csv

import numpy as np
from scipy import ndimage as ndi

from config import CM_PER_PIXEL, DATA, microbe_label, microbe_name
from preprocessing import read_image

def extract_features():
    with (DATA / "metadata.csv").open() as file:
        rows = sorted(csv.DictReader(file), key=lambda row: int(row["dpi"]))
    initial_centers, features = {}, []
    for row in rows:
        stem = "_".join(row[k] for k in ("fungus", "bacterium", "replicate", "date"))
        labels = read_image(DATA / "masks" / f"{stem}.png", "L")
        single = "None" in (row["fungus"], row["bacterium"])
        for kind, value in (("bacterium", 191), ("fungus", 63)):
            microbe = row[kind]
            if microbe == "None":
                continue
            mask = labels == 255 if single else (labels == value) | (labels == 255)
            key = (row["fungus"], row["bacterium"], row["replicate"], kind)
            if key not in initial_centers:
                initial_centers[key] = (np.array(ndi.center_of_mass(mask)).astype(int)
                                        if mask.any() else None)
            center = initial_centers[key]
            distance = np.nan
            if center is not None:
                y, x = center
                columns = np.flatnonzero(mask[y])
                if columns.size:
                    # Fungus is on the left; bacterium is on the right.
                    rightward = kind == "fungus"
                    pixels = columns[-1] - x if rightward else x - columns[0]
                    distance = max(13, pixels) * CM_PER_PIXEL  # Half the 26-pixel plug.
            features.append({
                **row, "microbe": microbe, "microbe_type": kind,
                "microbe_name": microbe_name(microbe), "microbe_label": microbe_label(microbe),
                "area_cm2": round(mask.sum() * CM_PER_PIXEL**2, 4) if mask.any() else np.nan,
                "distance_cm": round(distance, 4),
                "initial_center_row": center[0] if center is not None else np.nan,
                "initial_center_col": center[1] if center is not None else np.nan,
            })
    features.sort(key=lambda row: (row["fungus"], row["bacterium"],
                                  int(row["replicate"]), row["microbe"], int(row["dpi"])))
    with (DATA / "features.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(features[0]))
        writer.writeheader()
        writer.writerows(features)
    print(f"Saved {len(features)} colony measurements to data/features.csv.")


if __name__ == "__main__":
    extract_features()
