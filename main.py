"""Preprocess scans, extract features, and fit growth. Run: .venv/bin/python main.py"""

import csv
import os
import re
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import least_squares


DATA = Path(__file__).resolve().parent / "data"
FIGURE_FORMATS = ("png", "svg")  # Use ("png", "svg") to save both.
DPIS = [2, 4, 6, 8, 10, 12, 17, 23, 28, 32]
INOCULATION = date(2024, 12, 2)
CM_PER_PIXEL = 2.54 / 100  # Manual scans are 100 pixels per inch.
MICROBES = {  # Full species name, strain, and compact figure label.
    "mn5": ("Sphaerulina musiva", "MN5", "SM MN5"),
    "tn1": ("Sphaerulina musiva", "TN-1", "SM TN-1"),
    "oh-1": ("Sphaerulina musiva", "OH-1", "SM OH-1"),
    "bc14": ("Sphaerulina musiva", "BC14", "SM BC14"),
    "BA": ("Bacillus subtilis", "NCIB 3610", "BS NCIB 3610"),
    "BB": ("Bacillus subtilis", "RO-NN-1", "BS RO-NN-1"),
    "BC": ("Bacillus velezensis", "GB03", "BV GB03"),
    "BD": ("Bacillus amyloliquefaciens", "DSM7", "BA DSM7"),
}
SOURCE_FUNGI = {"mn5": "mn5", "tn1": "tn1", "oh-1": "oh-1", "sarm07": "bc14"}


def microbe_name(code, italic=False):
    if code == "None":
        return "Alone"
    species, strain, _ = MICROBES[code]
    if italic:
        species = r"$\mathit{" + species.replace(" ", r"\ ") + "}$"
    return f"{species} ({strain})"


def microbe_label(code):
    return "Alone" if code == "None" else MICROBES[code][2]


def organism_names(fungus, bacterium):
    return dict(fungus_name=microbe_name(fungus), fungus_label=microbe_label(fungus),
                bacterium_name=microbe_name(bacterium), bacterium_label=microbe_label(bacterium))


# Plate order and orientation from seed_lipopeptides/manual-experiment.ipynb.
MIXED = {
    "Bacillus rep 4-0": ["BC", "BD", "BA", "BB"],
    "Septoria mn5 rep 4-0": ["BD", "BC", "BA", "BB", "None"],
    "Septoria SARM07- rep4-0": ["BB", "BC", "BD", "BA", "None"],
    "Septoria OH-1 rep 4-0": ["BD", "BC", "BB", "BA", "None"],
    "Septoria TN1 rep 4-0": ["BD", "BC", "BA", "BB", "None"],
}
FLIP = set(MIXED) | {
    "MN5-BA-0", "SARM07-BA-0", "BB-REP1-2-0", "BC alone-0",
    "BC rep 2-0", "BC-rep 1 alone-0", "BA REP 1-3-0", "OH-1-BB-0",
    "SARM07-BD-0", "OH-1-BC rep 3-0", "BD REP 1-3-0",
    "TN1-BB REP 1-3-0", "TN1--BC rep1-3-0", "TN1-BD rep 1-3-0",
    "MN5-BB-0", "TN1-BA rep 1-3-0",
}


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


def logistic(t, u0, rate, capacity):
    return capacity / (1 + (capacity / u0 - 1) * np.exp(-rate * t))


def fit_growth(t, values):
    """Fit distance in cm with the reference workflow's logistic model and bounds."""
    t, values = np.asarray(t, float), np.asarray(values, float)
    valid = np.isfinite(t) & np.isfinite(values)
    t, values = t[valid], values[valid]
    if len(t) < 4 or len(np.unique(t)) < 4:
        raise ValueError("Logistic fitting needs at least four distinct time points")
    tau = t - t.min()  # u0 is growth at the first observation, not inoculation.
    # Match ColonyTracker: capacity is constrained above the observed maximum.
    lower = np.array([1e-8, 1e-8, min(1.001 * values.max(), 8.5 - 1e-8)])
    upper = np.array([4.25, 2., 8.5])
    capacity = min(max(1.1 * np.percentile(values, 90), 1.01 * values.max()), 8.5)
    usable = (values > 0) & (values < capacity)
    rate = (np.polyfit(tau[usable], np.log(values[usable] / (capacity - values[usable])), 1)[0]
            if usable.sum() >= 2 else 0.1)
    start = np.clip([values[0], max(rate, 0.05), capacity], lower + 1e-10, upper - 1e-10)
    fit = least_squares(lambda p: logistic(tau, *p) - values, start,
                        bounds=(lower, upper), max_nfev=10000,
                        ftol=1e-10, xtol=1e-10, gtol=1e-10)
    sse = np.sum(fit.fun**2)
    total = np.sum((values - values.mean())**2)
    grid = np.linspace(t.min(), t.max(), 301)
    prediction = logistic(grid - t.min(), *fit.x)
    speed = fit.x[1] * prediction * (1 - prediction / fit.x[2])
    peak = np.argmax(speed)
    result = dict(u0_cm=fit.x[0], rate_per_day=fit.x[1], capacity_cm=fit.x[2],
                  first_dpi=t.min(), n=len(t), success=fit.success, status=fit.status,
                  message=fit.message, nfev=fit.nfev, optimality=fit.optimality,
                  rmse_cm=np.sqrt(sse / len(t)), r2=1 - sse / total if total > 1e-12 else np.nan,
                  boundary_parameters=";".join(name + ("_lower" if x - lo < hi - x else "_upper")
                      for name, x, lo, hi in zip(
                      ["u0", "rate", "capacity"], fit.x, lower, upper)
                      if min(x - lo, hi - x) < 1e-5 * (hi - lo)),
                  jacobian_condition=np.linalg.cond(fit.jac),
                  peak_dpi=grid[peak], peak_growth_cm_per_day=speed[peak])
    return result, grid, prediction


def model_growth():
    groups = defaultdict(lambda: defaultdict(list))
    with (DATA / "features.csv").open() as file:
        for row in csv.DictReader(file):
            key = tuple(row[k] for k in ("fungus", "bacterium", "microbe"))
            groups[key][row["replicate"]].append((float(row["dpi"]), float(row["distance_cm"])))
    models, curves = [], []
    for key, replicates in groups.items():
        by_day = defaultdict(list)
        for series in replicates.values():
            for day, value in series:
                if np.isfinite(value):
                    by_day[day].append(value)
        replicates["median"] = [(day, np.median(values)) for day, values in by_day.items()]
        for replicate, series in replicates.items():
            t, values = np.array(sorted(series)).T
            result, grid, prediction = fit_growth(t, values)
            metadata = dict(zip(("fungus", "bacterium", "microbe"), key), replicate=replicate)
            metadata.update(organism_names(key[0], key[1]),
                            microbe_name=microbe_name(key[2]), microbe_label=microbe_label(key[2]))
            models.append({**metadata, **result})
            curves.extend({**metadata, "dpi": day, "distance_cm": value}
                          for day, value in zip(grid, prediction))
    for name, rows in (("modeling.csv", models), ("model_curves.csv", curves)):
        with (DATA / name).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Fitted {len(models)} logistic models; {sum(not m['success'] for m in models)} failed.")


def generate_figures():
    # Keep Matplotlib's development cache out of data/.
    os.environ.setdefault("MPLCONFIGDIR", str(DATA.parent / "scratch/matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(DATA.parent / "scratch/cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    observations, models = defaultdict(list), defaultdict(dict)
    with (DATA / "features.csv").open() as file:
        for row in csv.DictReader(file):
            key = tuple(row[k] for k in ("fungus", "bacterium", "microbe"))
            observations[key].append(row)
    with (DATA / "modeling.csv").open() as file:
        for row in csv.DictReader(file):
            key = tuple(row[k] for k in ("fungus", "bacterium", "microbe"))
            models[key][row["replicate"]] = row
    grid = np.linspace(2, 32, 301)
    group_maxima = {"Sphaerulina": [0., 0.], "Bacillus": [0., 0.]}
    fitted_curves = {}
    for key, rows in observations.items():
        group = "Sphaerulina" if key[2] == key[0] else "Bacillus"
        maxima = group_maxima[group]
        maxima[0] = max(maxima[0], np.nanmax([float(r["distance_cm"]) for r in rows
                                             if grid[0] <= float(r["dpi"]) <= grid[-1]]))
        for replicate, model in models[key].items():
            u0, rate, capacity = (float(model[k]) for k in
                                  ("u0_cm", "rate_per_day", "capacity_cm"))
            growth = logistic(grid - float(model["first_dpi"]), u0, rate, capacity)
            speed = rate * growth * (1 - growth / capacity)
            fitted_curves[key, replicate] = growth, speed
            maxima[0] = max(maxima[0], float(growth.max()))
            maxima[1] = max(maxima[1], float(speed.max()))
    names = {code: microbe_label(code) for code in [*MICROBES, "None"]}
    partner = lambda key: key[1] if key[2] == key[0] else key[0]
    plots = []
    for microbe in sorted({key[2] for key in observations}):
        keys = sorted((k for k in observations if k[2] == microbe),
                      key=lambda k: (partner(k) != "None", partner(k)))
        plots.append(("one_vs_all", microbe_label(microbe), keys))
        plots.extend(("one_vs_one", f"{microbe_label(microbe)} vs {microbe_label(partner(k))}", [k])
                     for k in keys)
    limits = []
    for folder, filename, keys in plots:
        combined = folder == "one_vs_all"
        group = "Sphaerulina" if keys[0][2] == keys[0][0] else "Bacillus"
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, layout="constrained")
        for group_index, key in enumerate(keys):
            group_color = "black" if partner(key) == "None" else f"C{group_index % 10}"
            rows = observations[key]
            for rep_index, (replicate, model) in enumerate(models[key].items()):
                median = replicate == "median"
                color = group_color if combined else ("black" if median else f"C{rep_index % 10}")
                data = sorted((r for r in rows if r["replicate"] == replicate),
                              key=lambda r: float(r["dpi"]))
                if data:
                    x = [float(r["dpi"]) for r in data]
                    y = [float(r["distance_cm"]) for r in data]
                    axes[0].scatter(x, y, s=15, color=color, alpha=.35 if combined else .75)
                u0, rate, capacity = (float(model[k]) for k in
                                      ("u0_cm", "rate_per_day", "capacity_cm"))
                growth, speed = fitted_curves[key, replicate]
                label = (names[partner(key)] if median else None) if combined else (
                    "Median fit" if median else f"Replicate {replicate}")
                for ax, values in zip(axes, (growth, speed)):
                    ax.plot(grid, values, "--" if median else "-", color=color,
                            lw=2 if median else 1, alpha=.25 if combined and not median else 1,
                            label=label)
                if (median or not combined) and capacity > u0 and rate > 0:
                    inflection = float(model["first_dpi"]) + np.log(capacity / u0 - 1) / rate
                    if grid[0] <= inflection <= grid[-1]:
                        for ax in axes:
                            ax.plot(inflection, 0, marker=(6, 2, 0), markersize=10,
                                    markeredgewidth=1.3, color=color, linestyle="none",
                                    transform=ax.get_xaxis_transform(), clip_on=False, zorder=6)
        for ax, maximum, ylabel in zip(axes, group_maxima[group],
                ("Horizontal growth (cm)", "Fitted growth rate (cm/day)")):
            ax.set_xlim(2, 32)
            ax.set_xticks([2, 8, 14, 20, 26, 32])
            ax.set_ylim(0, 1.03 * maximum if maximum > 0 else 1)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
            limits.append(dict(figure=f"{folder}/{filename}", group=group, panel=ylabel,
                               maximum=maximum, upper_limit=ax.get_ylim()[1],
                               x_min=ax.get_xlim()[0], x_max=ax.get_xlim()[1]))
        group_legend = axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1),
                                      frameon=True, fontsize=8,
                                      title="Treatment" if combined else "Replicate", title_fontsize=8)
        axes[0].add_artist(group_legend)
        style_handles = [
            Line2D([], [], color="black", marker="o", linestyle="none", markersize=4, label="Observations"),
            Line2D([], [], color="black", lw=1, label="Individual fit"),
            Line2D([], [], color="black", lw=2, linestyle="--", label="Median fit"),
            Line2D([], [], color="black", marker=(6, 2, 0), linestyle="none",
                   markersize=8, label="Inflection time"),
        ]
        axes[0].legend(handles=style_handles, loc="upper left", bbox_to_anchor=(1.02, .56),
                       frameon=True, fontsize=8, title="Plot symbols", title_fontsize=8)
        axes[1].set_xlabel("Days post inoculation")
        title = microbe_name(keys[0][2], italic=True) + (" vs all partners" if combined else
                                    " — " + ("alone" if partner(keys[0]) == "None" else
                                               "with " + names[partner(keys[0])]))
        fig.suptitle(title, fontsize=12)
        destination = DATA / "figures" / folder
        destination.mkdir(parents=True, exist_ok=True)
        for extension in FIGURE_FORMATS:
            fig.savefig(destination / f"{filename}.{extension}", dpi=180)
        plt.close(fig)
    print(f"Saved {len(plots)} growth figures as {', '.join(FIGURE_FORMATS)} in data/figures/.")
    return limits


def generate_collages():
    os.environ.setdefault("MPLCONFIGDIR", str(DATA.parent / "scratch/matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(DATA.parent / "scratch/cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = defaultdict(lambda: defaultdict(list))
    with (DATA / "metadata.csv").open() as file:
        for row in csv.DictReader(file):
            groups[row["fungus"], row["bacterium"]][int(row["replicate"])].append(row)
    destination = DATA / "figures/overlays"
    destination.mkdir(parents=True, exist_ok=True)
    colors = {kind: plt.colormaps[cmap](np.linspace(.3, .9, len(DPIS)))[:, :3]
              for kind, cmap in (("fungus", "Oranges"), ("bacterium", "Blues"))}
    for (fungus, bacterium), replicates in sorted(groups.items()):
        kinds = [kind for kind, microbe in (("bacterium", bacterium), ("fungus", fungus))
                 if microbe != "None"]
        fig, axes = plt.subplots(len(replicates), len(DPIS) + 1, squeeze=False,
                                 figsize=(22, 2.1 * len(replicates)), layout="constrained")
        for row_index, (replicate, rows) in enumerate(sorted(replicates.items())):
            rows.sort(key=lambda row: int(row["dpi"]))
            masks = []
            for row in rows:
                stem = "_".join(row[k] for k in ("fungus", "bacterium", "replicate", "date"))
                labels = read_image(DATA / "masks" / f"{stem}.png", "L")
                if "None" in (fungus, bacterium):
                    masks.append([labels == 255])
                else:
                    masks.append([np.isin(labels, [191, 255]), np.isin(labels, [63, 255])])
                if len(masks) == 1:
                    axes[row_index, 0].imshow(read_image(DATA / "images" / f"{stem}.jpg", "RGB"))
            axes[row_index, 0].set_ylabel(f"Replicate {replicate}", fontsize=10)
            h, w = masks[0][0].shape
            for kind, mask in zip(kinds, masks[0]):
                if not mask.any():
                    continue
                left = ndi.center_of_mass(mask)[1] < w / 2
                strain = microbe_label(bacterium if kind == "bacterium" else fungus)
                label_ax = axes[row_index, DPIS.index(int(rows[0]["dpi"])) + 1]
                label_ax.text(
                    .04 if left else .96, .96, f"{kind.capitalize()}\n{strain}",
                    transform=label_ax.transAxes,
                    ha="left" if left else "right", va="top", fontsize=7,
                    color=colors[kind][-1], fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=2))
            for j, row in enumerate(rows):
                ax = axes[row_index, DPIS.index(int(row["dpi"])) + 1]
                canvas = np.full((h, w, 3), .95)
                # Draw newest first so earlier colony boundaries remain visible.
                for k in range(j, -1, -1):
                    for kind, mask in zip(kinds, masks[k]):
                        color = colors[kind][DPIS.index(int(rows[k]["dpi"]))]
                        canvas[mask] = .55 * canvas[mask] + .45 * color
                        if mask.any():
                            ax.contour(mask, levels=[.5], colors="black", linewidths=.3, alpha=.8)
                ax.imshow(canvas, zorder=0)
                ax.add_patch(plt.Circle((w / 2, h / 2), min(h, w) / 2 - 2,
                                        fill=False, color="black", linewidth=.6))
                ax.set_xlim(-.5, w - .5)
                ax.set_ylim(h - .5, -.5)
            for ax in axes[row_index]:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines[:].set_visible(False)
        axes[0, 0].set_title("Original image", color="black", fontsize=11)
        for j, dpi in enumerate(DPIS):
            axes[0, j + 1].set_title(f"{dpi} DPI", color="black", fontsize=11)
        title = " × ".join(microbe_name(code, italic=True) for code in (fungus, bacterium) if code != "None")
        if "None" in (fungus, bacterium):
            title += " — alone"
        fig.suptitle(title, fontsize=15)
        for extension in FIGURE_FORMATS:
            filename = f"{microbe_label(fungus)} x {microbe_label(bacterium)}"
            fig.savefig(destination / f"{filename}.{extension}", dpi=160)
        plt.close(fig)
        print(f"Saved collage: {fungus} × {bacterium}", flush=True)


def main():
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
    extract_features()
    model_growth()
    generate_figures()
    generate_collages()


if __name__ == "__main__":
    main()
