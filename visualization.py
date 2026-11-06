"""Generate growth plots and colony overlays from processed data."""

import csv
import os
from collections import defaultdict

import numpy as np
from scipy import ndimage as ndi

from config import DATA, DPIS, FIGURE_FORMATS, MICROBES, microbe_label, microbe_name
from modeling import logistic
from preprocessing import read_image

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


if __name__ == "__main__":
    generate_figures()
    generate_collages()
