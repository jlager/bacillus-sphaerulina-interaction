# Bacillus–Sphaerulina Interaction

Lightweight image processing and growth modeling for Bacillus–Sphaerulina plate experiments using manually annotated colony masks.

## Setup

- Install dependencies with `uv sync --python 3.12 --locked` (uses `uv.lock` and creates the project-local `.venv/`).
- Activate the virtual environment with `source .venv/bin/activate`.
- Dataset layout: `data/raw/images/<scan>/` with `Layer 0.png`, `plates.png`, and colony subfolders containing numbered `Layer <n>.png` masks with alpha channels.
- Data are stored locally and excluded from Git.

## Processing & Modeling

- Experiment settings, strain names, plate assignments, and image orientation corrections live in `config.py`.
- Run `python main.py` to crop and align plates, extract colony area and horizontal growth, and fit logistic growth curves to individual replicates and median observations.
- Cropped images and masks save to `data/images/` and `data/masks/`; metadata and measurements save to `data/metadata.csv` and `data/features.csv`.
- Model parameters and fitted curves save to `data/modeling.csv` and `data/model_curves.csv`.
- To run individual stages, use `python preprocessing.py`, `python features.py`, `python modeling.py`, or `python visualization.py`. Run them in that order initially; later stages reuse the saved outputs from earlier stages.

## Visuals

- Growth and fitted growth-rate plots save to `data/figures/one_vs_one/` and `data/figures/one_vs_all/`.
- Colony growth overlays save to `data/figures/overlays/`.
- Figures export as PNG and SVG; formats are configured with `FIGURE_FORMATS` in `config.py`.

## License

Licensed under the GNU General Public License, version 3. See [LICENSE](LICENSE).
