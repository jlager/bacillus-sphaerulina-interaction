"""Shared experiment settings and strain labels."""

from datetime import date
from pathlib import Path

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


# Plate order and orientation
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
