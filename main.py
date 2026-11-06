"""Run the full plate-image analysis pipeline. Run: python main.py"""

from features import extract_features
from modeling import model_growth
from preprocessing import preprocess
from visualization import generate_collages, generate_figures


def main():
    preprocess()
    extract_features()
    model_growth()
    generate_figures()
    generate_collages()


if __name__ == "__main__":
    main()
