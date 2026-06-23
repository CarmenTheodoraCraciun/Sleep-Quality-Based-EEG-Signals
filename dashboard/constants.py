# Shared dashboard configuration values, palettes, and file path constants.

from pathlib import Path

COLOR_PALETTE = [
    "#0077BB",  # Deep Sky Blue
    "#EE7733",  # Vibrant Orange
    "#009988",  # Teal
    "#EE3377",  # Magenta
    "#CCBB44",  # Goldenrod
    "#33BBEE",  # Cyan
]

MODELS_PATH = './best_models_results/'

HIGH_CONTRAST_PALETTE = [COLOR_PALETTE[0], COLOR_PALETTE[1], COLOR_PALETTE[2]]
DEFAULT_BG = "white"
DEFAULT_FONT_COLOR = "black"

# Resolve data file locations relative to the dashboard package location.
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "results" / "model_results.csv"
ARCHITECTURE_PATH = BASE_DIR / "results" / "architectures.json"
EXPORT_FOLDER = Path(__file__).resolve().parent.parent / "charts_export"
