from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "results" / "model_results.csv"
ARCHITECTURE_PATH = BASE_DIR / "results" / "architectures.json"

EXPORT_FOLDER = BASE_DIR.parent / "charts_export"

COLOR_PALETTE = [
    "#0077BB",  # Deep Sky Blue
    "#EE7733",  # Vibrant Orange
    "#009988",  # Teal
    "#EE3377",  # Magenta
    "#CCBB44",  # Goldenrod
    "#33BBEE",  # Cyan
]

PERFORMANCE_COLORS = {
    "Tree-Based": "#0077BB",
    "DL": "#EE7733",
    "ANN_Raw": "#009988",
    "Probabilistic": "#EE3377",
    "Hybrid": "#CCBB44",
    "Ensemble": "#33BBEE"
}

HIGH_CONTRAST_PALETTE = [COLOR_PALETTE[0], COLOR_PALETTE[1], COLOR_PALETTE[2]]
DEFAULT_BG = "white"
DEFAULT_FONT_COLOR = "black"

MODELS_PATH = BASE_DIR.parent / "best_models_results"