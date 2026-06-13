"""
Shared utilities for ProZorro analysis notebooks.

Import at the top of every notebook:
    from utils import setup_matplotlib, PROJECT_ROOT, DATA_DIR, RESULTS_DIR
    from utils import SEVERITY_COLORS, RISK_COLORS, validate_dataframe, require_file, jaccard
    setup_matplotlib()
"""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path helpers — work regardless of cwd or how the notebook is launched
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """Walk up from the current directory until we find one that contains src/."""
    for p in [Path().resolve(), *Path().resolve().parents]:
        if (p / "src").is_dir():
            return p
    raise RuntimeError(
        "Cannot locate project root (expected a parent directory containing src/). "
        "Run notebooks from the project root or the notebooks/ subdirectory."
    )


PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Make src/ importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Matplotlib defaults — call setup_matplotlib() once per notebook
# ---------------------------------------------------------------------------

def setup_matplotlib() -> None:
    """Apply consistent plot style across all notebooks."""
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({
        "figure.figsize":  (12, 6),
        "figure.dpi":      100,
        "axes.titlesize":  14,
        "axes.labelsize":  12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "font.family":     "DejaVu Sans",
    })


# Shared colour palettes
SEVERITY_COLORS = {
    "critical": "#d32f2f",
    "high":     "#f57c00",
    "medium":   "#fbc02d",
    "low":      "#388e3c",
    "minimal":  "#1976d2",
}

RISK_COLORS = {
    "Critical": "#d32f2f",
    "High":     "#f57c00",
    "Medium":   "#fbc02d",
    "Low":      "#388e3c",
    "Minimal":  "#1976d2",
}

# ---------------------------------------------------------------------------
# Reusable analytics helpers
# ---------------------------------------------------------------------------

def jaccard(set_a, set_b) -> float:
    """Jaccard similarity coefficient between two collections."""
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def validate_dataframe(df, required_cols: list, name: str = "DataFrame") -> None:
    """Assert required columns exist and print a basic quality summary."""
    missing = [c for c in required_cols if c not in df.columns]
    assert not missing, f"{name} is missing columns: {missing}"
    nan_rate = df.isnull().mean().mean()
    print(f"  {name}: {len(df):,} rows x {len(df.columns)} cols  |  mean NaN rate: {nan_rate:.1%}")


def require_file(path, hint: str = "") -> Path:
    """Assert a result file exists before loading it (fail fast with a clear message)."""
    p = Path(path)
    if not p.is_absolute():
        p = RESULTS_DIR / p
    msg = f"Required file not found: {p}"
    if hint:
        msg += f"\n  Hint: {hint}"
    assert p.exists(), msg
    return p
