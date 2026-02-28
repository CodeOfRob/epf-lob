import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# --- SHAP FARB-KONSTANTEN ---
SHAP_RED = "#ff0051"
SHAP_BLUE = "#008bfb"
SHAP_GRAY = "#cccccc"

# Erstelle eine kontinuierliche Colormap (Blau -> Weiß -> Rot)
# Perfekt für Heatmaps oder Korrelationen
SHAP_CMAP = LinearSegmentedColormap.from_list(
    "shap_style", [SHAP_BLUE, "#ffffff", SHAP_RED]
)

# Zentrale Farbdefinitionen (Publication Ready)
MODEL_COLORS = {
    # Boosting Modelle (Warme Töne / SHAP-Style Magenta-Einfluss)
    "cb-gpu": "#ff0051",  # SHAP Red/Magenta (Top Modell)
    "cb-cpu": "#ae0038",  # Dunkleres Rot
    "catboost": "#FF9D39",  # Dunkleres Rot
    "xgb-gpu": "#ff7f0e",  # Orange
    "xgb-cpu": "#d62728",  # Klassisches Rot
    "lgbm-gpu": "#ffbb78",  # Hell-Orange
    "lgbm-cpu": "#e377c2",  # Pink

    # Andere ML Modelle
    "rf": "#98C125",  # Grün
    "randomforest": "#98C125",  # Grün
    "mlp": "#F45F73",  # Lila
    "logreg": "#1f77b4",  # Blau
    "sgd": "#1f77b4",  # Blau
    "lasso": "#013A7D",  # Blau
    "ebm": "#00AFBD",  # Braun
    "tabpfn": "#17becf",  # Türkis

    # Baselines (Neutrale Grautöne)
    "pers": "#7f7f7f",  # Mittelgrau
    "rw": "#c7c7c7",  # Hellgrau
    "Baseline_Majority": "#dbdbdb"
}
# Zentrale Marker-Definition (Formen)
MODEL_MARKERS = {
    "cb-gpu": "o",  # Kreis
    "catboost": "o",  # Kreis
    "cb-cpu": "v",  # Dreieck unten
    "xgb-gpu": "s",  # Quadrat
    "xgb-cpu": "D",  # Diamant
    "lgbm-gpu": "p",  # Kreuz
    "lgbm-cpu": "p",  # Kreuz
    "rf": "^",  # Dreieck oben
    "mlp": "X",  # Pentagramm
    "logreg": "h",  # Hexagon
    "sgd": "h",  # Hexagon
    "lasso": "s",  # Hexagon
    "ebm": "*",  # Stern
    "tabpfn": "d",  # Kleiner Diamant
    "pers": ".",  # Nur Linie für Baselines
    "rw": "."  # Nur Linie für Baselines
}


def get_model_style(model_names):
    """
    Gibt Paletten- und Marker-Dictionaries für die Liste der Modelle zurück.
    """
    palette = {}
    markers = {}

    for model in model_names:
        if model.lower() not in MODEL_COLORS:
            print(f"⚠️ Warnung: Modell '{model}' hat keine definierte Farbe. Verwende Standardgrau.")
        if model.lower() not in MODEL_MARKERS:
            print(f"⚠️ Warnung: Modell '{model}' hat keinen definierten Marker. Verwende Standardpunkt.")
        palette[model] = MODEL_COLORS[model.lower()] if model.lower() in MODEL_COLORS else "#333333"
        markers[model] = MODEL_MARKERS[model.lower()] if model.lower() in MODEL_MARKERS else "o"

    return palette, markers


def apply_shap_style():
    """Setzt das globale Styling für Matplotlib/Seaborn."""
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['grid.color'] = '#dddddd'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    sns.set_style("whitegrid")


def get_shap_cmap():
    """Gibt die SHAP-Colormap zurück."""
    return SHAP_CMAP


from pathlib import Path
import matplotlib.pyplot as plt


def save_plot(fig, name, folder="figures"):
    """
    Speichert eine Figure als PDF (Vektor) und PNG (Raster).
    """
    target_dir = Path(folder)
    target_dir.mkdir(parents=True, exist_ok=True)

    # PDF für LaTeX (Vektorgrafik, unendlich scharf)
    fig.savefig(target_dir / f"{name}.pdf", bbox_inches='tight', format='pdf', dpi=300)

    # PNG für schnelle Vorschau oder Word (300 DPI)
    fig.savefig(target_dir / f"{name}.png", bbox_inches='tight', dpi=300)

    print(f"✅ Grafik gespeichert in: {target_dir}/{name}")
