"""One plotting style for every thesis figure (serif text, three fixed method colours)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRED, REAL, GEN, BASE, INK = "#1F4E79", "#0F766E", "#E8590C", "#8A94A0", "#1F2A37"
TIP, REC = "#B03A2E", "#3D6B99"

def apply():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK, "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "lines.linewidth": 1.4, "patch.linewidth": 0.6,
    })

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/{name}.{ext}")
    plt.close(fig)
    print("saved figures/" + name)

def panel(ax, letter):
    ax.text(-0.14, 1.04, f"({letter})", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
