import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PLOT SETTINGS
# ============================================================

plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 17
})


# ============================================================
# SETTINGS
# ============================================================

# Folders containing the Metafor .ascii files
FOLDERS = {
    r"$\eta_k = 0$": Path(r"D:\ULiege\ASM\outputMetafor\NL_Kin\NL_Kin_0"),
    r"$\eta_k = 100$": Path(r"D:\ULiege\ASM\outputMetafor\NL_Kin\NL_Kin_100"),
    r"$\eta_k = 200$": Path(r"D:\ULiege\ASM\outputMetafor\NL_Kin\NL_Kin_200"),
    r"$\eta_k = 300$": Path(r"D:\ULiege\ASM\outputMetafor\NL_Kin\NL_Kin_300"),
}

# Plots folder
DATA_PLOTS = Path(r"D:\ULiege\ASM\outputMetafor\Graphiques")

# Save figures automatically
SAVE_FIGURES = True


# ============================================================
# LOAD DATA
# ============================================================

def load_data(folder, filename):
    filepath = folder / f"{filename}.ascii"

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    return np.atleast_1d(np.loadtxt(filepath))


FILES = [
    "EPL",
    "A_XX",
    "A_YY",
    "A_ZZ",
]


data = {}

for model, folder in FOLDERS.items():

    data[model] = {}

    # Load time vector
    data[model]["time"] = load_data(folder, "time")

    # Load parameters
    for filename in FILES:

        values = load_data(folder, filename)

        if len(values) != len(data[model]["time"]):
            raise ValueError(
                f"{model}/{filename}: {len(values)} values, "
                f"but time contains {len(data[model]['time'])} values."
            )

        data[model][filename] = values


# ============================================================
# CALCULATE EQUIVALENT BACKSTRESS
# ============================================================

# Equivalent backstress: sqrt(3/2 * alpha_ij * alpha_ij)
# Assuming zero shear backstress components.

for model in FOLDERS:

    ax = data[model]["A_XX"]
    ay = data[model]["A_YY"]
    az = data[model]["A_ZZ"]

    data[model]["Alpha_Equivalent"] = np.sqrt(
        1.5 * (
            ax**2
            + ay**2
            + az**2
        )
    )


# ============================================================
# PLOT FUNCTION
# ============================================================

def plot_parameter(filename, ylabel, output_name, legend_loc):

    fig, ax = plt.subplots()

    # Plot all curves with identical formatting
    for model in FOLDERS:

        ax.plot(
            data[model]["time"],
            data[model][filename],
            label=model,
            linewidth=1.5
        )

    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(ylabel)

    ax.grid(True, linewidth=0.5, alpha=0.3)

    ax.legend(
        fontsize=14,
        loc=legend_loc,
        frameon=True
    )

    fig.tight_layout()

    if SAVE_FIGURES:
        DATA_PLOTS.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            DATA_PLOTS / output_name,
            format="eps"
        )


# ============================================================
# PLOT 1: EQUIVALENT BACKSTRESS
# ============================================================

plot_parameter(
    "Alpha_Equivalent",
    r"$\bar{\alpha}$ [MPa]",
    "NL_alpha.eps",
    "upper right"
)


# ============================================================
# PLOT 2: EQUIVALENT PLASTIC STRAIN
# ============================================================

plot_parameter(
    "EPL",
    r"$\bar{\varepsilon}^{p}$ [-]",
    "NL_epsilon.eps",
    "upper left"
)


# ============================================================
# DISPLAY
# ============================================================

plt.show()