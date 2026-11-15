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
    "Perfect": Path(r"D:\ULiege\ASM\outputMetafor\PStress_Perf"),
    "Isotropic": Path(r"D:\ULiege\ASM\outputMetafor\PStress_Iso"),
    "Kinematic": Path(r"D:\ULiege\ASM\outputMetafor\PStress_Kin"),
    "Mixed": Path(r"D:\ULiege\ASM\outputMetafor\PStress_Mixed"),
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
    "SigmaVM",
    "Sigma_Yield",
    "Sigma_XX",
    "Sigma_YY",
    "Sigma_ZZ",
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
# CALCULATE VON MISES STRESS WITH BACKSTRESS
# ============================================================

# Sigma_VM = sqrt(3/2 * (s_ij - alpha_ij)(s_ij - alpha_ij))
# Assuming zero shear stress and backstress components.

for model in FOLDERS:

    # Stress components
    sx = data[model]["Sigma_XX"]
    sy = data[model]["Sigma_YY"]
    sz = data[model]["Sigma_ZZ"]

    # Deviatoric stress components
    mean_stress = (sx + sy + sz) / 3.0

    sxx = sx - mean_stress
    syy = sy - mean_stress
    szz = sz - mean_stress

    # Backstress components
    ax = data[model]["A_XX"]
    ay = data[model]["A_YY"]
    az = data[model]["A_ZZ"]

    # Von Mises stress accounting for backstress
    data[model]["SigmaVM"] = np.sqrt(
        1.5 * (
            (sxx - ax)**2
            + (syy - ay)**2
            + (szz - az)**2
        )
    )

    # Classical equivalent stress (without backstress)
    data[model]["Sigma_Equivalent"] = np.sqrt(
        1.5 * (
            sxx**2
            + syy**2
            + szz**2
        )
    )

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

def plot_parameter(filename, ylabel, output_name):

    fig, ax = plt.subplots()

    # Special case: Equivalent stress
    if filename == "Sigma_Equivalent":

        # Plot the hardening curve (Isotropic as representative)
        ax.plot(
            data["Isotropic"]["time"],
            data["Isotropic"][filename],
            label="Hardening",
            linewidth=1.5,
            zorder=2
        )

        # Plot Perfectly plastic last, in the foreground
        ax.plot(
            data["Perfect"]["time"],
            data["Perfect"][filename],
            label="Perfectly plastic",
            linewidth=2.5,
            zorder=10
        )

    else:

        # Plot all hardening models except Perfect first
        for model in ["Isotropic", "Kinematic", "Mixed"]:

            ax.plot(
                data[model]["time"],
                data[model][filename],
                label=model,
                linewidth=1.5,
                zorder=2
            )

        # Plot Perfectly plastic last, in the foreground
        ax.plot(
            data["Perfect"]["time"],
            data["Perfect"][filename],
            label="Perfectly plastic",
            linewidth=2.5,
            zorder=10
        )

    ax.set_xlabel(r"$\tau$ [s]")
    ax.set_ylabel(ylabel)
    if filename == "EPL":
        ax.set_ylim(0, 0.1)

    ax.grid(True, linewidth=0.5, alpha=0.3)

    if filename == "EPL":
        legend_loc = "upper left"

    elif filename in ["Alpha_Equivalent", "Sigma_Yield"]:
        legend_loc = "center right"

    else:
        legend_loc = "lower right"
    ax.legend(fontsize=14, loc=legend_loc, frameon=True)

    fig.tight_layout()

    if SAVE_FIGURES:
        DATA_PLOTS.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            DATA_PLOTS / output_name,
            format="eps"
        )


# ============================================================
# PLOT 1: VON MISES STRESS
# ============================================================

plot_parameter(
    "SigmaVM",
    r"$\bar{\sigma}_{VM}$ [MPa]",
    "PStress_VM.eps"
)


# ============================================================
# PLOT 2: YIELD STRESS
# ============================================================

plot_parameter(
    "Sigma_Yield",
    r"$\sigma_\text{y}$ [MPa]",
    "PStress_yield.eps"
)


# ============================================================
# PLOT 3: EQUIVALENT STRESS
# ============================================================

plot_parameter(
    "Sigma_Equivalent",
    r"$\bar{\sigma}$ [MPa]",
    "PStress_EqStress.eps"
)


# ============================================================
# PLOT 4: EQUIVALENT BACKSTRESS
# ============================================================

plot_parameter(
    "Alpha_Equivalent",
    r"$\bar{\alpha}$ [MPa]",
    "PStress_alpha.eps"
)


# ============================================================
# PLOT 5: EQUIVALENT PLASTIC STRAIN
# ============================================================

plot_parameter(
    "EPL",
    r"$\bar{\varepsilon}^\text{p}$ [-]",
    "PStress_epsilon.eps"
)


# ============================================================
# DISPLAY
# ============================================================

plt.show()
