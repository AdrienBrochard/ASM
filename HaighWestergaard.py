
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D


# ============================================================
# PLOT SETTINGS
# ============================================================

plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 19
})


# ============================================================
# SETTINGS
# ============================================================

FOLDERS = {
    "Isotropic": Path(r"D:\ULiege\ASM\outputMetafor\Linear\PStrain_Iso"),
    "Kinematic": Path(r"D:\ULiege\ASM\outputMetafor\Linear\PStrain_Kin"),
    "Mixed": Path(r"D:\ULiege\ASM\outputMetafor\Linear\PStrain_Mixed"),
}

DATA_PLOTS = Path(r"D:\ULiege\ASM\outputMetafor\Graphiques")

SAVE_FIGURES = True

CYCLE_DURATION = 3.0

COLORS = {
    "Isotropic": "tab:blue",
    "Kinematic": "tab:orange",
    "Mixed": "tab:green",
}

INSTANTS = [
    (0.0, "Initial"),
    (0.75, "Maximum tension"),
    (2.25, "Minimum tension"),
    (3.0, "Return to zero"),
]


# ============================================================
# LOAD DATA
# ============================================================

FILES = [
    "Sigma_XX",
    "Sigma_YY",
    "Sigma_ZZ",
    "Sigma_Yield",
    "A_XX",
    "A_YY",
    "A_ZZ",
]


def load_data(folder, filename):

    filepath = folder / f"{filename}.ascii"

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    return np.atleast_1d(np.loadtxt(filepath))


data = {}

for model, folder in FOLDERS.items():

    data[model] = {}

    data[model]["time"] = load_data(folder, "time")

    for filename in FILES:

        values = load_data(folder, filename)

        if len(values) != len(data[model]["time"]):
            raise ValueError(
                f"{model}/{filename}: incompatible data lengths."
            )

        data[model][filename] = values


# ============================================================
# HAIGH-WESTERGAARD PROJECTION
# ============================================================

# Three principal stress axes separated by 120 degrees.
#
# X = Sigma_XX - (Sigma_YY + Sigma_ZZ)/2
# Y = sqrt(3)/2 * (Sigma_YY - Sigma_ZZ)
#
# The distance from the origin is the von Mises
# equivalent stress (assuming zero shear stresses).
#
# The von Mises yield surface is therefore a circle
# with radius Sigma_Yield.


def project_to_plane(sxx, syy, szz):

    X = sxx - 0.5 * (syy + szz)

    Y = np.sqrt(3.0) / 2.0 * (syy - szz)

    return X, Y


# ============================================================
# EXTRACT DATA AT A GIVEN TIME
# ============================================================

def get_values(model, t):

    idx = np.argmin(
        np.abs(data[model]["time"] - t)
    )

    sigma_y = data[model]["Sigma_Yield"][idx]

    alpha_x = data[model]["A_XX"][idx]
    alpha_y = data[model]["A_YY"][idx]
    alpha_z = data[model]["A_ZZ"][idx]

    alpha_X, alpha_Y = project_to_plane(
        alpha_x,
        alpha_y,
        alpha_z
    )

    return sigma_y, alpha_X, alpha_Y


# ============================================================
# GENERATE YIELD SURFACE
# ============================================================

def get_circle(center_X, center_Y, radius):

    theta = np.linspace(0, 2 * np.pi, 300)

    X = center_X + radius * np.cos(theta)
    Y = center_Y + radius * np.sin(theta)

    return X, Y


# ============================================================
# PLOT FUNCTION
# ============================================================

def plot_yield_surfaces(t, event_name):

    fig, ax = plt.subplots(figsize=(9, 9))

    # --------------------------------------------------------
    # Initial yield surface
    # --------------------------------------------------------

    sigma_y_initial = data["Isotropic"]["Sigma_Yield"][0]

    X0, Y0 = get_circle(
        0.0,
        0.0,
        sigma_y_initial
    )

    # --------------------------------------------------------
    # Plot the three hardening models
    # --------------------------------------------------------

    centers = {}
    radii = {}

    for model in FOLDERS:

        sigma_y, alpha_X, alpha_Y = get_values(model, t)

        centers[model] = (alpha_X, alpha_Y)
        radii[model] = sigma_y

        X, Y = get_circle(
            alpha_X,
            alpha_Y,
            sigma_y
        )

        color = COLORS[model]

        # Yield surface
        ax.plot(
            X, Y,
            color=color,
            linewidth=2,
            zorder=3
        )

        # Backstress point
        ax.scatter(
            alpha_X,
            alpha_Y,
            color=color,
            edgecolors="black",
            linewidths=0.8,
            s=65,
            zorder=5,
        )

    # --------------------------------------------------------
    # Initial yield surface (black, foreground)
    # --------------------------------------------------------

    ax.plot(
        X0, Y0,
        color="black",
        linewidth=2.5,
        label="Initial yield surface",
        zorder=3
    )

    # --------------------------------------------------------
    # Determine plot limits
    # --------------------------------------------------------

    max_extent = sigma_y_initial

    for model in FOLDERS:

        alpha_X, alpha_Y = centers[model]
        radius = radii[model]

        max_extent = max(
            max_extent,
            abs(alpha_X) + radius,
            abs(alpha_Y) + radius,
            np.sqrt(alpha_X**2 + alpha_Y**2) + radius
        )

    limit = 1.6 * max_extent

    # --------------------------------------------------------
    # Three stress axes (120 degrees)
    # --------------------------------------------------------

    axis_length = limit * 0.83

    axis_directions = [
        (1.0, 0.0, r"$\sigma_{xx}$"),
        (-0.5, np.sqrt(3)/2, r"$\sigma_{yy}$"),
        (-0.5, -np.sqrt(3)/2, r"$\sigma_{zz}$"),
    ]

    for dx, dy, label in axis_directions:

        ax.annotate(
            "",
            xy=(axis_length * dx, axis_length * dy),
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="->",
                color="black",
                linewidth=1.4
            ),
            zorder=1
        )

        ax.text(
            axis_length * 1.07 * dx,
            axis_length * 1.07 * dy,
            label,
            ha="center",
            va="center",
            fontsize=25
        )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    ax.set_aspect("equal", adjustable="box")


    ax.set_xlabel(r"$\sigma_{D1}$ [MPa]", fontsize=25)
    ax.set_ylabel(r"$\sigma_{D2}$ [MPa]", fontsize=25)


    legend_elements = [
        Line2D([0], [0], color=COLORS["Isotropic"], linewidth=2,
            marker="o", markerfacecolor=COLORS["Isotropic"],
            markeredgecolor="black", label="Isotropic"),

        Line2D([0], [0], color=COLORS["Kinematic"], linewidth=2,
            marker="o", markerfacecolor=COLORS["Kinematic"],
            markeredgecolor="black", label="Kinematic"),

        Line2D([0], [0], color=COLORS["Mixed"], linewidth=2,
            marker="o", markerfacecolor=COLORS["Mixed"],
            markeredgecolor="black", label="Mixed"),

        Line2D([0], [0], color="black", linewidth=2.5,
            label=r"Initial $\sigma_\text{y}$"),
        ]

    ax.legend(
        handles=legend_elements,
        fontsize=18,
        loc="upper right",
        frameon=True
    )

    fig.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if SAVE_FIGURES:

        DATA_PLOTS.mkdir(
            parents=True,
            exist_ok=True
        )

        filename = (
            f"Yield_Surface_{event_name.replace(' ', '_')}.eps"
        )

        fig.savefig(
            DATA_PLOTS / filename,
            format="eps",
            bbox_inches="tight"
        )


# ============================================================
# GENERATE THE FOUR FIGURES
# ============================================================

for t, event_name in INSTANTS:

    plot_yield_surfaces(t, event_name)


# ============================================================
# DISPLAY
# ============================================================

plt.show()
