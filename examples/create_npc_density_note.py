"""Create a typeset PDF note about the density entering the NPC optical depth."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def page(title: str):
    figure = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    figure.text(0.08, 0.94, title, fontsize=20, weight="bold", va="top")
    return figure


def text(figure, y, value, *, size=12, weight="normal", color="black"):
    figure.text(
        0.09,
        y,
        value,
        fontsize=size,
        weight=weight,
        color=color,
        va="top",
        linespacing=1.45,
    )


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "docs" / "npc_target_density_note.pdf"
    with PdfPages(output) as pdf:
        figure = page("Which density belongs in the NPC optical depth?")
        text(
            figure,
            0.885,
            "Short answer: use the number density of the collision targets, not\n"
            "automatically the total mass density divided by the proton mass.",
            size=14,
            weight="bold",
            color="#174a7e",
        )
        text(figure, 0.79, "For matter made of free protons and neutrons:", size=13)
        figure.text(0.5, 0.705, r"$n_b=\dfrac{\rho}{m_p}$", fontsize=24, ha="center")
        figure.text(
            0.5,
            0.62,
            r"$n_p=Y_e n_b,\qquad n_n=(1-Y_e)n_b$",
            fontsize=24,
            ha="center",
        )
        text(
            figure,
            0.53,
            "Here nb is the total baryon number density.  The electron fraction Ye\n"
            "sets the proton fraction only while the material is represented as free\n"
            "nucleons and charge neutrality applies.",
        )
        text(figure, 0.405, "The target depends on the projectile:", size=13, weight="bold")
        figure.text(
            0.5,
            0.31,
            r"$\tau_{n\rightarrow p}=n_p\,\sigma_{np}\,\dfrac{\Delta r}{\Gamma_a}$",
            fontsize=22,
            ha="center",
        )
        figure.text(
            0.5,
            0.22,
            r"$\tau_{p\rightarrow n}=n_n\,\sigma_{pn}\,\dfrac{\Delta r}{\Gamma_a}$",
            fontsize=22,
            ha="center",
        )
        text(
            figure,
            0.12,
            "Therefore one composition-independent optical depth cannot describe both\n"
            "legs of a neutron--proton converter cycle.",
            size=13,
            weight="bold",
        )
        pdf.savefig(figure)
        plt.close(figure)

        figure = page("Why the distinction matters for BNS ejecta")
        text(
            figure,
            0.875,
            "Example: neutron-rich material with Ye = 0.10",
            size=15,
            weight="bold",
        )
        figure.text(
            0.5,
            0.77,
            r"$n_p=0.10\,n_b,\qquad n_n=0.90\,n_b$",
            fontsize=25,
            ha="center",
        )
        figure.text(
            0.5,
            0.68,
            r"$\tau_{n\rightarrow p}=0.10\,\tau_{\rm baryon}$",
            fontsize=25,
            ha="center",
        )
        text(
            figure,
            0.59,
            "If the present calculation gives tau_baryon = 4.4--6, the free-proton\n"
            "approximation gives tau_(n->p) = 0.44--0.60 for Ye = 0.10.  This lies\n"
            "inside the tau_pn = 0.1--2 range explored by Kashiyama et al. (2013).",
            size=13,
        )
        text(figure, 0.43, "Important limitation", size=15, weight="bold", color="#a13222")
        text(
            figure,
            0.385,
            "After nuclei form, 1-Ye is NOT the free-neutron mass fraction.  A realistic\n"
            "composition must distinguish free protons, free neutrons, and nuclei:",
            size=13,
        )
        figure.text(
            0.5,
            0.28,
            r"$X_p^{\rm free},\qquad X_n^{\rm free},\qquad X_{\rm nuclei}$",
            fontsize=24,
            ha="center",
        )
        text(
            figure,
            0.19,
            "Collisions with heavy nuclei need appropriate nuclear cross sections and\n"
            "fragmentation physics.  Treating them as free pn collisions is not justified.",
            size=13,
        )
        pdf.savefig(figure)
        plt.close(figure)

        figure = page("Recommended quantities for the Monte Carlo interface")
        text(figure, 0.875, "Export separately at every trajectory sample:", size=14, weight="bold")
        text(
            figure,
            0.82,
            "1.  Mass density:  rho\n"
            "2.  Total baryon density:  nb = rho/mp\n"
            "3.  Electron fraction:  Ye\n"
            "4.  Free-proton density:  np = Xp_free nb\n"
            "5.  Free-neutron density:  nn = Xn_free nb\n"
            "6.  tau_(n->p) and tau_(p->n)\n"
            "7.  tau_baryon only as a reference upper-bound calculation",
            size=13,
        )
        text(figure, 0.56, "If only Ye is available", size=15, weight="bold")
        text(
            figure,
            0.515,
            "Use Xp_free = Ye and Xn_free = 1-Ye only as an explicitly labelled\n"
            "free-nucleon approximation.  Give the Monte Carlo both species densities\n"
            "so it can select the target appropriate to the projectile at each step.",
            size=13,
        )
        text(figure, 0.37, "Do not mix two different optical depths", size=15, weight="bold")
        figure.text(
            0.5,
            0.285,
            r"$\tau_{pn}:\ \mathrm{hadronuclear\ collisions}$",
            fontsize=20,
            ha="center",
        )
        figure.text(
            0.5,
            0.22,
            r"$\tau_{\gamma}:\ \mathrm{photon/electron\ breakout\ opacity}$",
            fontsize=20,
            ha="center",
        )
        text(
            figure,
            0.145,
            "Changing the pn target fraction does not automatically change the shock-\n"
            "breakout opacity.  The latter requires its own composition-dependent model.",
            size=12,
        )
        text(
            figure,
            0.06,
            "Primary reference: Kashiyama, Murase & Meszaros, Phys. Rev. D 88,\n"
            "015005 (2013), arXiv:1304.1945, especially equations (5)--(7).",
            size=10,
            color="#444444",
        )
        pdf.savefig(figure)
        plt.close(figure)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
