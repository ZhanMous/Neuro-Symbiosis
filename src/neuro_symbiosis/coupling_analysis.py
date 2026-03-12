"""3-D Coupling Curve Analysis: spike_rate × effective_token_length × energy.

This script sweeps LIF hyper-parameters (beta, threshold) to vary the firing
regime of the SNN encoder, measures the resulting spike_rate and
effective_token_length, and estimates energy cost via a simplified MAC/AC
energy model.  The output is a 3-D scatter plot (+ optional surface fit) saved
to ``outputs/coupling/coupling_3d.png`` and the raw data to
``outputs/coupling/coupling_data.csv``.

The core falsifiable hypothesis:
    E_total ≈ E_snn(r) + E_attn(L)
where r = spike_rate, L = effective_token_length, and we check whether their
product term dominates the measured energy.

Run
---
    PYTHONPATH=src python src/neuro_symbiosis/coupling_analysis.py
    PYTHONPATH=src python src/neuro_symbiosis/coupling_analysis.py \\
        --out-dir outputs/coupling --n-samples 512 --n-time 256

"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Energy model constants (Horowitz 2014 / standard SNN literature estimates)
# ---------------------------------------------------------------------------
E_MAC_PJ = 4.6   # pJ per float32 multiply-accumulate (45 nm CMOS estimate)
E_AC_PJ = 0.9    # pJ per integer accumulate (spike-based, ~5× cheaper)


def estimate_energy_mj(
    spike_rate: float,
    effective_token_length: float,
    n_snn_ops: int,
    n_attn_ops_per_token: int,
) -> float:
    """Simplified energy model.

    SNN layer: spike events = spike_rate × n_snn_ops AC operations.
    Attention:  quadratic cost over effective_token_length tokens.
                n_attn_ops = effective_token_length^2 × attn_ops_per_token

    Returns energy in millijoules.
    """
    # SNN stage: only spikes generate AC ops (binary activations)
    snn_energy_pj = spike_rate * n_snn_ops * E_AC_PJ
    # Transformer attention stage: MAC cost over active token sequence
    attn_energy_pj = (effective_token_length ** 2) * n_attn_ops_per_token * E_MAC_PJ
    return (snn_energy_pj + attn_energy_pj) * 1e-9  # pJ → mJ


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(
    in_channels: int = 22,
    hidden_dim: int = 64,
    embed_dim: int = 128,
    n_time: int = 256,
    n_samples: int = 256,
    device_name: str = "cpu",
) -> list[dict]:
    """Sweep beta × threshold grid, record coupling metrics, return rows."""
    # Local import so the script is usable stand-alone.
    from neuro_symbiosis.models.snn_encoder import SNNTemporalEncoder

    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")

    beta_values = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    threshold_values = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    # Rough op counts for energy model (22ch × 64h × 256t conv + LIF)
    n_snn_ops = in_channels * hidden_dim * n_time
    # Attention: embed_dim head ops per (token, token) pair
    n_attn_ops_per_token = embed_dim

    rows: list[dict] = []

    x = torch.randn(n_samples, in_channels, n_time, device=device)

    for beta in beta_values:
        for threshold in threshold_values:
            encoder = SNNTemporalEncoder(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                embed_dim=embed_dim,
                beta=beta,
                threshold=threshold,
            ).to(device)
            encoder.eval()

            t0 = time.perf_counter()
            with torch.no_grad():
                _tokens, spike_rate_t, eff_tok_len_t = encoder(x)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            spike_rate = float(spike_rate_t.item())
            eff_tok_len = float(eff_tok_len_t.item())
            energy_mj = estimate_energy_mj(
                spike_rate,
                eff_tok_len,
                n_snn_ops=n_snn_ops,
                n_attn_ops_per_token=n_attn_ops_per_token,
            )

            rows.append(
                {
                    "beta": beta,
                    "threshold": threshold,
                    "spike_rate": round(spike_rate, 6),
                    "effective_token_length": round(eff_tok_len, 4),
                    "energy_mj": round(energy_mj, 8),
                    "latency_ms": round(latency_ms, 3),
                }
            )
            print(
                f"  beta={beta:.2f} thr={threshold:.2f} | "
                f"spike_rate={spike_rate:.4f}  eff_tok={eff_tok_len:.1f}  "
                f"E={energy_mj:.4e} mJ"
            )

    return rows


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def make_3d_plot(rows: list[dict], out_path: Path) -> None:
    """Render 3-D scatter (spike_rate, eff_token_length, energy) and save."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("[warning] matplotlib not available – skipping plot.")
        return

    spike_rates = np.array([r["spike_rate"] for r in rows])
    eff_tokens = np.array([r["effective_token_length"] for r in rows])
    energies_mj = np.array([r["energy_mj"] for r in rows])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        spike_rates,
        eff_tokens,
        energies_mj * 1e6,   # mJ → nJ for readability
        c=energies_mj,
        cmap="plasma",
        s=60,
        alpha=0.85,
    )
    fig.colorbar(sc, ax=ax, label="Energy (mJ)", pad=0.1)
    ax.set_xlabel("Spike Rate $r$", labelpad=10)
    ax.set_ylabel("Eff. Token Length $L$", labelpad=10)
    ax.set_zlabel("Energy (nJ)", labelpad=10)
    ax.set_title(
        "Coupling Surface: $r \\times L \\to E$\n"
        "(Neuro-Symbiosis energy mechanism hypothesis)",
        fontsize=11,
    )

    # Overlay iso-energy contour lines projected on the base plane
    # as a rough surface-fit check.
    try:
        from scipy.interpolate import griddata
        grid_r = np.linspace(spike_rates.min(), spike_rates.max(), 40)
        grid_l = np.linspace(eff_tokens.min(), eff_tokens.max(), 40)
        GR, GL = np.meshgrid(grid_r, grid_l)
        GE = griddata((spike_rates, eff_tokens), energies_mj, (GR, GL), method="cubic")
        ax.plot_surface(GR, GL, GE * 1e6, alpha=0.25, cmap="plasma")
    except Exception:
        pass  # scipy unavailable or insufficient points

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot saved] {out_path}")


def make_2d_marginals(rows: list[dict], out_dir: Path) -> None:
    """Produce 2-D marginal plots: r→E and L→E."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    spike_rates = [r["spike_rate"] for r in rows]
    eff_tokens = [r["effective_token_length"] for r in rows]
    energies = [r["energy_mj"] * 1e6 for r in rows]  # → nJ

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].scatter(spike_rates, energies, c=eff_tokens, cmap="viridis", s=50)
    axes[0].set_xlabel("Spike Rate $r$")
    axes[0].set_ylabel("Energy (nJ)")
    axes[0].set_title("Spike Rate vs Energy")

    axes[1].scatter(eff_tokens, energies, c=spike_rates, cmap="viridis", s=50)
    axes[1].set_xlabel("Effective Token Length $L$")
    axes[1].set_ylabel("Energy (nJ)")
    axes[1].set_title("Token Length vs Energy")

    plt.tight_layout()
    path = out_dir / "coupling_2d_marginals.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot saved] {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Coupling curve analysis script")
    p.add_argument("--out-dir", type=str, default="outputs/coupling")
    p.add_argument("--n-samples", type=int, default=256)
    p.add_argument("--n-time", type=int, default=256)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Coupling Curve Sweep ===")
    rows = run_sweep(n_samples=args.n_samples, n_time=args.n_time, device_name=args.device)

    # Save CSV
    csv_path = out_dir / "coupling_data.csv"
    fieldnames = ["beta", "threshold", "spike_rate", "effective_token_length", "energy_mj", "latency_ms"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[csv  saved] {csv_path}")

    # Plot
    make_3d_plot(rows, out_dir / "coupling_3d.png")
    make_2d_marginals(rows, out_dir)

    # Print summary table
    print("\n=== Summary Table ===")
    print(f"{'beta':>6} {'thr':>6} {'spike_rate':>12} {'eff_tok':>10} {'E(mJ)':>14}")
    for r in rows:
        print(
            f"{r['beta']:6.2f} {r['threshold']:6.2f} "
            f"{r['spike_rate']:12.4f} {r['effective_token_length']:10.2f} "
            f"{r['energy_mj']:14.4e}"
        )

    # Compute Pearson correlation as a quick hypothesis check
    import csv as _csv
    sr = np.array([r["spike_rate"] for r in rows])
    el = np.array([r["effective_token_length"] for r in rows])
    en = np.array([r["energy_mj"] for r in rows])
    corr_sr_e = float(np.corrcoef(sr, en)[0, 1])
    corr_el_e = float(np.corrcoef(el, en)[0, 1])
    corr_prod_e = float(np.corrcoef(sr * el, en)[0, 1])

    print("\n=== Correlation with Energy ===")
    print(f"  corr(spike_rate, energy)      = {corr_sr_e:+.4f}")
    print(f"  corr(eff_token_len, energy)   = {corr_el_e:+.4f}")
    print(f"  corr(r × L, energy)           = {corr_prod_e:+.4f}   ← hypothesis check")
