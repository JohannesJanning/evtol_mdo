"""
Multistart SLSQP Optimization for Aircraft GWP Minimization
-----------------------------------------------------------
This script performs a multistart SLSQP optimization to minimize
the annual Global Warming Potential (GWP) of an aircraft concept.
It uses a combination of scaled variables, random perturbations,
and feasibility checks to identify optimal design parameters.

Dependencies:
    numpy, scipy, matplotlib
    local modules: src.analysis.evaluation_model,
                   src.parameters.model_parameters,
                   src.optimizer.constraints

Outputs:
    - Console/log summary of optimization
    - Evaluation log plot
    - Excel file with results (via write_results_to_excel)
"""

import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from src.analysis.evaluation_model import (
    full_model_evaluation,
    write_results_to_excel,
)
from src.parameters import model_parameters as p
from src.optimizer.constraints import constraint_functions

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

EVAL_LOG = []
MODEL_CALLS = 0

# Design variable bounds: [b, c, R_cruise, R_hover, c_charge]
BOUNDS = [
    (6.0, 15.0),   # wingspan b
    (1.0, 2.5),    # chord c
    (0.6, 2.5),    # cruise prop radius
    (0.5, 2.0),    # hover prop radius
    (1.0, 4.0),    # battery charging rate c_charge
]

N_STARTS = 10
# Optimizer options
MAXITER = 1000
FTOL = 1e-6
EPS = 1e-2
SEED = 42

# Initial design vector (order: [b, c, R_cruise, R_hover, c_charge]).
# Provide a documented, reproducible initial guess.
INITIAL = np.array([15, 1.0, 1.215, 1.586, 1.154])

# Indices to vary during multistart sampling (defaults to a small subset
# near the anchor to reduce generation of infeasible starts).
VARY_IDX = [0, 3, 4]

# Per-variable perturbation windows (physical units). If `WINDOWS` is None
# the sampler will use a relative fraction of the initial value for each
# variable in `VARY_IDX` (see `sample_partial_start`).
WINDOWS = {0: 0.0, 3: 0.25, 4: 0.0}

# Small epsilon used to nudge values off exact bounds (for finite-difference
# stability). 
BOUND_EPS = 1e-3


def nudge_to_open_interval(x, bounds, eps=BOUND_EPS):
    """Return `x` nudged slightly away from exact bounds.

    Nudging prevents optimizer finite-difference evaluations from landing
    exactly on bounds which can cause numerical instability.
    """
    x = x.copy()
    for i, (lo, hi) in enumerate(bounds):
        if np.isclose(x[i], lo):
            x[i] = min(lo + eps, hi)
        elif np.isclose(x[i], hi):
            x[i] = max(hi - eps, lo)
    return x


def sample_partial_start(x_anchor, bounds, vary_idx=None, windows=None, rel_window=0.1):
    """Create a randomized start near `x_anchor`.

    Parameters
    - x_anchor: physical-space anchor vector
    - bounds: list of (lo, hi) tuples matching `x_anchor`
    - vary_idx: indices to perturb (defaults to `VARY_IDX`)
    - windows: dict mapping index -> absolute perturbation half-width (physical units)
      If omitted, a relative window `rel_window * |x_anchor[i]|` is used for each
      index in `vary_idx`.

    Returns
    - x0_scaled: start in scaled [0,1] space
    - x0_phys: start in physical units
    """
    if vary_idx is None:
        vary_idx = VARY_IDX

    x0 = x_anchor.copy()
    # build per-index windows if none provided
    if windows is None:
        windows = {i: rel_window * abs(float(x_anchor[i])) for i in vary_idx}

    for i in vary_idx:
        lo, hi = bounds[i]
        w = windows.get(i, 0.0)
        x0[i] = np.random.uniform(max(lo, x_anchor[i] - w), min(hi, x_anchor[i] + w))

    x0 = nudge_to_open_interval(x0, bounds, eps=BOUND_EPS)
    return scale(x0, bounds), x0


# --------------------------------------------------------------------------- #
# Utility Functions
# --------------------------------------------------------------------------- #

def scale(x, bounds):
    """Scale parameters x into [0, 1] based on bounds."""
    return np.array([(xi - low) / (high - low) for xi, (low, high) in zip(x, bounds)])


def unscale(x_scaled, bounds):
    """Unscale [0, 1] parameters back to physical bounds."""
    return np.array([low + xi * (high - low) for xi, (low, high) in zip(x_scaled, bounds)])


def are_constraints_satisfied(constraints, tol=0.0):
    """Return True if all constraints g >= -tol are satisfied."""
    return all(g >= -tol for g in constraints)


# --------------------------------------------------------------------------- #
# Objective and Constraints
# --------------------------------------------------------------------------- #

def scaled_objective(x_scaled):
    """
    Objective function in scaled space.

    Returns
    -------
    float
        Annual GWP value (minimize).
    """
    global MODEL_CALLS
    MODEL_CALLS += 1
    x = unscale(x_scaled, BOUNDS)

    try:
        # x ordering now: b, c, R_cruise, R_hover, c_charge
        b, c, r_cruise, r_hover, c_charge = x
        results, _ = full_model_evaluation(b, c, r_cruise, r_hover, c_charge, p)
        annual_gwp_value = results.get("GWP MODEL", {}).get(
            "Ops GWP total per year", (None,)
        )[0]

        if annual_gwp_value is None or np.isnan(annual_gwp_value):
            logging.warning("Invalid evaluation (None or NaN).")
            return 1e6

        EVAL_LOG.append(annual_gwp_value)
        return annual_gwp_value
    except Exception as exc:
        logging.error(f"Exception during evaluation: {exc}")
        return 1e6


def scaled_constraints(x_scaled):
    """Constraint function in scaled space."""
    x = unscale(x_scaled, BOUNDS)
    return constraint_functions(x)


# Use same constraint vector size as profit optimization (12 constraints, includes ROC)
CONSTRAINTS = [
    {"type": "ineq", "fun": lambda x, i=i: scaled_constraints(x)[i]}
    for i in range(11)
]


# --------------------------------------------------------------------------- #
# Optimization Helpers
# --------------------------------------------------------------------------- #

def clip01(x):
    """Clip array values to [0, 1]."""
    return np.clip(x, 0.0, 1.0)


def make_perturbations(x0_scaled, n_starts, radius=0.1):
    """
    Generate perturbed starting points in scaled space.

    Parameters
    ----------
    x0_scaled : ndarray
        Base design point.
    n_starts : int
        Number of starting points.
    radius : float
        Local search radius.

    Returns
    -------
    list of ndarray
        Perturbed starting points (including original).
    """
    d = len(x0_scaled)
    starts = [x0_scaled.copy()]
    for _ in range(n_starts - 1):
        delta = np.random.normal(loc=0.0, scale=radius / 2, size=d)
        norm = np.linalg.norm(delta)
        if norm > radius > 0:
            delta *= radius / norm
        starts.append(clip01(x0_scaled + delta))
    return starts


def run_one_start(x_start_scaled):
    """
    Run one SLSQP optimization from a starting point.

    Returns
    -------
    dict
        Optimization results and feasibility.
    """
    t0 = time.time()
    try:
        res = minimize(
            scaled_objective,
            x_start_scaled,
            method="SLSQP",
            bounds=[(0, 1)] * len(BOUNDS),
            constraints=CONSTRAINTS,
            options={"disp": False, "maxiter": MAXITER, "ftol": FTOL, "eps": EPS},
        )

        x_unscaled = unscale(res.x, BOUNDS)
        cons_vals = constraint_functions(x_unscaled)
        feasible = are_constraints_satisfied(cons_vals, tol=0.0)

        return {
            "success": bool(res.success),
            "feasible": bool(feasible),
            "fun": float(res.fun),
            "x_scaled": res.x.copy(),
            "x_unscaled": x_unscaled.copy(),
            "cons": np.array(cons_vals, dtype=float),
            "nit": int(getattr(res, "nit", -1)),
            "time": time.time() - t0,
            "message": res.message,
        }

    except Exception as exc:
        return {
            "success": False,
            "feasible": False,
            "fun": np.inf,
            "x_scaled": x_start_scaled.copy(),
            "x_unscaled": unscale(x_start_scaled, BOUNDS),
            "cons": np.full(len(CONSTRAINTS), np.nan),
            "nit": -1,
            "time": time.time() - t0,
            "message": f"Exception: {exc}",
        }


def total_violation(cons_vals):
    """Return total constraint violation magnitude."""
    cons_vals = np.asarray(cons_vals)
    return float(np.abs(cons_vals[cons_vals < 0]).sum())


# --------------------------------------------------------------------------- #
# Main Routine
# --------------------------------------------------------------------------- #

def main():
    """Main routine: run multistart optimization and export results."""
    np.random.seed(SEED)

    # Use partial multistart sampling around the INITIAL vector
    x0_scaled, x0_phys = sample_partial_start(INITIAL, BOUNDS, VARY_IDX, WINDOWS)
    starts_scaled = [x0_scaled]
    # generate additional starts by small perturbations around initial
    for _ in range(N_STARTS - 1):
        pert, _ = sample_partial_start(INITIAL, BOUNDS, VARY_IDX, WINDOWS)
        starts_scaled.append(pert)

    logging.info("Starting multistart SLSQP (N=%d) around anchor...", N_STARTS)

    all_runs = []
    best_feasible, best_idx = None, None

    for i, x_start in enumerate(starts_scaled, start=1):
        logging.info("-- Start %d/%d --", i, N_STARTS)
        info = run_one_start(x_start)
        all_runs.append(info)

        if info["feasible"] and np.isfinite(info["fun"]):
            if best_feasible is None or info["fun"] < best_feasible["fun"]:
                best_feasible, best_idx = info, i

        status = "FEAS" if info["feasible"] else "INFEAS"
        logging.info(
            "Result: success=%s %s f=%.2f nit=%d t=%.2fs",
            info["success"], status, info["fun"], info["nit"], info["time"]
        )

    if best_feasible is None:
        logging.warning("No feasible solution found. Selecting by min violation.")
        violations = [
            total_violation(r["cons"]) if r["cons"] is not None else np.inf
            for r in all_runs
        ]
        best_idx = int(np.argmin(violations))
        best_feasible = all_runs[best_idx]

    # Summary
    logging.info("=== Best Result (Start %d) ===", best_idx)
    logging.info("Feasible=%s, Success=%s", best_feasible["feasible"], best_feasible["success"])
    logging.info("Objective (min annual GWP kg CO₂): %.2f", best_feasible["fun"])
    logging.info("x* (unscaled): %s", np.round(best_feasible["x_unscaled"], 6))
    logging.info(
        "Constraint min: %.3e (>=0 feasible)", np.min(best_feasible["cons"])
    )
    # Define labels to match your constraint_functions list
    cons_labels = [
        "Wing vs Rotor Spacing", "Vertiport Span (15m)", "MTOM (5700kg)",
        "Noise Cruise (67dB)", "Noise Climb (67dB)", "Noise Hover (77dB)",
        "RPM Cruise (3000)", "RPM Climb (3000)", "RPM Hover (3000)",
        "Speed Cruise (129m/s)", "Speed Climb (129m/s)"
    ]

    logging.info("--- Constraint Residuals (Positive = Feasible, ~0 = Active) ---")
    for label, val in zip(cons_labels, best_feasible["cons"]):
        logging.info(f"{label:25}: {val:>10.4f}")

    # Plot evaluations
    plt.figure()
    plt.plot(EVAL_LOG)
    plt.xlabel("Evaluation")
    plt.ylabel("Annual GWP (kg CO₂)")
    plt.title("All evaluations across multistart")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("eval_log.png", dpi=300)

    # Final evaluation and export (use reduced design vector ordering)
    b, c, r_cruise, r_hover, c_charge = best_feasible["x_unscaled"]
    model_results, comparison_table = full_model_evaluation(
        b, c, r_cruise, r_hover, c_charge, p
    )
    write_results_to_excel(
        results_dict=model_results,
        comparison_list=comparison_table,
        mode="GWP",
    )


if __name__ == "__main__":
    main()