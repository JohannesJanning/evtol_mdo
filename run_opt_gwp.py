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

BOUNDS = [
    (6.0, 15.0),
    (1.0, 2.5),
    (0.6, 2.5),
    (0.5, 2),
    (200, 400),
    (1, 4),
]

N_STARTS = 30
LOCAL_RADIUS = 0.3
MAXITER = 1000
FTOL = 1e-6
EPS = 1e-2
SEED = 42


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
        results, _ = full_model_evaluation(*x, p)
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

    # Initial seed from GA
    x0_ga = [
        14.97352057,
        1.7603492,
        1.59805585,
        1.49530137,
        399.87531272,
        1.03502159,
    ]
    x0_scaled = scale(x0_ga, BOUNDS)

    starts_scaled = make_perturbations(x0_scaled, N_STARTS, radius=LOCAL_RADIUS)

    logging.info(
        "Starting multistart SLSQP (N=%d, radius=%.2f)...", N_STARTS, LOCAL_RADIUS
    )

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

    # Plot evaluations
    plt.figure()
    plt.plot(EVAL_LOG)
    plt.xlabel("Evaluation")
    plt.ylabel("Annual GWP (kg CO₂)")
    plt.title("All evaluations across multistart")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("eval_log.png", dpi=300)

    # Final evaluation and export
    b, c, r_cruise, r_hover, rho_bat, c_charge = best_feasible["x_unscaled"]
    model_results, comparison_table = full_model_evaluation(
        b, c, r_cruise, r_hover, rho_bat, c_charge, p
    )
    write_results_to_excel(
        results_dict=model_results,
        comparison_list=comparison_table,
        mode="GWP",
    )


if __name__ == "__main__":
    main()