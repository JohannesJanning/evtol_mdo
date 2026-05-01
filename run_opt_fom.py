"""
Multistart SLSQP Optimization for eVTOL FoM Maximization
--------------------------------------------------------
This script runs a multistart SLSQP optimization to maximize the
Figure of Merit (FoM) of an eVTOL aircraft concept. It repeatedly
samples random starting points, records convergence, and identifies
the best feasible design.

Dependencies:
    numpy, scipy, matplotlib
    local modules: src.analysis.evaluation_model,
                   src.parameters.model_parameters,
                   src.optimizer.constraints

Outputs:
    - Console/log summary of optimization results
    - Convergence and distribution plots (saved as PNG)
    - Excel export of the best design (via write_results_to_excel)
"""

import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from src.analysis.evaluation_model import full_model_evaluation, write_results_to_excel
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

MODEL_CALLS = 0
BOUNDS = [
    (6.0, 15.0),  # wingspan b
    (1.0, 2.5),   # chord c
    (0.6, 2.5),   # cruise prop radius
    (0.5, 2.0),   # hover prop radius
    (1.0, 4.0),   # battery charging rate c_charge
]

N_RUNS = 10
MAXITER = 1000
FTOL = 1e-8
EPS = 1e-2
# Initial anchor (physical units). Battery energy density is fixed in parameters.
# Order: [b, c, R_cruise, R_hover, c_charge]
X_INITIAL = np.array([14.648, 1.0, 1.215, 1.586, 1.154])
BOUND_EPS = 1e-2
# Indices to perturb around the anchor and optional absolute windows
VARY_IDX = [0, 2, 3, 4]
WINDOWS = {0: 0.5, 2: 0.2, 3: 0.2, 4: 0.2}


# --------------------------------------------------------------------------- #
# Utility Functions
# --------------------------------------------------------------------------- #

def scale(x, bounds):
    """Scale physical parameters x into [0, 1] based on bounds."""
    return np.array([(xi - low) / (high - low) for xi, (low, high) in zip(x, bounds)])


def unscale(x_scaled, bounds):
    """Unscale [0, 1] parameters back to physical values using bounds."""
    return np.array([low + xi * (high - low) for xi, (low, high) in zip(x_scaled, bounds)])


def nudge_to_open_interval(x, bounds, eps=BOUND_EPS):
    x = x.copy()
    for i, (lo, hi) in enumerate(bounds):
        if np.isclose(x[i], lo):
            x[i] = min(lo + eps, hi)
        elif np.isclose(x[i], hi):
            x[i] = max(hi - eps, lo)
    return x


def sample_partial_start(x_anchor, bounds, vary_idx=None, windows=None, rel_window=0.1):
    if vary_idx is None:
        vary_idx = list(range(len(bounds)))
    x0 = x_anchor.copy()
    if windows is None:
        windows = {i: rel_window * abs(float(x_anchor[i])) for i in vary_idx}
    for i in vary_idx:
        lo, hi = bounds[i]
        w = windows.get(i, 0.0)
        x0[i] = np.random.uniform(max(lo, x_anchor[i] - w), min(hi, x_anchor[i] + w))
    x0 = nudge_to_open_interval(x0, bounds, eps=BOUND_EPS)
    return scale(x0, bounds), x0


# --------------------------------------------------------------------------- #
# Objective and Constraints
# --------------------------------------------------------------------------- #

def scaled_objective(x_scaled):
    """
    Objective function in scaled space.

    Returns
    -------
    float
        Negative FoM (since SLSQP minimizes).
    """
    global MODEL_CALLS
    MODEL_CALLS += 1
    x = unscale(x_scaled, BOUNDS)

    try:
        # x ordering: b, c, R_cruise, R_hover, c_charge
        b, c, r_cruise, r_hover, c_charge = x
        results, comparison_table = full_model_evaluation(b, c, r_cruise, r_hover, c_charge, p)
        fom = next(
            (entry["FoM"] for entry in comparison_table if "eVTOL" in entry["Mode (LF)" ]),
            None,
        )
        if fom is None or np.isnan(fom):
            logging.warning("Invalid evaluation (FoM missing or NaN).")
            return 1e6

        return -fom
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
# Main Routine
# --------------------------------------------------------------------------- #

def main():
    """Run multistart SLSQP optimization and analyze results."""
    all_results = []
    all_eval_logs = []
    best_result = None
    best_x = None
    best_run_num = None
    best_result_obj = None
    best_runtime = None

    start_time_total = time.time()

    for run_idx in range(N_RUNS):
        eval_log = []  # local convergence log per run
        # sample a randomized start near the anchor
        x0_rand, x0_phys = sample_partial_start(X_INITIAL, BOUNDS, VARY_IDX, WINDOWS)

        logging.info("Run %d/%d | Init x0 = %s", run_idx + 1, N_RUNS, np.round(x0_phys, 3))

        t0 = time.time()
        result = minimize(
            scaled_objective,
            x0_rand,
            method="SLSQP",
            bounds=[(0, 1)] * len(BOUNDS),
            constraints=CONSTRAINTS,
            options={"disp": False, "maxiter": MAXITER, "ftol": FTOL, "eps": EPS},
        )
        run_time = time.time() - t0

        x_phys = unscale(result.x, BOUNDS)
        fom = -result.fun
        valid = all(v >= 0 for v in constraint_functions(x_phys))

        logging.info("Result: FoM = %.3f | Constraints ok = %s", fom, valid)

        all_results.append((fom, x_phys, valid))
        all_eval_logs.append(eval_log.copy())

        if valid and (best_result is None or fom > best_result):
            best_result, best_x = fom, x_phys
            best_run_num = run_idx + 1
            best_result_obj, best_runtime = result, run_time

    end_time_total = time.time()

    # ------------------------------------------------------------------ #
    # Results
    # ------------------------------------------------------------------ #
    if best_result is None or best_x is None:
        logging.error("No valid design found.")
        return

    logging.info("=== Best Design Found ===")
    logging.info("Best Run: %d", best_run_num)
    logging.info("Max FoM = %.3f", best_result)
    logging.info("x* = %s", np.round(best_x, 3))
    logging.info("Total runtime (all runs) = %.1f s", end_time_total - start_time_total)

    if best_result_obj is not None:
        iters = getattr(best_result_obj, "nit", "N/A")
        nfev = getattr(best_result_obj, "nfev", "N/A")
        njev = getattr(best_result_obj, "njev", "N/A")
        logging.info("--- Optimizer stats (best run) ---")
        logging.info("Iterations: %s | Function evals: %s | Gradient evals: %s", iters, nfev, njev)
        logging.info("Runtime (best run): %.2f s", best_runtime)

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    # Convergence logs
    plt.figure()
    for i, log in enumerate(all_eval_logs):
        if log:
            plt.plot(log, label=f"Run {i+1}")
    plt.title("Convergence per Multi-Start Run")
    plt.xlabel("Evaluation")
    plt.ylabel("FoM")
    if any(log for log in all_eval_logs):
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("convergence.png", dpi=300)

    # Boxplot of valid FoMs
    foms_valid = [r[0] for r in all_results if r[2]]
    if foms_valid:
        plt.figure()
        plt.boxplot(foms_valid)
        plt.title("FoM Distribution over Multi-start Runs")
        plt.ylabel("FoM")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("fom_distribution.png", dpi=300)

    # Design space scatter
    x_valid = np.array([r[1] for r in all_results if r[2]])
    if x_valid.size > 0:
        plt.figure()
        plt.scatter(x_valid[:, 0], x_valid[:, 2], c=foms_valid, cmap="viridis")
        plt.xlabel("Wingspan b [m]")
        plt.ylabel("Cruise Radius [m]")
        plt.title("Design Space colored by FoM")
        plt.colorbar(label="FoM")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("design_space.png", dpi=300)

    # ------------------------------------------------------------------ #
    # Excel Export
    # ------------------------------------------------------------------ #
    if best_x is not None:
        try:
            # best_x ordering now: b, c, R_cruise, R_hover, c_charge
            b, c, r_cruise, r_hover, c_charge = best_x
            model_results, comparison_table = full_model_evaluation(
                b, c, r_cruise, r_hover, c_charge, p
            )
            write_results_to_excel(
                results_dict=model_results,
                comparison_list=comparison_table,
                mode="FoM",
            )
            logging.info("Excel export completed.")
        except Exception as exc:
            logging.error("Excel export skipped due to error: %s", exc)


if __name__ == "__main__":
    main()