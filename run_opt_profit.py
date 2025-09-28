"""
Partial Multistart SLSQP Optimization for Annual Profit Maximization
-------------------------------------------------------------------
This script runs a partial multistart SLSQP optimization around a
known near-optimum anchor to maximize the annual profit of an
eVTOL aircraft concept.

It perturbs selected variables (span, hover radius, battery density)
within specified windows, evaluates multiple starting points,
and reports the best feasible design.

Dependencies:
    numpy, scipy, matplotlib
    local modules: src.analysis.evaluation_model,
                   src.parameters.model_parameters,
                   src.optimizer.constraints

Outputs:
    - Console/log summary of optimization results
    - Convergence, distribution, and design space plots (saved as PNG)
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

BOUNDS = [
    (6.0, 15.0),   # 0: wingspan b
    (1.0, 2.5),    # 1
    (0.6, 2.5),    # 2
    (0.5, 2.0),    # 3: hover prop radius
    (200, 400),    # 4: battery density
    (1.0, 4.0),    # 5
]

N_RUNS = 10
MAXITER = 1000
FTOL = 1e-6
EPS = 0.018
BOUND_EPS = 1e-3  # nudge off exact bounds to help gradient methods

# Anchor point (near-optimum)
X_ANCHOR = np.array([11.51092996, 1.0, 2.5, 1.72424795, 263.54160762, 4.0])

# Indices to vary
VARY_IDX = [0, 3, 4]
WINDOWS = {0: 0.8, 3: 0.25, 4: 30.0}  # search ranges in physical units


# --------------------------------------------------------------------------- #
# Utility Functions
# --------------------------------------------------------------------------- #

def scale(x, bounds):
    """Scale physical parameters x into [0, 1] based on bounds."""
    return np.array([(xi - lo) / (hi - lo) for xi, (lo, hi) in zip(x, bounds)])


def unscale(x_scaled, bounds):
    """Unscale [0, 1] parameters back to physical values using bounds."""
    return np.array([lo + xi * (hi - lo) for xi, (lo, hi) in zip(x_scaled, bounds)])


def are_constraints_satisfied(constraints, tol=0.3):
    """Check feasibility with tolerance on constraints."""
    return all(g >= -tol for g in constraints)


def nudge_to_open_interval(x, bounds, eps=BOUND_EPS):
    """Shift values slightly off exact bounds for better gradient handling."""
    x = x.copy()
    for i, (lo, hi) in enumerate(bounds):
        if np.isclose(x[i], lo):
            x[i] = min(lo + eps, hi)
        elif np.isclose(x[i], hi):
            x[i] = max(hi - eps, lo)
    return x


def sample_partial_start(x_anchor, bounds, vary_idx, windows):
    """
    Generate a perturbed starting point around anchor.

    Returns
    -------
    x0_scaled : ndarray
        Initial guess in scaled space.
    x0_phys : ndarray
        Initial guess in physical space.
    """
    x0 = x_anchor.copy()
    for i in vary_idx:
        lo, hi = bounds[i]
        w = windows[i]
        x0[i] = np.random.uniform(x_anchor[i] - w, x_anchor[i] + w)
        x0[i] = np.clip(x0[i], lo, hi)
    x0 = nudge_to_open_interval(x0, bounds, eps=BOUND_EPS)
    return scale(x0, bounds), x0


# --------------------------------------------------------------------------- #
# Objective Tracker (no globals)
# --------------------------------------------------------------------------- #

class ObjectiveTracker:
    """
    Callable wrapper for the objective function that tracks
    evaluations and stores per-run logs.
    """

    def __init__(self, bounds, parameters):
        self.bounds = bounds
        self.parameters = parameters
        self.eval_log = []
        self.model_calls = 0

    def __call__(self, x_scaled):
        """Evaluate objective at x_scaled (scaled design vector)."""
        self.model_calls += 1
        x = unscale(x_scaled, self.bounds)
        try:
            results, _ = full_model_evaluation(*x, self.parameters)
            profit = results.get(
                "ECONOMIC MODEL - PROFIT", {}
            ).get("Total profit per year", (None,))[0]

            if profit is None or np.isnan(profit):
                logging.warning("Invalid evaluation (profit missing or NaN).")
                return 1e6

            self.eval_log.append(profit)
            return -profit  # SLSQP minimizes
        except Exception as exc:
            logging.error("Exception during evaluation: %s", exc)
            return 1e6


def scaled_constraints(x_scaled):
    """Constraint function in scaled space."""
    x = unscale(x_scaled, BOUNDS)
    return constraint_functions(x)


CONSTRAINTS = [
    {"type": "ineq", "fun": lambda x, i=i: scaled_constraints(x)[i] + 0.3}
    for i in range(11)
]


# --------------------------------------------------------------------------- #
# Main Routine
# --------------------------------------------------------------------------- #

def main():
    """Run partial multistart SLSQP optimization around anchor point."""
    np.random.seed(42)
    all_results = []
    all_eval_logs = []
    best_result, best_x = None, None
    best_result_obj, best_runtime, best_run_idx = None, None, None

    start_time_total = time.time()

    for i in range(N_RUNS):
        tracker = ObjectiveTracker(BOUNDS, p)
        x0_rand_scaled, x0_phys = sample_partial_start(
            X_ANCHOR, BOUNDS, VARY_IDX, WINDOWS
        )

        logging.info("Run %d/%d | Init x0 = %s", i + 1, N_RUNS, np.round(x0_phys, 3))

        try:
            t0 = time.time()
            result = minimize(
                tracker,
                x0_rand_scaled,
                method="SLSQP",
                bounds=[(0, 1)] * len(BOUNDS),
                constraints=CONSTRAINTS,
                options={"disp": False, "maxiter": MAXITER, "ftol": FTOL, "eps": EPS},
            )
            run_time = time.time() - t0

            x_physical = unscale(result.x, BOUNDS)
            profit = -result.fun
            constraint_vals = constraint_functions(x_physical)
            valid = are_constraints_satisfied(constraint_vals, tol=1.0)

            logging.info("Profit = %.2f € | Feasible = %s", profit, valid)

            all_results.append((profit, x_physical, valid))
            all_eval_logs.append(tracker.eval_log)

            if valid and (best_result is None or profit > best_result):
                best_result = profit
                best_x = x_physical
                best_result_obj = result
                best_runtime = run_time
                best_run_idx = i + 1

        except Exception as exc:
            logging.error("Error in run %d: %s", i + 1, exc)

    end_time_total = time.time()

    # ------------------------------------------------------------------ #
    # Results
    # ------------------------------------------------------------------ #
    if best_result is not None:
        logging.info("=== Best Design Found ===")
        logging.info("Run %d/%d", best_run_idx, N_RUNS)
        logging.info("Max Annual Profit: %.2f €", best_result)
        logging.info("x* = %s", np.round(best_x, 3))
        logging.info("Total wall time: %.2f s", end_time_total - start_time_total)

        if best_result_obj is not None:
            iters = getattr(best_result_obj, "nit", "N/A")
            nfev = getattr(best_result_obj, "nfev", "N/A")
            njev = getattr(best_result_obj, "njev", "N/A")
            logging.info("--- Optimizer stats ---")
            logging.info(
                "Iterations: %s | Function evals: %s | Gradient evals: %s",
                iters, nfev, njev,
            )
            logging.info("Runtime (best run): %.2f s", best_runtime)

        # Excel export
        try:
            b, c, r_cruise, r_hover, rho_bat, c_charge = best_x
            model_results, comparison_table = full_model_evaluation(
                b, c, r_cruise, r_hover, rho_bat, c_charge, p
            )
            write_results_to_excel(
                results_dict=model_results,
                comparison_list=comparison_table,
                mode="Profit",
            )
            logging.info("Excel export completed.")
        except Exception as exc:
            logging.error("Excel export failed: %s", exc)
    else:
        logging.error("No valid design found.")

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    # Convergence
    plt.figure()
    for i, log in enumerate(all_eval_logs):
        if log:
            plt.plot(log, label=f"Run {i+1}")
    plt.title("Convergence per Multi-Start Run")
    plt.xlabel("Evaluation")
    plt.ylabel("Annual Profit (€)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("convergence_profit.png", dpi=300)

    # Boxplot
    profits = [r[0] for r in all_results if r[2]]
    if profits:
        plt.figure()
        plt.boxplot(profits)
        plt.title("Annual Profit Distribution (Feasible Runs)")
        plt.ylabel("Annual Profit (€)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("profit_distribution.png", dpi=300)

    # Design space
    x_valid = np.array([r[1] for r in all_results if r[2]])
    profits_valid = [r[0] for r in all_results if r[2]]
    if x_valid.size > 0:
        plt.figure()
        plt.scatter(x_valid[:, 0], x_valid[:, 3], c=profits_valid, cmap="viridis")
        plt.xlabel("Wingspan b [m]")
        plt.ylabel("Hover Prop Radius [m]")
        plt.title("Design Space colored by Annual Profit")
        plt.colorbar(label="Annual Profit (€)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("design_space_profit.png", dpi=300)


if __name__ == "__main__":
    main()