"""
Multistart SLSQP Optimization for Trip Operating Cost (ToC) Minimization
------------------------------------------------------------------------
This script runs a multistart SLSQP optimization to minimize the
total operating cost (ToC) per trip of an eVTOL aircraft concept.

It samples random starting points, evaluates feasibility,
tracks convergence, and reports the best feasible design.

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
    (6.0, 15.0),
    (1.0, 2.5),
    (0.6, 2.5),
    (0.5, 2.0),
    (200, 400),
    (1.0, 4.0),
]

N_RUNS = 10
MAXITER = 1000
FTOL = 1e-6
EPS = 0.018


# --------------------------------------------------------------------------- #
# Utility Functions
# --------------------------------------------------------------------------- #

def scale(x, bounds):
    """Scale physical parameters x into [0, 1] based on bounds."""
    return np.array([(xi - lo) / (hi - lo) for xi, (lo, hi) in zip(x, bounds)])


def unscale(x_scaled, bounds):
    """Unscale [0, 1] parameters back to physical values using bounds."""
    return np.array([lo + xi * (hi - lo) for xi, (lo, hi) in zip(x_scaled, bounds)])


def are_constraints_satisfied(constraints, tol=0.0):
    """Check feasibility with tolerance on constraints."""
    return all(g >= -tol for g in constraints)


# --------------------------------------------------------------------------- #
# Objective Tracker (no globals)
# --------------------------------------------------------------------------- #

class ObjectiveTracker:
    """
    Callable wrapper for the ToC objective function.
    Tracks evaluations and stores per-run logs.
    """

    def __init__(self, bounds, parameters):
        self.bounds = bounds
        self.parameters = parameters
        self.eval_log = []
        self.model_calls = 0

    def __call__(self, x_scaled):
        """Evaluate ToC at x_scaled (scaled design vector)."""
        self.model_calls += 1
        x = unscale(x_scaled, self.bounds)
        try:
            results, _ = full_model_evaluation(*x, self.parameters)
            toc = results["ECONOMIC MODEL - COST"]["Total operating cost per trip"][0]

            if toc is None or np.isnan(toc):
                logging.warning("Invalid evaluation (ToC missing or NaN).")
                return 1e6

            self.eval_log.append(toc)
            return toc  # minimize ToC
        except Exception as exc:
            logging.error("Exception during evaluation: %s", exc)
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
    """Run multistart SLSQP optimization for ToC minimization."""
    np.random.seed(42)
    all_results = []
    all_eval_logs = []
    best_result, best_x = None, None
    best_result_obj, best_runtime, best_run_idx = None, None, None

    start_time_total = time.time()

    for i in range(N_RUNS):
        tracker = ObjectiveTracker(BOUNDS, p)
        x0_rand = np.random.uniform(0, 1, size=len(BOUNDS))
        x0_phys = unscale(x0_rand, BOUNDS)

        logging.info("Run %d/%d | Init x0 = %s", i + 1, N_RUNS, np.round(x0_phys, 3))

        try:
            t0 = time.time()
            result = minimize(
                tracker,
                x0_rand,
                method="SLSQP",
                bounds=[(0, 1)] * len(BOUNDS),
                constraints=CONSTRAINTS,
                options={"disp": False, "maxiter": MAXITER, "ftol": FTOL, "eps": EPS},
            )
            run_time = time.time() - t0

            x_physical = unscale(result.x, BOUNDS)
            toc = result.fun
            valid = are_constraints_satisfied(constraint_functions(x_physical))

            logging.info("ToC = %.2f € | Feasible = %s", toc, valid)

            all_results.append((toc, x_physical, valid))
            all_eval_logs.append(tracker.eval_log)

            if valid and (best_result is None or toc < best_result):
                best_result = toc
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
        logging.info("Min ToC: %.2f €", best_result)
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
                mode="TOC",
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
    plt.ylabel("ToC [€]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("convergence_toc.png", dpi=300)

    # Boxplot
    toc_vals = [r[0] for r in all_results if r[2]]
    if toc_vals:
        plt.figure()
        plt.boxplot(toc_vals)
        plt.title("ToC Distribution over Multi-start Runs")
        plt.ylabel("Total Operating Cost per Trip [€]")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("toc_distribution.png", dpi=300)

    # Design space
    x_valid = np.array([r[1] for r in all_results if r[2]])
    toc_valid = [r[0] for r in all_results if r[2]]
    if x_valid.size > 0:
        plt.figure()
        plt.scatter(x_valid[:, 0], x_valid[:, 2], c=toc_valid, cmap="viridis")
        plt.xlabel("Wingspan b [m]")
        plt.ylabel("Cruise Radius [m]")
        plt.title("Design Space colored by ToC")
        plt.colorbar(label="ToC [€]")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("design_space_toc.png", dpi=300)


if __name__ == "__main__":
    main()