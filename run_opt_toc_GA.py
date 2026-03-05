import logging
import time
import numpy as np
from scipy.optimize import differential_evolution
from tqdm.notebook import tqdm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from src.analysis.evaluation_model import full_model_evaluation
from src.parameters import model_parameters as p
from src.optimizer.constraints import constraint_functions

# --------------------------------------------------------------------------- #
# Configuration & Utility
# --------------------------------------------------------------------------- #
logging.getLogger().setLevel(logging.CRITICAL)
console = Console()

BOUNDS = [(6.0, 15.0), (1.0, 2.5), (0.6, 2.5), (0.5, 2.0), (200, 400), (1.0, 4.0)]
# Genetic Algorithm specific parameters
POPSIZE = 15      # Population multiplier (Total pop = POPSIZE * n_vars)
MAX_GEN = 50      # Maximum generations
TOL = 0.01        # Convergence tolerance

def unscale(x_scaled, bounds):
    return np.array([lo + xi * (hi - lo) for xi, (lo, hi) in zip(x_scaled, bounds)])

# --------------------------------------------------------------------------- #
# GA-Compatible Objective (Penalty Method)
# --------------------------------------------------------------------------- #
def ga_objective(x_scaled):
    """Objective function for GA with static penalty for constraints."""
    x = unscale(x_scaled, BOUNDS)
    try:
        # 1. Evaluate Physics/Economics
        results, _ = full_model_evaluation(*x, p)
        toc = results["ECONOMIC MODEL - COST"]["Total operating cost per trip"][0]
        
        # 2. Evaluate Constraints
        cons = constraint_functions(x)
        penalty = sum(abs(g) for g in cons if g < 0) * 1e5 # Heavy penalty for violations
        
        if toc is None or np.isnan(toc):
            return 1e9
            
        return toc + penalty
    except Exception:
        return 1e9

# --------------------------------------------------------------------------- #
# Main Routine
# --------------------------------------------------------------------------- #
def main():
    start_time = time.time()
    
    # 1. Academic Loading Dashboard
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, pulse_style="magenta"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("[bold magenta]Evolving eVTOL Design (Genetic Algorithm)...", total=MAX_GEN)
        
        # Callback to update the bar per generation
        def update_progress(xk, convergence):
            if not progress.finished:
                progress.update(task, advance=1)

        # 2. Differential Evolution (GA) Execution
        # We use 'scaled' space [(0,1)] for consistent mutation rates
        result = differential_evolution(
            ga_objective,
            bounds=[(0, 1)] * len(BOUNDS),
            strategy='best1bin',
            maxiter=MAX_GEN,
            popsize=POPSIZE,
            tol=TOL,
            mutation=(0.5, 1),
            recombination=0.7,
            callback=update_progress,
            disp=False
        )

    end_time = time.time()
    best_x = unscale(result.x, BOUNDS)
    best_toc = result.fun # Note: In GA with penalties, verify constraints manually at the end
    final_cons = constraint_functions(best_x)
    is_feasible = all(g >= -1e-5 for g in final_cons)

    # ------------------------------------------------------------------ #
    # Academic Results Presentation
    # ------------------------------------------------------------------ #
    console.print("\n")
    console.print(Panel.fit(
        f"[bold magenta]STOCHASTIC GLOBAL OPTIMIZATION: GENETIC ALGORITHM[/bold magenta]\n"
        f"Objective: Minimization of Total Operating Cost (ToC)",
        subtitle="Evolutionary Computing Results",
        border_style="magenta"
    ))

    # Variable Table
    var_names = ["Span [m]", "Chord [m]", "R_cruise [m]", "R_hover [m]", "Battery Density [Wh/kg]", "C-Rate [-]"]
    res_table = Table(title="Optimal Genome (Converged Design)", header_style="bold white on magenta", show_lines=True)
    res_table.add_column("Design Parameter", justify="left")
    res_table.add_column("Value", justify="right", style="bold green")
    res_table.add_column("Status", justify="center")

    for name, val, bound in zip(var_names, best_x, BOUNDS):
        if abs(val - bound[0]) < 1e-2: status = "[yellow]Lower Bound[/yellow]"
        elif abs(val - bound[1]) < 1e-2: status = "[yellow]Upper Bound[/yellow]"
        else: status = "[dim]Slack[/dim]"
        res_table.add_row(name, f"{val:.4f}", status)
    
    console.print(res_table)

    # Stats Table
    stats_table = Table(show_header=False, box=None)
    stats_table.add_row("Final Total Operating Cost", f"[bold green]{best_toc:.2f} €[/bold green]")
    stats_table.add_row("Feasibility Status", "[bold green]PASS[/bold green]" if is_feasible else "[bold red]FAIL[/bold red]")
    stats_table.add_row("Generations Evolved", str(result.nit))
    stats_table.add_row("Total Function Evaluations", str(result.nfev))
    stats_table.add_row("Total Wall-Clock Runtime", f"{end_time - start_time:.2f} s")
    stats_table.add_row("Avg. Time per Evaluation", f"{(end_time - start_time)/result.nfev:.4f} s")

    console.print(Panel(stats_table, title="Evolutionary Statistics", border_style="magenta", expand=False))

    # Rigor Margin
    console.print(f"[bold magenta]Min. Feasibility Margin:[/bold magenta] {np.min(final_cons):.8f}")

if __name__ == "__main__":
    main()