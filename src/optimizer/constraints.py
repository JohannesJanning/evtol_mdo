import numpy as np

from src.parameters import model_parameters as parameters
from src.analysis.evaluation_model import full_model_evaluation


def constraint_functions(x):
    """
    Compute feasibility constraints for an eVTOL design.

    Parameters
    ----------
    x : array-like
        Design variables in the order:
        [wingspan b, chord c, cruise rotor radius, hover rotor radius,
         battery energy density rho_bat, charging C-rate].

    Returns
    -------
    constraints : list of float
        Constraint values, where positive values indicate feasibility.

        1. Wing length ≥ rotor spacing
        2. Vertiport span ≤ 15 m
        3. MTOM ≤ 5700 kg
        4–6. SPL (cruise, climb ≤ 67 dB; hover ≤ 77 dB)
        7–9. RPM (all ≤ 3000)
        10–11. Speeds (cruise, climb ≤ 129 m/s)

    Notes
    -----
    - If evaluation fails, returns a vector of large negative penalties.
    """
    b, c, r_cruise, r_hover, rho_bat, c_charge = x

    try:
        results, _ = full_model_evaluation(
            b, c, r_cruise, r_hover, rho_bat, c_charge, parameters
        )
    except Exception:
        # Hard penalty in case of model failure
        return [-1e3] * 11

    constraints = []

    # 1) Wing length constraint
    rotor_spacing = 2 * (3 * r_hover + 2 * parameters.d_rotors_space + parameters.r_fus_m)
    constraints.append(b - rotor_spacing)

    # 2) Vertiport width constraint (max 15 m)
    vertiport_span = 2 * (4 * r_hover + 2 * parameters.d_rotors_space + parameters.r_fus_m)
    constraints.append(15.0 - vertiport_span)

    # 3) MTOM constraint
    constraints.append(5700.0 - results["MASS MODEL"]["MTOM iterated"][0])

    # 4–6) SPL (noise)
    constraints.append(67.0 - results["NOISE MODEL"]["SPL in cruise"][0])
    constraints.append(67.0 - results["NOISE MODEL"]["SPL in climb"][0])
    constraints.append(77.0 - results["NOISE MODEL"]["SPL in hover"][0])

    # 7–9) RPM
    constraints.append(3000.0 - results["NOISE MODEL"]["RPM in cruise"][0])
    constraints.append(3000.0 - results["NOISE MODEL"]["RPM in climb"][0])
    constraints.append(3000.0 - results["NOISE MODEL"]["RPM in hover"][0])

    # 10–11) Speed
    constraints.append(129.0 - results["SPEED MODEL"]["Speed (cruise)"][0])
    constraints.append(129.0 - results["SPEED MODEL"]["Speed (climb)"][0])

    return constraints