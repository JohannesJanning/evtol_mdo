from pyxdsm.XDSM import XDSM, OPT, FUNC
import os
xdsm = XDSM()

xdsm.add_system('Opt', OPT, r"\text{Optimizer}")
xdsm.add_system('Aero', FUNC, r"\text{Aerodynamics}")
xdsm.add_system('FlightMech', FUNC, r"\text{Balance \& Performance}")
xdsm.add_system('Time', FUNC, r"\text{Time Model}")
xdsm.add_system('Energy', FUNC, r"\text{Energy Model}")
xdsm.add_system('Mass', FUNC, r"\text{Mass Model}")
xdsm.add_system('Env', FUNC, r"\text{Environment Model}")
xdsm.add_system('Economic', FUNC, r"\text{Economic Model}")
xdsm.add_system('FoM', FUNC, r"\text{FoM Model}")

xdsm.connect('Opt', 'Aero', r"c, b, R_{lifter}, R_{pusher}")
xdsm.connect('Opt', 'FlightMech', r"c, b, R_{lifter}, R_{pusher}")
xdsm.connect('Opt', 'Mass', r"c, b, R_{lifter}, R_{pusher}")
xdsm.connect('Opt', 'Env', r"C_{charge}")
xdsm.connect('Opt', 'Economic', r"C_{charge}")



xdsm.connect('Aero', 'FlightMech', r"C_L, C_D")
xdsm.connect('FlightMech', 'Energy', r"\{P\}_{hover/climb/cruise}")
xdsm.connect('FlightMech', 'Mass', r"\{P\}_{hover/cruise}, V_{cruise}")
xdsm.connect('FlightMech', 'Time', r"RoC, V_{climb/cruise}")
xdsm.connect('Time', 'Energy', r"\{t\}_{hover/climb/cruise}")
xdsm.connect('Energy', 'Mass', r"\{E\}_{trip/res}")
xdsm.connect('Mass', 'FlightMech', r"m_{MTO}") 


xdsm.connect('Mass', 'Economic', r"\{m\}_{MTO/empty/bat}")
xdsm.connect('Mass', 'Env', r"m_{bat}")
xdsm.connect('Energy', 'Economic', r"\{E\}_{trip/bat}")
xdsm.connect('Energy', 'Env', r"E_{trip}")
xdsm.connect('Time', 'Economic', r"t_{trip}")
xdsm.connect('Time', 'FoM', r"t_{trip}")
xdsm.connect('Env', 'FoM', r"CO_2e_{trip}")
xdsm.connect('Economic', 'FoM', r"\mathcal{C}_{TOC}")


xdsm.connect('Env', 'Opt', r"GWP_{ops}")
xdsm.connect('Economic', 'Opt', r"\mathcal{C}_{TOC}, \Pi_{\text{ops}}")
xdsm.connect('FoM', 'Opt', r"\text{FoM}")


xdsm.write('xdsm_comparative_study', build=True, cleanup=True)

for ext in ['.tex', '.aux', '.log', '.tikz']:

    try:

        os.remove('xdsm_comparative_study' + ext)

    except FileNotFoundError:

        pass