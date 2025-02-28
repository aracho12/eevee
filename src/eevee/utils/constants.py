# src/jolteon/utils/constants.py
from typing import Dict, List, Tuple

IDEAL_GAS_PARAMS: Dict[str, List] = {
    # 'name':[symmetrynumber,geometry,spin]
    'H2_g': [2, 'linear', 0],
    'N2_g': [2, 'linear', 0],
    'O2_g': [2, 'linear', 1.0],
    'H2O_g': [2, 'nonlinear', 0],
    'CO_g': [1, 'linear', 0],
    'CH4_g':[12,'nonlinear',0],
    'NH3_g':[3,'nonlinear',0],
    'CH3OH_g':[1,'nonlinear',0],
    'CH3CH2OH_g':[1,'nonlinear',0],
    'CO2_g':[2,'linear',0],
    'CH2O_g':[2,'nonlinear',0],
    'HCOOH_g':[1,'nonlinear',0],
    'HCOO_g':[1,'nonlinear',0],
    'CH2CH2_g':[4,'nonlinear',0],
    'CH3CHCH2_g':[1,'nonlinear',0], #propene
    'CH3CH2CHCH2_g':[1,'nonlinear',0], #1-butene
    'CH3CHCHCH3_g':[2,'nonlinear',0], #2-butene, ok for both trans and cis
    'CH3CH3CCH2_g':[2,'nonlinear',0], #isobutene
    'pe_g':[2,'linear',0], # fictitious gas molecule corresponding to proton electron pair
    'H_g':[2,'linear',0], # fictitious gas molecule corresponding to proton electron pair
    'C2H2_g':[2,'linear',0],
    'C2H4_g':[4,'nonlinear',0],
    'C2H6_g':[6,'nonlinear',0],
    'C3H6_g':[1,'nonlinear',0],
    'CH3COOH_g':[1,'nonlinear',0],
    'CH3CHO_g':[1,'nonlinear',0],
    'C5H4O2_g':[1,'nonlinear',0], #Added by N. Shan, KSU from NIST/CRC
    'C5H6O2_g':[1,'nonlinear',0], #Added by N. Shan, KSU from NIST/CRC
    'C5H6O_g':[1,'nonlinear',0], #Added by N. Shan, KSU from NIST/CRC
    'HCl_g':[1,'linear',0], #Added by M. Andersen, TUM from NIST
    'Cl2_g':[2,'linear',0], #Added by M. Andersen, TUM from NIST
    'HCOOCH3_g':[1,'nonlinear',0], #Added by A. Cho, SUNCAT from NIST
    'C3H8_g':[2,'nonlinear',0], #Added by A. Cho, SUNCAT from NIST
    'butadiene_g':[2, 'nonlinear', 0], #Added by A. Cho, SUNCAT from NIST
    'glyoxal_g':[2, 'nonlinear', 0], #Added by A. Cho, SUNCAT from NIST
    'CH3CH2COOH_g':[1, 'nonlinear', 0], #Added by A. Cho, SUNCAT from NIST
    }

class PhysicalConstants:
    """Physical constants in SI units"""
    h = 6.62607015e-34  # Planck constant
    c = 299792458       # Speed of light
    e = 1.602176634e-19 # Elementary charge
    kb = 1.380649e-23   # Boltzmann constant
    
CONSTANTS = PhysicalConstants()