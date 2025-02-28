from typing import List, Union, Tuple
import numpy as np
from scipy import constants
from ase.thermochemistry import HarmonicThermo, IdealGasThermo
import catmap
import ast

from ..utils.constants import IDEAL_GAS_PARAMS, CONSTANTS

def cm1_to_ev(freq_cm1: float) -> float:
    """Convert frequency from cm^-1 to eV"""
    return freq_cm1 * CONSTANTS.c * CONSTANTS.h * 100 / CONSTANTS.e

def safe_eval_frequencies(freq_string: Union[str, List]) -> List[float]:
    """Safely evaluate frequency string to list"""
    try:
        if isinstance(freq_string, str):
            return ast.literal_eval(freq_string)
        return freq_string if isinstance(freq_string, list) else []
    except:
        return []

class ThermoCorrector:
    def __init__(self, row):       
        self.row = row
        self.frequencies = row['frequencies']
        self.species_name = row['species_name']
        self.site_name = row['site_name']
        
    def get_free_energy(self, 
                       temperature: float = 300, 
                       fugacity: float = 1e5) -> Tuple:
        """
        Calculate free energy corrections for a species
        
        Returns:
        --------
        tuple : (status, ZPE, Cp, H, dS, TS, F)
        """
        is_gas = self.site_name == 'gas'

        # Convert frequencies from cm^-1 to eV
        vib_energies = [cm1_to_ev(freq) for freq in self.frequencies if freq != 0]

        if is_gas:
            return self._calculate_gas_phase(vib_energies, temperature, fugacity)
        return self._calculate_surface_phase(vib_energies, temperature)

    def _calculate_gas_phase(self, vib_energies: List[float], 
                           temperature: float, 
                           fugacity: float) -> Tuple:
        """Calculate gas phase thermodynamic corrections"""
        species_name = self.species_name

        if species_name == 'H2_ref':
            species_name = 'H2'
        gas_name_g = f"{species_name}_g"
        
        if gas_name_g in ['H_g', 'ele_g']:
            return 'gas', 0, 0, 0, 0, 0, 0
        
        # Get gas parameters
        gpars = IDEAL_GAS_PARAMS.get(gas_name_g, [1, 'nonlinear', 0])
        symmetry, geometry, spin = gpars[:3]
        
        try:
            atoms = catmap.molecule(species_name)
        except:
            print(f"Warning: {species_name} not found in catmap.molecule. Using a dummy Atoms object.")
            atoms = None
        
        # Calculate thermodynamic corrections using IdealGasThermo
        therm = IdealGasThermo(
            vib_energies=vib_energies,
            geometry=geometry,
            atoms=atoms,
            symmetrynumber=symmetry,
            spin=spin
        )

        status = 'gas'
        ZPE = therm.get_ZPE_correction()
        H = therm.get_enthalpy(temperature, verbose=False)
        Cp = H - ZPE
        dS = therm.get_entropy(temperature, fugacity, verbose=False)
        TS = temperature * dS
        F = H - TS
        return status, ZPE, Cp, H, dS, TS, F
    
    def _calculate_surface_phase(self, vib_energies: List[float], 
                               temperature: float) -> Tuple:
        """Calculate surface phase thermodynamic corrections"""
        therm = HarmonicThermo(vib_energies)
        status = 'ads'
        ZPE = therm.get_ZPE_correction()
        H = therm.get_internal_energy(temperature=temperature, verbose=False)
        F = therm.get_helmholtz_energy(temperature=temperature, verbose=False)
        dS = therm.get_entropy(temperature=temperature, verbose=False)
        Cp = H - ZPE
        TS = temperature * dS

        return status, ZPE, Cp, H, dS, TS, F       
