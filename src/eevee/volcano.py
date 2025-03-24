from typing import List, Union, Tuple
import numpy as np
from scipy import constants
from ase.thermochemistry import HarmonicThermo, IdealGasThermo
import catmap
import ast
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.font_manager as fm
sqlite3.register_adapter(np.int64, lambda val: int(val))
sqlite3.register_adapter(np.int32, lambda val: int(val))
helvetica_bold_path = '/Users/aracho/Dropbox/Resources/Fonts-downloaded/Helvetica/Helvetica-Bold.ttf'

# 폰트 등록
try:
    fm.fontManager.addfont(helvetica_bold_path)
    helvetica_bold_prop = fm.FontProperties(fname=helvetica_bold_path)
    print(f"Helvetica Bold font registered successfully from: {helvetica_bold_path}")
    helvetica_available = True
except Exception as e:
    print(f"Error loading Helvetica Bold font: {e}")
    helvetica_available = False

reaction_list = {
    1: 'CO2_g + * + H_g + ele_g -> COOH*',
    2: 'COOH* + H_g + ele_g -> CO* + H2O_g',
    3: 'CO* + H_g + ele_g -> COH*',
    4: 'CO* + H_g + ele_g -> CHO*',
    5: 'COH* + H_g + ele_g -> C* + H2O_g', 
    6: 'C* + H_g + ele_g -> CH*',
    7: 'CH* + H_g + ele_g -> CH2*',
    8: 'CH2* + H_g + ele_g -> CH3*',
    9: 'CH3* + H_g + ele_g -> CH4_g + *',
   # 10: 'CHO* + H_g + ele_g -> CH2O*',
    11: 'H_g + ele_g -> H*',
    12: 'H* + H_g + ele_g -> H2_g',
   # 13: 'H* + H* -> H2_g',
   14: 'CO2_g + H_g + ele_g -> CH4_g',
   15: 'CO_g -> CO*', 
   16: 'CO* -> CO_g', #CO desorption 
   17: 'CO_g + H_g + ele_g -> CHO*',
}

DB_NAME = '/Users/aracho/Library/CloudStorage/GoogleDrive-choara@stanford.edu/My Drive/CRADA-Projects/00_CO2R_DB/co2r-drive.db'

conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

# Query to get frequencies data with metadata
query = """
SELECT 
    f.reference_id,
    r.reference,
    sp.species_name,
    st.status,
    f.frequencies
FROM Frequencies f
JOIN Reference r ON f.reference_id = r.reference_id
JOIN Species sp ON f.species_id = sp.species_id
JOIN Status st ON f.status_id = st.status_id

"""
frequencies_df = pd.read_sql_query(query, conn)

#display_df(frequencies_df, "Frequencies Table")


def calculate_binding_energy(co_binding: float, species: str, scaling_relation=None) -> float:
    """
    Calculate binding energy of a species using scaling relations with CO binding energy
    
    Parameters:
    -----------
    co_binding : float
        CO binding energy (eV)
    species : str
        Name of species (e.g., 'C*', 'CH*', etc.)
        
    Returns:
    --------
    float
        Binding energy of the species (eV)
    """
    
    # Remove '*' from species name for matching with dictionary
    species = species.strip('*')
    species = species + '*'  # Add back '*' for consistent lookup
    
    if species not in scaling_relation:
        raise ValueError(f"No scaling relation found for species: {species}")
        
    # Get scaling coefficients (a, b) from the dictionary
    if scaling_relation is None:
        a, b = scaling_relation[species]
    else:
        a, b = scaling_relation[species]
    
    # Calculate binding energy using linear scaling relation: E = a*E_CO + b
    binding_energy = a * co_binding + b
    
    return binding_energy

# Example usage:
#species = 'CO*'
#co_binding = -0.5  # Example CO binding energy
#ch_binding = calculate_binding_energy(co_binding, species)
#print(f"{species} binding energy: {ch_binding:.2f} eV")


def get_species_frequencies(species: str, frequencies_df: pd.DataFrame) -> List[float]:
    """
    Get frequencies for a given species from the frequencies database
    
    Parameters:
    -----------
    species : str
        Name of species (e.g., 'CO*', 'CO_g', 'H2O', etc.)
    frequencies_df : pd.DataFrame
        DataFrame containing frequencies data with columns:
        ['reference_id', 'reference', 'species_name', 'status', 'frequencies']
        
    Returns:
    --------
    List[float]
        List of frequencies in cm^-1. Returns empty list if not found.
    """
    frequencies_df = pd.read_sql_query(query, conn)
    # Determine status based on species name
    if '*' in species:
        status = 'ads'
        species = species.strip('*')
    elif '_g' in species:
        status = 'gas'
        species = species.replace('_g', '')
    else:
        return []
    
    # Filter DataFrame for matching species and status
    matching_rows = frequencies_df[
        (frequencies_df['species_name'] == species) & 
        (frequencies_df['status'] == status)
    ]
    
    if matching_rows.empty:
        print(f"Warning: No frequencies found for {species} ({status})")
        return []
    
    # Get frequencies from the first matching row
    try:
        frequencies = ast.literal_eval(matching_rows.iloc[0]['frequencies'])
        return [float(f) for f in frequencies if f != 0]  # Convert to float and remove zero frequencies
    except:
        print(f"Error: Could not parse frequencies for {species}")
        return []

# # Example usage:
# species='CH4_g'
# species_freq = get_species_frequencies(species, frequencies_df)
# print(f"{species} frequencies: {species_freq}")


from typing import Dict, List, Tuple

class PhysicalConstants:
    """Physical constants in SI units"""
    h = 6.62607015e-34  # Planck constant
    c = 299792458       # Speed of light
    e = 1.602176634e-19 # Elementary charge
    kb = 1.380649e-23   # Boltzmann constant
    
CONSTANTS = PhysicalConstants()

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
def calculate_thermo_corrections(species: str, 
                               frequencies: List[float], 
                               temperature: float = 298, 
                               fugacity: float = 1e5) -> Dict[str, float]:
    
    

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


    """
    Calculate thermodynamic corrections for a given species
    
    Parameters:
    -----------
    species : str
        Species name (e.g., 'CO*', 'CO_g')
    frequencies : List[float]
        List of vibrational frequencies in cm^-1
    temperature : float
        Temperature in K (default: 298K)
    fugacity : float
        Fugacity in Pa (default: 1e5 Pa), only used for gas phase
        
    Returns:
    --------
    Dict[str, float]
        Dictionary containing thermodynamic corrections:
        {'ZPE', 'Cp', 'H', 'S', 'TS', 'G'}
    """

    #print(f"Calculating thermo corrections for {species} at {temperature} K and fugacity {fugacity} Pa")
    
    # Determine if species is gas or adsorbed
    if '*' in species:
        status = 'ads'
        species_name = species.strip('*')
    elif '_g' in species:
        status = 'gas'
        species_name = species.replace('_g', '')
    else:
        raise ValueError(f"Invalid species format: {species}")
        
    # Convert frequencies from cm^-1 to eV
    vib_energies = [cm1_to_ev(freq) for freq in frequencies if freq != 0]
    
    if status == 'gas':
        # Special case for H_g and ele_g
        if species in ['H_g', 'ele_g']:
            return {
                'ZPE': 0, 'Cp': 0, 'H': 0,
                'S': 0, 'TS': 0, 'F': 0
            }
            
        # Get gas parameters
        gas_params = IDEAL_GAS_PARAMS.get(species, [1, 'nonlinear', 0])
        symmetry, geometry, spin = gas_params[:3]
        
        try:
            atoms = catmap.molecule(species_name)
        except:
            print(f"Warning: {species_name} not found in catmap.molecule. Using a dummy Atoms object.")
            atoms = None        
        
        # Calculate gas phase corrections
        thermo = IdealGasThermo(
            vib_energies=vib_energies,
            geometry=geometry,
            atoms=atoms,
            symmetrynumber=symmetry,
            spin=spin
        )
        
        ZPE = thermo.get_ZPE_correction()
        H = thermo.get_enthalpy(temperature, verbose=False)
        Cp = H - ZPE
        S = thermo.get_entropy(temperature, fugacity, verbose=False)
        TS = temperature * S
        F = H - TS
        
    else:  # status == 'ads'
        # Calculate surface phase corrections
        thermo = HarmonicThermo(vib_energies)
        
        ZPE = thermo.get_ZPE_correction()
        H = thermo.get_internal_energy(temperature=temperature, verbose=False)
        Cp = H - ZPE
        S = thermo.get_entropy(temperature=temperature, verbose=False)
        TS = temperature * S
        F = thermo.get_helmholtz_energy(temperature=temperature, verbose=False)
    
    return {
        'ZPE': ZPE,
        'Cp': Cp,
        'H': H,
        'S': S,
        'TS': TS,
        'F': F
    }

# Example usage:
# fugacity_dict = {
#     'H2_g': 101325,
#     'CO2_g': 101325,
#     'CO_g': 5562,
#     'HCOOH_g': 2,
#     'CH3OH_g': 6079,
#     'H2O_g': 3534,
#     'CH4_g': 20467,
# }
# species='CH3-H-ele*'
# temperature=413
# fugacity = fugacity_dict.get(species, 1e5)
# frequencies = get_species_frequencies(species, frequencies_df)
# #corrections_ads = calculate_thermo_corrections('CO2_g', frequencies, temperature=300)
# corrections_gas = calculate_thermo_corrections(species, frequencies, temperature=temperature, fugacity=fugacity)
# #print("Adsorbed CO corrections:", corrections_ads)

# print(f"\nGas phase {species} corrections at {temperature} K and {fugacity} Pa:")
# print("--------------------------------")
# for key, value in corrections_gas.items():
#     print(f"{key:>4}: {value:>10.4f} eV")


def load_solvation_data(db_path: str = '/Users/aracho/Library/CloudStorage/GoogleDrive-choara@stanford.edu/My Drive/CRADA-Projects/00_CO2R_DB/co2r-drive.db') -> pd.DataFrame:
    """
    Load solvation data from SQLite database
    
    Parameters:
    -----------
    db_path : str
        Path to SQLite database file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing solvation data
    """
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        
        # Query solvation data
        query = """
        SELECT * FROM SolvationEffectView
        """
        
        # Load data into DataFrame
        solvation_df = pd.read_sql_query(query, conn)
        
        # Close connection
        conn.close()
        
        return solvation_df
    
    except Exception as e:
        print(f"Error loading solvation data: {str(e)}")
        return pd.DataFrame()

# Example usage:
# Load solvation data
solvation_df = load_solvation_data()

# Print first few rows to verify
# print("Loaded solvation data:")
# print(solvation_df.head())

def get_solvation_energy(species: str, 
                        solvation_df: pd.DataFrame, 
                        approach: str = 'explicit',
                        manual_solv=None) -> float:
    """
    Get solvation energy for a given species
    
    Parameters:
    -----------
    species : str
        Species name (e.g., 'CO*', 'CO_g')
    solvation_df : pd.DataFrame
        DataFrame containing solvation data
    approach : str
        'explicit' or 'implicit' (default: 'explicit')
        
    Returns:
    --------
    float
        Solvation energy in eV. Returns 0.0 if not found.
    """

    # Clean species name
    if '*' in species:
        status = 'ads'
        species = species.strip('*')
        #print(species)
    elif '_g' in species:
        status = 'gas'
        species = species.replace('_g', '')
    else:
        return 0.0
    
    if manual_solv is not None:
        
        if species in manual_solv.keys():
            #print(f"Using manual solvation energy for {species}: {manual_solv[species]} eV")
            return manual_solv[species]

    # Filter DataFrame
    mask = (
        (solvation_df['species_name'] == species) & 
        (solvation_df['status'] == status) &
        (solvation_df['approach'] == approach)
    )
    matching_rows = solvation_df[mask]
    
    if matching_rows.empty:
        #print(f"Warning: No solvation energy found for {species} ({status}, {approach})")
        return 0.0
    
    return float(matching_rows.iloc[0]['solvation_energy'])


def convert_deltaG_CO_ads_to_E_ads(dG_CO_ads: float, 
                                  temperature: float = 298, 
                                  fugacity: Dict[str, float] = {},
                                  manual_solv=None) -> float:
    co_ads_frequencies = get_species_frequencies('CO*', frequencies_df)
    co_ads_thermo_corr = calculate_thermo_corrections('CO*', co_ads_frequencies, 
                                                    temperature, fugacity=1e5)
    co_ads_solv_energy = get_solvation_energy('CO*', solvation_df, approach='explicit',
                                                manual_solv=manual_solv)
    dF_CO_ads = co_ads_thermo_corr['F'] + co_ads_solv_energy

    co_gas_frequencies = get_species_frequencies('CO_g', frequencies_df)
    co_gas_thermo_corr = calculate_thermo_corrections('CO_g', co_gas_frequencies, 
                                                    temperature, fugacity=5562)
    co_gas_solv_energy = get_solvation_energy('CO_g', solvation_df, approach='explicit',
                                                manual_solv=manual_solv)
    dF_CO_gas = co_gas_thermo_corr['F'] + co_gas_solv_energy

    dE_CO_ads = dG_CO_ads - dF_CO_ads + dF_CO_gas

    return dE_CO_ads

def calculate_reaction_energy(reaction: str, dG_CO_ads: float, 
                            temperature: float = 298, 
                            fugacity: Dict[str, float] = {},
                            manual_solv=None,
                            scaling_relation=None) -> float:
    """
    Calculate reaction free energy (ΔG) for a given reaction
    
    Parameters:
    -----------
    reaction : str
        Reaction equation (e.g., 'CO2_g + * + H_g + ele_g -> COOH*')
    co_binding : float
        CO binding energy (eV)
    temperature : float
        Temperature in K (default: 298K)
    fugacity : float
        Fugacity in Pa (default: 1e5 Pa)
        
    Returns:
    --------
    float
        Reaction free energy (ΔG) in eV
    """  

    dE_CO_ads = convert_deltaG_CO_ads_to_E_ads(dG_CO_ads, temperature, fugacity, manual_solv)


    E_form_gas_dict = {
        'H2_g': 0.0,
        'CO_g': 0.0,
        'CH4_g': -2.8047879999999594,
        'CO2_g': -0.28416299999980765,
        'H2O_g':0.0,
        'H_g': 0.0,
        'ele_g': 0.0,
    }
    if fugacity is {} or None:
        for gas in E_form_gas_dict.keys():
            fugacity[gas] = 1e5
    else:
        for gas in fugacity.keys():
            if gas not in E_form_gas_dict.keys():
                fugacity[gas] = 1e5
    # Split reaction into reactants and products
    reactants, products = reaction.split(' -> ')
    reactants = [r.strip() for r in reactants.split(' + ')]
    products = [p.strip() for p in products.split(' + ')]
    
    # Initialize energies
    G_reactants = 0
    G_products = 0
    
    # Calculate G for reactants
    for species in reactants:
        if species in ['*', 'H_g', 'ele_g']:
            continue  # These species have G = 0
            
        # Get DFT energy from scaling relations

        if '*' in species:
            E_dft = calculate_binding_energy(dE_CO_ads, species, scaling_relation)
            fugacity[species] = 1e5
        else:
            E_dft = E_form_gas_dict[species]
            
        # Get frequencies and calculate thermal corrections
        frequencies = get_species_frequencies(species, frequencies_df)
        thermo_corr = calculate_thermo_corrections(species, frequencies, 
                                                 temperature, fugacity[species])
        
        # Get solvation energy
        solv_energy = get_solvation_energy(species, solvation_df, approach='explicit',
                                          manual_solv=manual_solv)
        
        # Sum up the energies: G = E_dft + F + E_solv
        G = E_dft + thermo_corr['F'] + solv_energy
        G_reactants += G
        
    # Calculate G for products
    for species in products:
        if species in ['*', 'H_g', 'ele_g']:
            continue
            
        # Get DFT energy from scaling relations
        if '*' in species:
            E_dft = calculate_binding_energy(dE_CO_ads, species, scaling_relation)
            fugacity[species] = 1e5
        else:
            E_dft = E_form_gas_dict[species]
            
        # Get frequencies and calculate thermal corrections
        frequencies = get_species_frequencies(species, frequencies_df)
        thermo_corr = calculate_thermo_corrections(species, frequencies, 
                                                 temperature, fugacity[species])
        
        # Get solvation energy
        solv_energy = get_solvation_energy(species, solvation_df, approach='explicit',
                                          manual_solv=manual_solv)
        
        # Sum up the energies: G = E_dft + F + E_solv
        G = E_dft + thermo_corr['F'] + solv_energy
        G_products += G
    
    # Calculate ΔG
    delta_G = G_products - G_reactants

    if reaction == 'CO2_g + H_g + ele_g -> CH4_g':
        delta_G = delta_G/8
        #print(f"CO2R reduction energy: {delta_G} eV")
    
    return delta_G

# Example usage:
# reaction = 'CO_g + H_g + ele_g -> COH*'
# co_binding = -0.6  # Example CO binding energy ∆G_CO
# temperature = 312  # Example temperature
# print(fugacity_dict)
# delta_G = calculate_reaction_energy(reaction, co_binding, temperature, fugacity_dict)
# print(f"Reaction: {reaction}")
# print(f"Reaction energy: {delta_G:.2f} eV ({co_binding:.2f} eV ∆G_CO at {temperature} K)")


def calculate_reaction_energy_before(reaction: str, dE_CO_ads: float, 
                            temperature: float = 298, 
                            fugacity: Dict[str, float] = {},
                            manual_solv=None,
                            scaling_relation=None) -> float:
    """
    Calculate reaction free energy (ΔG) for a given reaction
    
    Parameters:
    -----------
    reaction : str
        Reaction equation (e.g., 'CO2_g + * + H_g + ele_g -> COOH*')
    co_binding : float
        CO binding energy (eV)
    temperature : float
        Temperature in K (default: 298K)
    fugacity : float
        Fugacity in Pa (default: 1e5 Pa)
        
    Returns:
    --------
    float
        Reaction free energy (ΔG) in eV
    """

    E_form_gas_dict = {
        'H2_g': 0.0,
        'CO_g': 0.0,
        'CH4_g': -2.8047879999999594,
        'CO2_g': -0.28416299999980765,
        'H2O_g':0.0,
        'H_g': 0.0,
        'ele_g': 0.0,
    }
    if fugacity is {} or None:
        for gas in E_form_gas_dict.keys():
            fugacity[gas] = 1e5
    else:
        for gas in fugacity.keys():
            if gas not in E_form_gas_dict.keys():
                fugacity[gas] = 1e5
    # Split reaction into reactants and products
    reactants, products = reaction.split(' -> ')
    reactants = [r.strip() for r in reactants.split(' + ')]
    products = [p.strip() for p in products.split(' + ')]
    
    # Initialize energies
    G_reactants = 0
    G_products = 0
    
    # Calculate G for reactants
    for species in reactants:
        if species in ['*', 'H_g', 'ele_g']:
            continue  # These species have G = 0
            
        # Get DFT energy from scaling relations
        if '*' in species:
            E_dft = calculate_binding_energy(dE_CO_ads, species, scaling_relation)
            fugacity[species] = 1e5
        else:
            E_dft = E_form_gas_dict[species]
            
        # Get frequencies and calculate thermal corrections
        frequencies = get_species_frequencies(species, frequencies_df)
        thermo_corr = calculate_thermo_corrections(species, frequencies, 
                                                 temperature, fugacity[species])
        
        # Get solvation energy
        solv_energy = get_solvation_energy(species, solvation_df, approach='explicit',
                                          manual_solv=manual_solv)
        
        # Sum up the energies: G = E_dft + F + E_solv
        G = E_dft + thermo_corr['F'] + solv_energy
        G_reactants += G
        
    # Calculate G for products
    for species in products:
        if species in ['*', 'H_g', 'ele_g']:
            continue
            
        # Get DFT energy from scaling relations
        if '*' in species:
            E_dft = calculate_binding_energy(dE_CO_ads, species, scaling_relation)
            fugacity[species] = 1e5
        else:
            E_dft = E_form_gas_dict[species]
            
        # Get frequencies and calculate thermal corrections
        frequencies = get_species_frequencies(species, frequencies_df)
        thermo_corr = calculate_thermo_corrections(species, frequencies, 
                                                 temperature, fugacity[species])
        
        # Get solvation energy
        solv_energy = get_solvation_energy(species, solvation_df, approach='explicit',
                                          manual_solv=manual_solv)
        
        # Sum up the energies: G = E_dft + F + E_solv
        G = E_dft + thermo_corr['F'] + solv_energy
        G_products += G
    
    # Calculate ΔG
    delta_G = G_products - G_reactants

    if reaction == 'CO2_g + H_g + ele_g -> CH4_g':
        delta_G = delta_G/8
        #print(f"CO2R reduction energy: {delta_G} eV")
    
    return delta_G


# Example usage:
# co_binding = -0.1  # Example CO binding energy ∆G_CO
# temperature = 298  # Example temperature
# #print(f"{co_binding:.2f} eV ∆G_CO at {temperature} K)")
# for reaction in reaction_list.values(): 
#     delta_G1 = calculate_reaction_energy(reaction, co_binding, temperature, fugacity_dict)
#     print(f"Reaction: {reaction}")
#     print(f"NEW Reaction energy: {delta_G1:.2f} eV")
#     delta_G2 = calculate_reaction_energy_before(reaction, co_binding, temperature, fugacity_dict)
#     print(f"OLD Reaction energy: {delta_G2:.2f} eV ({co_binding:.2f} eV ∆G_CO at {temperature} K)")

def plot_reaction_energies(co_binding_range: List[float], 
                          temperature: float = 298,
                          fugacity: Dict[str, float] = {},
                          manual_solv=None,
                          scaling_relation=None,
                          label_color_dict=None):
    """
    Plot reaction energies vs CO binding energy
    
    Parameters:
    -----------
    co_binding_range : List[float]
        List of CO binding energies to calculate ΔG for
    temperature : float
        Temperature in K (default: 298K)
    fugacity : float
        Fugacity in Pa (default: 1e5 Pa)
    """
    plt.figure(figsize=(10/1.5, 6/1.5))

    
    for rxn_num, reaction in reaction_list.items():
        
        delta_Gs = []
        for co_binding in co_binding_range:
            delta_G = calculate_reaction_energy_before(reaction, co_binding, 
                                             temperature, fugacity, manual_solv, scaling_relation)
            delta_Gs.append(-delta_G)  # Note: plotting -ΔG
            # if rxn_num == 1:
            #     print(f"{rxn_num}: {reaction} \t {co_binding:.2f} eV \t {delta_G:.2f} eV")
        if label_color_dict is not None:
            if rxn_num in label_color_dict.keys():
                color, label = label_color_dict[rxn_num]
                plt.plot(co_binding_range, delta_Gs, label=f'{rxn_num}: {label}', color = color)
            else:
                plt.plot(co_binding_range, delta_Gs, color = 'grey')
        else:
            plt.plot(co_binding_range, delta_Gs, label=f'Reaction {rxn_num}')
    
    plt.xlabel('ΔEads(CO) (eV)')
    plt.ylabel('-ΔG (eV)')
    plt.title(f'Reaction Free Energies at {temperature}K')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.ylim(-2.0, 2.0)
    plt.xlim(-2.0, 0.5)
    
    return plt.gcf()

def plot_reaction_energies_multiple_temperatures_dE_CO(co_binding_range: List[float], 
                                               temperatures: List[float],
                                               fugacity: Dict[str, float] = {},
                                               manual_solv=None,
                                               scaling_relation=None,
                                               E_CO_dict_100=None):
    """
    Plot reaction energies for multiple temperatures with CO2R and HER reactions
    """
    # Reaction labels dictionary
    rxn_labels = {
        1: 'CO$_2$ $\\rightarrow$ COOH*',
        3: 'CO* $\\rightarrow$ COH*',
        4: 'CO* $\\rightarrow$ CHO*',
        11: 'H$^+$ + e$^-$ $\\rightarrow$ H*',
        12: '2H* $\\rightarrow$ H$_2$',
        15: 'CO$_g$ $\\rightarrow$ CO*',
        17: 'CO$_g$ + H$^+$ + e$^-$ $\\rightarrow$ CHO*',
    }
    
    alpha = 0.8
    plt.figure(figsize=(10/2, 6/2))
    
    # 온도별 색상 정의
    temp_colors = {
        'low': '#1f77b4',   # 푸른색
        'high': '#d62728'   # 붉은색
    }
    
    # Dictionary to store differences between HER and CO2R
    delta_G_differences = {metal: [] for metal in E_CO_dict_100.keys()}
    
    # Convert x-axis values using reaction 15
    # x_values_converted = []
    # for x in co_binding_range:
    #     # Calculate reaction 15 energy for each CO binding energy
    #     delta_G_15 = calculate_reaction_energy(reaction_list[15], x, 
    #                                          temperatures[0], fugacity, 
    #                                          manual_solv, scaling_relation)
    #     x_values_converted.append(delta_G_15)
    
    # Plot for each temperature
    for temp_idx, temperature in enumerate(temperatures):
        # 현재 온도가 낮은 온도인지 높은 온도인지 결정
        is_low_temp = temp_idx == 0
        current_color = temp_colors['low'] if is_low_temp else temp_colors['high']
        
        # Calculate ΔG values for CO2R reactions
        co2r_rxns = [17, 3, 4]
        delta_G_dict = {}
        
        # Convert metal points x-coordinates
        G_CO_dict = {}
        # fugacity_dict = {
        #     'H2_g': 101325,
        #     'CO2_g': 101325,
        #     'CO_g': 5562,
        #     'HCOOH_g': 2,
        #     'CH3OH_g': 6079,
        #     'H2O_g': 3534,
        #     'CH4_g': 20467,
        # }
        for metal, data in E_CO_dict_100.items():
            e_co = data['E']
            frequencies = get_species_frequencies('CO_g', frequencies_df)
            thermo_corr = calculate_thermo_corrections('CO_g', frequencies, 
                                                 temperature, 5562)
            co_gas_solv_energy = get_solvation_energy('CO_g', solvation_df, approach='explicit',    
                                                      manual_solv=manual_solv)
            dF_CO_gas = thermo_corr['F'] + co_gas_solv_energy

            frequencies = get_species_frequencies('CO*', frequencies_df)
            thermo_corr = calculate_thermo_corrections('CO*', frequencies, 
                                                 temperature, 1e5)
            co_ads_solv_energy = get_solvation_energy('CO*', solvation_df, approach='explicit',    
                                                      manual_solv=manual_solv)
            dF_CO_ads = thermo_corr['F'] + co_ads_solv_energy
            G_CO_ads = e_co + dF_CO_ads
            G_CO_gas = dF_CO_gas
            dG_CO_ads = G_CO_ads - G_CO_gas

            

            # Calculate reaction 15 energy for this metal
            # delta_G_15 = calculate_reaction_energy(reaction_list[15], e_co, 
            #                                      temperature, fugacity, 
            #                                      manual_solv, scaling_relation)
            G_CO_dict[metal] = {
                'G': e_co,
                'highlight': data['highlight']
            }

        for rxn_num in co2r_rxns + [11, 12]:  # Calculate all reactions
            print(f"Calculating reaction {rxn_num} at {temperature}K: {reaction_list[rxn_num]}")
            delta_Gs = []
            for co_binding in co_binding_range:
                delta_G = calculate_reaction_energy_before(reaction_list[rxn_num], co_binding, 
                                                 temperature, fugacity, manual_solv, scaling_relation)
                delta_Gs.append(-delta_G)
            delta_G_dict[rxn_num] = delta_Gs
        
        # Process CO2R reactions (1, 3, 4)
        higher_values = []
        higher_rxn = []
        x_values = co_binding_range
        
        for i in range(len(x_values)):
            if delta_G_dict[3][i] > delta_G_dict[4][i]:
                higher_values.append(delta_G_dict[3][i])
                higher_rxn.append(3)
            else:
                higher_values.append(delta_G_dict[4][i])
                higher_rxn.append(4)
        
        bold_rxn_co2r = []
        for i in range(len(x_values)):
            if delta_G_dict[17][i] < higher_values[i]:
                bold_rxn_co2r.append(17)
            else:
                bold_rxn_co2r.append(higher_rxn[i])
        
        # Plot CO2R reactions
        for rxn_num in co2r_rxns:
            y_values = delta_G_dict[rxn_num]
            
            solid_x, solid_y = [], []
            dash_x, dash_y = [], []
            
            for i, x in enumerate(x_values):
                if bold_rxn_co2r[i] == rxn_num:
                    solid_x.append(x)
                    solid_y.append(y_values[i])
                else:
                    dash_x.append(x)
                    dash_y.append(y_values[i])
            
            if solid_x:
                plt.plot(solid_x, solid_y, '-', linewidth=1.5, color=current_color)
            if dash_x:
                plt.plot(dash_x, dash_y, '--', color=current_color, alpha=alpha)
        
        # Process HER reactions (11, 12)
        her_rxns = [11, 12]
        bold_rxn_her = []
        for i in range(len(x_values)):
            if delta_G_dict[11][i] < delta_G_dict[12][i]:
                bold_rxn_her.append(11)
            else:
                bold_rxn_her.append(12)
        
        # Plot HER reactions
        for rxn_num in her_rxns:
            y_values = delta_G_dict[rxn_num]
            
            solid_x, solid_y = [], []
            dash_x, dash_y = [], []
            
            for i, x in enumerate(x_values):
                if bold_rxn_her[i] == rxn_num:
                    solid_x.append(x)
                    solid_y.append(y_values[i])
                else:
                    dash_x.append(x)
                    dash_y.append(y_values[i])
            
            if solid_x:
                plt.plot(solid_x, solid_y, '-', linewidth=1.5, color=current_color, alpha=0.8)
            if dash_x:
                plt.plot(dash_x, dash_y, '--', color=current_color, alpha=alpha, zorder=0)
            
        # Add reaction labels only for the last temperature
        if temp_idx == len(temperatures) - 1:
            for rxn_num in co2r_rxns:
                y_at_rightmost = delta_G_dict[rxn_num][-1]
                plt.text(0.55, y_at_rightmost, rxn_labels[rxn_num], 
                        color='black', va='center', fontsize=8)
            for rxn_num in her_rxns:
                y_at_rightmost = delta_G_dict[rxn_num][-1]
                plt.text(0.55, y_at_rightmost, rxn_labels[rxn_num], 
                        color='black', va='center', fontsize=8)
        
        # Add legend entries
        if temp_idx == 0:
            plt.plot([], [], '-', color=current_color,  
                    label=f'T = {temperature-273}°C')
        else:
            plt.plot([], [], '-', color=current_color, 
                    label=f'T = {temperature-273}°C')

        # Only plot connecting lines for first and last temperature
        is_extreme_temp = (temp_idx == 0) or (temp_idx == len(temperatures) - 1)
        
        # Find y-values at metal points
        for metal, data in G_CO_dict.items():
            g_co = data['G']
            is_highlight = data['highlight']
            
            # Find nearest x-value index
            idx = np.abs(np.array(co_binding_range) - g_co).argmin()
            
            # Find which reaction is solid at this point for CO2R and HER
            co2r_solid_rxn = bold_rxn_co2r[idx]
            her_solid_rxn = bold_rxn_her[idx]
            co2r_y = delta_G_dict[co2r_solid_rxn][idx]
            her_y = delta_G_dict[her_solid_rxn][idx]
            
            # Calculate and store difference between HER and CO2R
            difference = abs(her_y - co2r_y)
            delta_G_differences[metal].append((temperature, difference, g_co, co2r_y, her_y))
            
            # If it's an extreme temperature, print the difference
            if is_extreme_temp:
                #print(f"{metal} at {temperature}K: ΔG(HER-CO2R) = {difference:.3f} eV")
                marker = 'o'
                point_size = 20

                plt.plot([g_co, g_co], [her_y, co2r_y], '-', 
                        color=current_color, linewidth=1, zorder=4)
                plt.scatter(g_co, co2r_y, color=current_color, s=point_size, 
                          marker='o', edgecolors='none', zorder=5)
                plt.scatter(g_co, her_y, color=current_color, s=point_size, 
                          marker='o', edgecolors='none', zorder=5)

                # Metal labels with temperature-matched colors
                #if is_highlight:
                if metal in ['Pd', 'Ir', 'Ag']:
                                    plt.annotate(metal, 
                            (g_co, her_y),
                            xytext=(0, 10),
                            textcoords='offset points',
                            ha='center',
                            va='top',
                            color=current_color,
                            fontsize=8,
                            fontweight='normal')
                else:
                    plt.annotate(metal, 
                            (g_co, co2r_y),
                            xytext=(0, -5),
                            textcoords='offset points',
                            ha='center',
                            va='top',
                            color=current_color,
                            fontsize=8,
                            fontweight='normal')
    
    plt.xlabel('$∆E_{ads}(CO)$ $(eV)$')
    plt.ylabel('Limiting Potential $(V_{RHE})$')
    plt.legend(loc='upper right')
    #plt.grid(True, alpha=0.3)
    plt.ylim(-2.0, 1.0)
    plt.xlim(-2.0, 0.5)
    plt.tight_layout()
    
    # Print summary of differences for each metal
    print("\nSummary of ΔG differences (HER-CO2R):")
    print("\nMetal   Low T(K)   Low T(eV)   High T(K)   High T(eV)   Difference(eV)")
    print("-" * 65)
    data = []
    for metal in delta_G_differences:
        # Sort by temperature to get low and high T values
        sorted_diffs = sorted(delta_G_differences[metal], key=lambda x: x[0])
        low_temp = sorted_diffs[0]  # (temp, diff, g_co, co2r_y, her_y) at lowest T
        high_temp = sorted_diffs[-1]  # (temp, diff, g_co, co2r_y, her_y) at highest T
        temp_diff = high_temp[1] - low_temp[1]  # Difference between high T and low T
        
        data.append({
            'Metal': metal,
            'Low T(K)': low_temp[0],
            'Low T(eV)': low_temp[1],
            'High T(K)': high_temp[0],
            'High T(eV)': high_temp[1],
            'Difference(eV)': temp_diff,
            'Low T G_CO': low_temp[2],
            'High T G_CO': high_temp[2],
            'Delta G_CO': high_temp[2] - low_temp[2],
            'Low T CO2R limiting potential': low_temp[3],
            'High T CO2R limiting potential': high_temp[3],
            'Delta CO2R limiting potential': high_temp[3] - low_temp[3],
            'Low T HER limiting potential': low_temp[4],
            'High T HER limiting potential': high_temp[4],
            'Delta HER limiting potential': high_temp[4] - low_temp[4],
        })
        
        # Print summary line for all metals
        print(f"{metal:6s}  {low_temp[0]:8.0f}  {low_temp[1]:10.3f}  {high_temp[0]:8.0f}  {high_temp[1]:10.3f}  {temp_diff:12.3f}")
    
    df = pd.DataFrame(data)

    # Print overall statistics
    temp_differences = {metal: sorted(diffs, key=lambda x: x[0])[-1][1] - sorted(diffs, key=lambda x: x[0])[0][1]
                       for metal, diffs in delta_G_differences.items()}
    
    print("\nOverall Temperature Effect Statistics:")
    print(f"Smallest T effect: {min(temp_differences.items(), key=lambda x: x[1])[0]} ({min(temp_differences.values()):.3f} eV)")
    print(f"Largest T effect:  {max(temp_differences.items(), key=lambda x: x[1])[0]} ({max(temp_differences.values()):.3f} eV)")
    return plt.gcf(), df

def plot_reaction_energies_multiple_temperatures_dG_CO(co_binding_range: List[float], 
                                               temperatures: List[float],
                                               fugacity: Dict[str, float] = {},
                                               manual_solv=None,
                                               scaling_relation=None,
                                               E_CO_dict_100=None):
    """
    Plot reaction energies for multiple temperatures with CO2R and HER reactions
    """
    # Reaction labels dictionary
    rxn_labels = {
        1: 'CO$_2$ $\\rightarrow$ COOH*',
        3: 'CO* $\\rightarrow$ COH*',
        4: 'CO* $\\rightarrow$ CHO*',
        11: 'H$^+$ + e$^-$ $\\rightarrow$ H*',
        12: '2H* $\\rightarrow$ H$_2$',
        15: 'CO$_g$ $\\rightarrow$ CO*',
        17: 'CO$_g$ + H$^+$ + e$^-$ $\\rightarrow$ CHO*',
    }
    
    alpha = 0.8
    plt.figure(figsize=(10/2, 6/2))
    
    # 온도별 색상 정의
    temp_colors = {
        'low': '#1f77b4',   # 푸른색
        'high': '#d62728'   # 붉은색
    }
    
    # Dictionary to store differences between HER and CO2R
    delta_G_differences = {metal: [] for metal in E_CO_dict_100.keys()}
    
    # Convert x-axis values using reaction 15
    # x_values_converted = []
    # for x in co_binding_range:
    #     # Calculate reaction 15 energy for each CO binding energy
    #     delta_G_15 = calculate_reaction_energy(reaction_list[15], x, 
    #                                          temperatures[0], fugacity, 
    #                                          manual_solv, scaling_relation)
    #     x_values_converted.append(delta_G_15)
    
    # Plot for each temperature
    for temp_idx, temperature in enumerate(temperatures):
        # 현재 온도가 낮은 온도인지 높은 온도인지 결정
        is_low_temp = temp_idx == 0
        current_color = temp_colors['low'] if is_low_temp else temp_colors['high']
        
        # Calculate ΔG values for CO2R reactions
        co2r_rxns = [17, 3, 4]
        delta_G_dict = {}
        
        # Convert metal points x-coordinates
        G_CO_dict = {}
        # fugacity_dict = {
        #     'H2_g': 101325,
        #     'CO2_g': 101325,
        #     'CO_g': 5562,
        #     'HCOOH_g': 2,
        #     'CH3OH_g': 6079,
        #     'H2O_g': 3534,
        #     'CH4_g': 20467,
        # }
        for metal, data in E_CO_dict_100.items():
            e_co = data['E']
            frequencies = get_species_frequencies('CO_g', frequencies_df)
            thermo_corr = calculate_thermo_corrections('CO_g', frequencies, 
                                                 temperature, 5562)
            co_gas_solv_energy = get_solvation_energy('CO_g', solvation_df, approach='explicit',    
                                                      manual_solv=manual_solv)
            dF_CO_gas = thermo_corr['F'] + co_gas_solv_energy

            frequencies = get_species_frequencies('CO*', frequencies_df)
            thermo_corr = calculate_thermo_corrections('CO*', frequencies, 
                                                 temperature, 1e5)
            co_ads_solv_energy = get_solvation_energy('CO*', solvation_df, approach='explicit',    
                                                      manual_solv=manual_solv)
            dF_CO_ads = thermo_corr['F'] + co_ads_solv_energy
            G_CO_ads = e_co + dF_CO_ads
            G_CO_gas = dF_CO_gas
            dG_CO_ads = G_CO_ads - G_CO_gas

            

            # Calculate reaction 15 energy for this metal
            # delta_G_15 = calculate_reaction_energy(reaction_list[15], e_co, 
            #                                      temperature, fugacity, 
            #                                      manual_solv, scaling_relation)
            G_CO_dict[metal] = {
                'G': dG_CO_ads,
                'highlight': data['highlight']
            }

        for rxn_num in co2r_rxns + [11, 12]:  # Calculate all reactions
            print(f"Calculating reaction {rxn_num} at {temperature}K: {reaction_list[rxn_num]}")
            delta_Gs = []
            for co_binding in co_binding_range:
                delta_G = calculate_reaction_energy(reaction_list[rxn_num], co_binding, 
                                                 temperature, fugacity, manual_solv, scaling_relation)
                delta_Gs.append(-delta_G)
            delta_G_dict[rxn_num] = delta_Gs
        
        # Process CO2R reactions (1, 3, 4)
        higher_values = []
        higher_rxn = []
        x_values = co_binding_range
        
        for i in range(len(x_values)):
            if delta_G_dict[3][i] > delta_G_dict[4][i]:
                higher_values.append(delta_G_dict[3][i])
                higher_rxn.append(3)
            else:
                higher_values.append(delta_G_dict[4][i])
                higher_rxn.append(4)
        
        bold_rxn_co2r = []
        for i in range(len(x_values)):
            if delta_G_dict[co2r_rxns[0]][i] < higher_values[i]:
                bold_rxn_co2r.append(co2r_rxns[0])
            else:
                bold_rxn_co2r.append(higher_rxn[i])
        
        # Plot CO2R reactions
        for rxn_num in co2r_rxns:
            y_values = delta_G_dict[rxn_num]
            
            solid_x, solid_y = [], []
            dash_x, dash_y = [], []
            
            for i, x in enumerate(x_values):
                if bold_rxn_co2r[i] == rxn_num:
                    solid_x.append(x)
                    solid_y.append(y_values[i])
                else:
                    dash_x.append(x)
                    dash_y.append(y_values[i])
            
            if solid_x:
                plt.plot(solid_x, solid_y, '-', linewidth=1.5, color=current_color)
            if dash_x:
                plt.plot(dash_x, dash_y, '--', color=current_color, alpha=alpha)
        
        # Process HER reactions (11, 12)
        her_rxns = [11, 12]
        bold_rxn_her = []
        for i in range(len(x_values)):
            if delta_G_dict[11][i] < delta_G_dict[12][i]:
                bold_rxn_her.append(11)
            else:
                bold_rxn_her.append(12)
        
        # Plot HER reactions
        for rxn_num in her_rxns:
            y_values = delta_G_dict[rxn_num]
            
            solid_x, solid_y = [], []
            dash_x, dash_y = [], []
            
            for i, x in enumerate(x_values):
                if bold_rxn_her[i] == rxn_num:
                    solid_x.append(x)
                    solid_y.append(y_values[i])
                else:
                    dash_x.append(x)
                    dash_y.append(y_values[i])
            
            if solid_x:
                plt.plot(solid_x, solid_y, '-', linewidth=1.5, color=current_color, alpha=0.8)
            if dash_x:
                plt.plot(dash_x, dash_y, '--', color=current_color, alpha=alpha, zorder=0)
            
        # Add reaction labels only for the last temperature
        if temp_idx == len(temperatures) - 1:
            for rxn_num in co2r_rxns:
                y_at_rightmost = delta_G_dict[rxn_num][-1]
                plt.text(0.55, y_at_rightmost, rxn_labels[rxn_num], 
                        color='black', va='center', fontsize=8)
            for rxn_num in her_rxns:
                y_at_rightmost = delta_G_dict[rxn_num][-1]
                plt.text(0.55, y_at_rightmost, rxn_labels[rxn_num], 
                        color='black', va='center', fontsize=8)
        
        # Add legend entries
        if temp_idx == 0:
            plt.plot([], [], '-', color=current_color,  
                    label=f'T = {temperature-273}°C')
        else:
            plt.plot([], [], '-', color=current_color, 
                    label=f'T = {temperature-273}°C')

        # Only plot connecting lines for first and last temperature
        is_extreme_temp = (temp_idx == 0) or (temp_idx == len(temperatures) - 1)
        
        # Find y-values at metal points
        for metal, data in G_CO_dict.items():
            g_co = data['G']
            is_highlight = data['highlight']
            
            # Find nearest x-value index
            idx = np.abs(np.array(co_binding_range) - g_co).argmin()
            
            # Find which reaction is solid at this point for CO2R and HER
            co2r_solid_rxn = bold_rxn_co2r[idx]
            her_solid_rxn = bold_rxn_her[idx]
            co2r_y = delta_G_dict[co2r_solid_rxn][idx]
            her_y = delta_G_dict[her_solid_rxn][idx]
            
            # Calculate and store difference between HER and CO2R
            difference = abs(her_y - co2r_y)
            delta_G_differences[metal].append((temperature, difference, g_co, co2r_y, her_y))
            
            # If it's an extreme temperature, print the difference
            if is_extreme_temp:
                #print(f"{metal} at {temperature}K: ΔG(HER-CO2R) = {difference:.3f} eV")
                marker = 'o'
                point_size = 20

                plt.plot([g_co, g_co], [her_y, co2r_y], '-', 
                        color=current_color, linewidth=1, zorder=4)
                plt.scatter(g_co, co2r_y, color=current_color, s=point_size, 
                          marker='o', edgecolors='none', zorder=5)
                plt.scatter(g_co, her_y, color=current_color, s=point_size, 
                          marker='o', edgecolors='none', zorder=5)

                # Metal labels with temperature-matched colors
                #if is_highlight:
                if metal in ['Pd', 'Ir', 'Ag']:
                                    plt.annotate(metal, 
                            (g_co, her_y),
                            xytext=(0, 10),
                            textcoords='offset points',
                            ha='center',
                            va='top',
                            color=current_color,
                            fontsize=8,
                            fontweight='normal')
                else:
                    plt.annotate(metal, 
                            (g_co, co2r_y),
                            xytext=(0, -5),
                            textcoords='offset points',
                            ha='center',
                            va='top',
                            color=current_color,
                            fontsize=8,
                            fontweight='normal')
    
    plt.xlabel('$∆G_{ads}(CO)$ $(eV)$')
    plt.ylabel('Limiting Potential $(V_{RHE})$')
    plt.legend(loc='upper right')
    #plt.grid(True, alpha=0.3)
    plt.ylim(-2.0, 1.0)
    plt.xlim(-2.0, 0.5)
    plt.tight_layout()
    
    # Print summary of differences for each metal
    print("\nSummary of ΔG differences (HER-CO2R):")
    print("\nMetal   Low T(K)   Low T(eV)   High T(K)   High T(eV)   Difference(eV)")
    print("-" * 65)
    data = []
    for metal in delta_G_differences:
        # Sort by temperature to get low and high T values
        sorted_diffs = sorted(delta_G_differences[metal], key=lambda x: x[0])
        low_temp = sorted_diffs[0]  # (temp, diff, g_co, co2r_y, her_y) at lowest T
        high_temp = sorted_diffs[-1]  # (temp, diff, g_co, co2r_y, her_y) at highest T
        temp_diff = high_temp[1] - low_temp[1]  # Difference between high T and low T
        
        data.append({
            'Metal': metal,
            'Low T(K)': low_temp[0],
            'Low T(eV)': low_temp[1],
            'High T(K)': high_temp[0],
            'High T(eV)': high_temp[1],
            'Difference(eV)': temp_diff,
            'Low T G_CO': low_temp[2],
            'High T G_CO': high_temp[2],
            'Delta G_CO': high_temp[2] - low_temp[2],
            'Low T CO2R limiting potential': low_temp[3],
            'High T CO2R limiting potential': high_temp[3],
            'Delta CO2R limiting potential': high_temp[3] - low_temp[3],
            'Low T HER limiting potential': low_temp[4],
            'High T HER limiting potential': high_temp[4],
            'Delta HER limiting potential': high_temp[4] - low_temp[4],
        })
        
        # Print summary line for all metals
        print(f"{metal:6s}  {low_temp[0]:8.0f}  {low_temp[1]:10.3f}  {high_temp[0]:8.0f}  {high_temp[1]:10.3f}  {temp_diff:12.3f}")
    
    df = pd.DataFrame(data)

    # Print overall statistics
    temp_differences = {metal: sorted(diffs, key=lambda x: x[0])[-1][1] - sorted(diffs, key=lambda x: x[0])[0][1]
                       for metal, diffs in delta_G_differences.items()}
    
    print("\nOverall Temperature Effect Statistics:")
    print(f"Smallest T effect: {min(temp_differences.items(), key=lambda x: x[1])[0]} ({min(temp_differences.values()):.3f} eV)")
    print(f"Largest T effect:  {max(temp_differences.items(), key=lambda x: x[1])[0]} ({max(temp_differences.values()):.3f} eV)")
    return plt.gcf(), df

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def create_reaction_rate_contours_dE_CO(co_binding_range, temp_range, fugacity_dict, manual_solv, scaling_relation, E_CO_dict_100):
    """
    CO binding energy와 온도에 따른 반응 속도 비율을 계산하고 시각화합니다.
    """
    # Constants
    k_b = 8.617333262e-5  # eV/K
    h = 4.135667696e-15   # eV·s
    # Plot for each temperature

    solid_y_co2r_dict = {}  # {temp_idx: bold_reactions}
    solid_y_her_dict = {}  # {temp_idx: bold_reactions}
    metal_binding_dict = {}  # {temp_idx: metal_dict}
    
    for temp_idx, temperature in enumerate(temp_range):
        temperature = temperature + 273.15
        solid_y_co2r_dict[temp_idx] = []
        solid_y_her_dict[temp_idx] = []
        # Calculate ΔG values for CO2R reactions
        co2r_rxns = [17, 3, 4]
        delta_G_dict = {} 

        for rxn_num in co2r_rxns + [11, 12]:  # Calculate all reactions
            #print(f"Calculating reaction {rxn_num} at {temperature}K: {reaction_list[rxn_num]}")
            delta_Gs = []
            for co_binding in co_binding_range:
                
                delta_G = calculate_reaction_energy_before(reaction_list[rxn_num], co_binding, 
                                                 temperature, fugacity_dict, manual_solv, scaling_relation)
                delta_Gs.append(-delta_G)
            delta_G_dict[rxn_num] = delta_Gs
        
        # Process CO2R reactions (1, 3, 4)
        # COR (15, 3, 4 )
        higher_values = []
        higher_rxn = []
        x_values = co_binding_range
        
        for i in range(len(x_values)):
            if delta_G_dict[3][i] > delta_G_dict[4][i]:
                higher_values.append(delta_G_dict[3][i])
                higher_rxn.append(3)
            else:
                higher_values.append(delta_G_dict[4][i])
                higher_rxn.append(4)
        
        bold_rxn_co2r = []
        for i in range(len(x_values)):
            if delta_G_dict[17][i] < higher_values[i]:
                bold_rxn_co2r.append(17)
            else:
                bold_rxn_co2r.append(higher_rxn[i])

        
        # Plot CO2R reactions
        for rxn_num in co2r_rxns:
            y_values = delta_G_dict[rxn_num]
            
            solid_x_co2r, solid_y_co2r = [], []
            dash_x_co2r, dash_y_co2r = [], []
            
            for i, x in enumerate(x_values):
                if bold_rxn_co2r[i] == rxn_num:
                    solid_x_co2r.append(x)
                    solid_y_co2r.append(y_values[i])
                    solid_y_co2r_dict[temp_idx].append((x,y_values[i]))
                else:
                    dash_x_co2r.append(x)
                    dash_y_co2r.append(y_values[i])
        
        # Convert metal points x-coordinates
        G_CO_dict = {}
        for metal, data in E_CO_dict_100.items():
            e_co = data['E']

            frequencies = get_species_frequencies('CO_g', frequencies_df)
            thermo_corr = calculate_thermo_corrections('CO_g', frequencies, 
                                                 temperature, fugacity_dict['CO_g'])
            co_gas_solv_energy = get_solvation_energy('CO_g', solvation_df, approach='explicit',    
                                                      manual_solv=manual_solv)
            dF_CO_gas = thermo_corr['F'] + co_gas_solv_energy

            frequencies = get_species_frequencies('CO*', frequencies_df)
            thermo_corr = calculate_thermo_corrections('CO*', frequencies, 
                                                 temperature, fugacity_dict['CO*'])
            co_ads_solv_energy = get_solvation_energy('CO*', solvation_df, approach='explicit',    
                                                      manual_solv=manual_solv)
            dF_CO_ads = thermo_corr['F'] + co_ads_solv_energy
            G_CO_ads = e_co + dF_CO_ads
            G_CO_gas = dF_CO_gas
            dG_CO_ads = G_CO_ads - G_CO_gas
            G_CO_dict[metal] = {
                'G': e_co,
                'highlight': data['highlight']
            }           

     
        metal_binding_dict[temp_idx] = G_CO_dict
        
        
        # Process HER reactions (11, 12)
        her_rxns = [11, 12]
        bold_rxn_her = []
        for i in range(len(x_values)):
            if delta_G_dict[11][i] < delta_G_dict[12][i]:
                bold_rxn_her.append(11)
            else:
                bold_rxn_her.append(12)
        
        # Plot HER reactions
        for rxn_num in her_rxns:
            y_values = delta_G_dict[rxn_num]
            
            solid_x_her, solid_y_her = [], []
            dash_x_her, dash_y_her = [], []
            
            for i, x in enumerate(x_values):
                if bold_rxn_her[i] == rxn_num:
                    solid_x_her.append(x)
                    solid_y_her.append(y_values[i])
                    solid_y_her_dict[temp_idx].append((x,y_values[i]))
                else:
                    dash_x_her.append(x)
                    dash_y_her.append(y_values[i])
            
        for metal, data in G_CO_dict.items():
            g_co = data['G']
            is_highlight = data['highlight']
            # Find nearest x-value index
            idx = np.abs(np.array(co_binding_range) - g_co).argmin()
            
            # Find which reaction is solid at this point for CO2R and HER
            co2r_solid_rxn = bold_rxn_co2r[idx]
            her_solid_rxn = bold_rxn_her[idx]
            co2r_y = delta_G_dict[co2r_solid_rxn][idx]
            her_y = delta_G_dict[her_solid_rxn][idx]


        

    # Store data for each temperature
    
    print(solid_y_co2r_dict.keys())

    print(solid_y_her_dict.keys())
    
    

    #print(solid_y_co2r_dict)
    
    # Create figure with two subplots
    fig, ax1 = plt.subplots(1, 1, figsize=(10/1.5, 8/1.5))
    
    # First plot: Full calculation
    X1, X2 = np.meshgrid(co_binding_range, temp_range + 273.15)
    
    Z1 = np.zeros_like(X1)
    print(X1[1,0])
    print(X2[0,1])
    for i in range(len(temp_range)):
        for j in range(len(co_binding_range)):
            co_binding = X1[i,j] 

            T = X2[i,j]

            # Get CO2R limiting potential from bold reactions
            # solid_y_co2r_dict[i] contains list of (x,y) tuples
            # Find tuple where x matches co_binding and get its y value
            co2r_tuple = next((t for t in solid_y_co2r_dict[i] if t[0] == co_binding), None)
            co2r_limiting = -co2r_tuple[1] if co2r_tuple else 0
            
            # Get HER limiting potential from bold reactions  
            her_tuple = next((t for t in solid_y_her_dict[i] if t[0] == co_binding), None)
            her_limiting = -her_tuple[1] if her_tuple else 0
            
            if i == 25 and j == 25:
                print(f"{co_binding}, {T}")
                print(f"T: {T}, co2r_limiting: {co2r_limiting}, her_limiting: {her_limiting}")
            # Calculate rate constants and ratio
            k_co2r = (k_b * T / h) * np.exp(-co2r_limiting / (k_b * T))
            k_her = (k_b * T / h) * np.exp(-her_limiting / (k_b * T))
            Z1[i,j] = k_co2r / k_her


    # Create lines for each metal
    metals_data = {}
    for metal in metal_binding_dict[0].keys():
        temps = []
        g_cos = []
        for i in range(len(temp_range)):
            temps.append(temp_range[i])
            g_cos.append(metal_binding_dict[i][metal]['G'])
        metals_data[metal] = {'temps': temps, 'g_cos': g_cos}

        
    # Plot line for each metal and add label next to line
    # Define vertical offsets for each metal relative to Fe (which is at 0)
    vertical_offsets = {}
    # Get g_cos values at middle temperature point for ranking
    mid_temp_idx = len(temp_range) // 2
    g_cos_values = {metal: data['g_cos'][mid_temp_idx] for metal, data in metals_data.items()}
    
    # Sort metals by g_cos values
    sorted_metals = sorted(g_cos_values.items(), key=lambda x: x[1])
    
    # Find Cu's rank
    cu_rank = next(i for i, (metal, _) in enumerate(sorted_metals) if metal == 'Cu')
    
    # Set offsets based on distance from Cu's rank
    for rank, (metal, _) in enumerate(sorted_metals):
        if rank < cu_rank:
            # Metals with more negative g_cos than Cu
            vertical_offsets[metal] = (rank - cu_rank) * 10
        elif rank > cu_rank:
            # Metals with more positive g_cos than Cu
            vertical_offsets[metal] = (rank - cu_rank) * 10
        else:
            # Cu itself
            vertical_offsets[metal] = 0

    # vertical_offsets = {
    #     'Pt': -15,  # Pt label below Fe
    #     'Ni': -10,  # Ni label below Fe
    #     'Fe': 0,    # Fe at center
    #     'Cu': 5,    # Cu label above Fe
    #     'Au': 10,   # Au label above Fe
    #     'Ag': 15    # Ag label above Fe
    # }
    
    for metal, data in metals_data.items():
        line = ax1.plot(data['g_cos'], data['temps'], ':', linewidth=1.5, color='black')[0]
        # Add label at the middle point with vertical offset
        mid_idx = int(len(data['temps']) * 0.25)  # 25% point from bottom
        offset = vertical_offsets.get(metal, 0)  # Get offset for this metal, default 0
        ax1.annotate(metal, 
                    xy=(data['g_cos'][mid_idx], data['temps'][mid_idx]),
                    xytext=(5, offset),  # 5 points right, variable vertical offset
                    textcoords='offset points',
                    va='center',
                    color='black')
                
    
    # Plot settings for Full Calculation
    log_scale = (-14, 0)
    ticks_num = log_scale[1] - log_scale[0] + 1   
    levels = np.logspace(log_scale[0], log_scale[1], 40)
    #RdBu_r
    import seaborn as sns
    from matplotlib.colors import ListedColormap

    palette = 'rocket'
    cmap = sns.color_palette(palette, as_cmap=True)
    cmap = 'RdBu_r'
    cmap = 'Reds'
    contour = ax1.contourf(X1, X2-273.15, Z1, levels=levels, norm=LogNorm(), cmap=cmap, alpha=1.0)
    
    # Add colorbar with integer ticks

    cbar = plt.colorbar(contour, ax=ax1, ticks=np.logspace(log_scale[0], log_scale[1], ticks_num//2))
    cbar.set_label('k(T)$_{CO2R}$ / k(T)$_{HER}$', rotation=270, labelpad=15)
    # cbar.ax.set_yticklabels(['$10^{-14}$', '$10^{-13}$', '$10^{-12}$', '$10^{-11}$', 
    #                         '$10^{-10}$', '$10^{-9}$', '$10^{-8}$', '$10^{-7}$',
    #                         '$10^{-6}$', '$10^{-5}$', '$10^{-4}$', '$10^{-3}$',
    #                         '$10^{-2}$', '$10^{-1}$', '$10^{0}$', '$10^{1}$',
    #                         '$10^{2}$', '$10^{3}$', '$10^{4}$', '$10^{5}$',
    #                         '$10^{6}$', '$10^{7}$'])
    # Add k_CO2R = k_HER line
    CS = ax1.contour(X1, X2-273.15, Z1, levels=[1.0], colors='black', 
                   linestyles='--', linewidths=2)
    CS = ax1.contour(X1, X2-273.15, Z1, levels=[1e-2], colors='black', alpha=0.5, 
                   linestyles='--', linewidths=1)
    #ax1.clabel(CS, inline=True, fmt='k$_{CO2R}$ = k$_{HER}$')
    ax1.clabel(CS, inline=True, fmt='1e-2')

    CS2 = ax1.contour(X1, X2-273.15, Z1, levels=[1e-3], colors='black', alpha=0.5,
                   linestyles='--', linewidths=1)
    ax1.clabel(CS2, inline=True, fmt='1e-3')
    
    # Labels and title
    ax1.set_xlabel('ΔE$_{CO}$ (eV)')
    ax1.set_ylabel('Temperature (°C)')
    #ax1.set_title('Full Calculation')
    ax1.set_ylim(0, 140)
    ax1.set_xlim(co_binding_range[0], co_binding_range[-1])

    plt.tight_layout()
    return fig

def create_reaction_rate_contours_dG_CO(co_binding_range, temp_range, fugacity_dict, manual_solv, scaling_relation, E_CO_dict_100):
    """
    CO binding energy와 온도에 따른 반응 속도 비율을 계산하고 시각화합니다.
    """
    # Constants
    k_b = 8.617333262e-5  # eV/K
    h = 4.135667696e-15   # eV·s
    # Plot for each temperature

    
    # co_binding_range: list of delta G_CO values 

    solid_y_co2r_dict = {}  # {temp_idx: bold_reactions}
    solid_y_her_dict = {}  # {temp_idx: bold_reactions}
    metal_binding_dict = {}  # {temp_idx: metal_dict}
    
    for temp_idx, temperature in enumerate(temp_range):
        temperature = temperature + 273.15
        solid_y_co2r_dict[temp_idx] = []
        solid_y_her_dict[temp_idx] = []
        # Calculate ΔG values for CO2R reactions
        co2r_rxns = [17, 3, 4]
        delta_G_dict = {} 

        for rxn_num in co2r_rxns + [11, 12]:  # Calculate all reactions
            #print(f"Calculating reaction {rxn_num} at {temperature}K: {reaction_list[rxn_num]}")
            delta_Gs = []
            for co_binding in co_binding_range:
                
                delta_G = calculate_reaction_energy(reaction_list[rxn_num], co_binding, 
                                                 temperature, fugacity_dict, manual_solv, scaling_relation)
                delta_Gs.append(-delta_G)
            delta_G_dict[rxn_num] = delta_Gs
        
        # Process CO2R reactions (1, 3, 4)
        # COR (15, 3, 4 )
        higher_values = []
        higher_rxn = []
        x_values = co_binding_range
        
        for i in range(len(x_values)):
            if delta_G_dict[3][i] > delta_G_dict[4][i]:
                higher_values.append(delta_G_dict[3][i])
                higher_rxn.append(3)
            else:
                higher_values.append(delta_G_dict[4][i])
                higher_rxn.append(4)
        
        bold_rxn_co2r = []
        for i in range(len(x_values)):
            if delta_G_dict[co2r_rxns[0]][i] < higher_values[i]:
                bold_rxn_co2r.append(co2r_rxns[0])
            else:
                bold_rxn_co2r.append(higher_rxn[i])

        
        # Plot CO2R reactions
        for rxn_num in co2r_rxns:
            y_values = delta_G_dict[rxn_num]
            
            solid_x_co2r, solid_y_co2r = [], []
            dash_x_co2r, dash_y_co2r = [], []
            
            for i, x in enumerate(x_values):
                if bold_rxn_co2r[i] == rxn_num:
                    solid_x_co2r.append(x)
                    solid_y_co2r.append(y_values[i])
                    solid_y_co2r_dict[temp_idx].append((x,y_values[i]))
                else:
                    dash_x_co2r.append(x)
                    dash_y_co2r.append(y_values[i])
        
        # Convert metal points x-coordinates
        G_CO_dict = {}
        for metal, data in E_CO_dict_100.items():
            e_co = data['E']

            frequencies = get_species_frequencies('CO_g', frequencies_df)
            thermo_corr = calculate_thermo_corrections('CO_g', frequencies, 
                                                 temperature, fugacity_dict['CO_g'])
            co_gas_solv_energy = get_solvation_energy('CO_g', solvation_df, approach='explicit',    
                                                      manual_solv=manual_solv)
            dF_CO_gas = thermo_corr['F'] + co_gas_solv_energy

            frequencies = get_species_frequencies('CO*', frequencies_df)
            thermo_corr = calculate_thermo_corrections('CO*', frequencies, 
                                                 temperature, fugacity_dict['CO*'])
            co_ads_solv_energy = get_solvation_energy('CO*', solvation_df, approach='explicit',    
                                                      manual_solv=manual_solv)
            dF_CO_ads = thermo_corr['F'] + co_ads_solv_energy
            G_CO_ads = e_co + dF_CO_ads
            G_CO_gas = dF_CO_gas
            dG_CO_ads = G_CO_ads - G_CO_gas
            G_CO_dict[metal] = {
                'G': dG_CO_ads,
                'highlight': data['highlight']
            }           

     
        metal_binding_dict[temp_idx] = G_CO_dict
        
        
        # Process HER reactions (11, 12)
        her_rxns = [11, 12]
        bold_rxn_her = []
        for i in range(len(x_values)):
            if delta_G_dict[11][i] < delta_G_dict[12][i]:
                bold_rxn_her.append(11)
            else:
                bold_rxn_her.append(12)
        
        # Plot HER reactions
        for rxn_num in her_rxns:
            y_values = delta_G_dict[rxn_num]
            
            solid_x_her, solid_y_her = [], []
            dash_x_her, dash_y_her = [], []
            
            for i, x in enumerate(x_values):
                if bold_rxn_her[i] == rxn_num:
                    solid_x_her.append(x)
                    solid_y_her.append(y_values[i])
                    solid_y_her_dict[temp_idx].append((x,y_values[i]))
                else:
                    dash_x_her.append(x)
                    dash_y_her.append(y_values[i])
            
        for metal, data in G_CO_dict.items():
            g_co = data['G']
            is_highlight = data['highlight']
            # Find nearest x-value index
            idx = np.abs(np.array(co_binding_range) - g_co).argmin()
            
            # Find which reaction is solid at this point for CO2R and HER
            co2r_solid_rxn = bold_rxn_co2r[idx]
            her_solid_rxn = bold_rxn_her[idx]
            co2r_y = delta_G_dict[co2r_solid_rxn][idx]
            her_y = delta_G_dict[her_solid_rxn][idx]


        

    # Store data for each temperature
    
    print(solid_y_co2r_dict.keys())

    print(solid_y_her_dict.keys())
    
    

    #print(solid_y_co2r_dict)
    
    # Create figure with two subplots
    fig, ax1 = plt.subplots(1, 1, figsize=(3.5*1.2, 2.8*1.2))
    
    # First plot: Full calculation
    X1, X2 = np.meshgrid(co_binding_range, temp_range + 273.15)
    
    Z1 = np.zeros_like(X1)
    print(X1[1,0])
    print(X2[0,1])
    for i in range(len(temp_range)):
        for j in range(len(co_binding_range)):
            co_binding = X1[i,j] 

            T = X2[i,j]

            # Get CO2R limiting potential from bold reactions
            # solid_y_co2r_dict[i] contains list of (x,y) tuples
            # Find tuple where x matches co_binding and get its y value
            co2r_tuple = next((t for t in solid_y_co2r_dict[i] if t[0] == co_binding), None)
            co2r_limiting = -co2r_tuple[1] if co2r_tuple else 0
            
            # Get HER limiting potential from bold reactions  
            her_tuple = next((t for t in solid_y_her_dict[i] if t[0] == co_binding), None)
            her_limiting = -her_tuple[1] if her_tuple else 0
            
            if i == 25 and j == 25:
                print(f"{co_binding}, {T}")
                print(f"T: {T}, co2r_limiting: {co2r_limiting}, her_limiting: {her_limiting}")
            # Calculate rate constants and ratio
            k_co2r = (k_b * T / h) * np.exp(-co2r_limiting / (k_b * T))
            k_her = (k_b * T / h) * np.exp(-her_limiting / (k_b * T))
            Z1[i,j] = k_co2r / k_her


    # Create lines for each metal
    metals_data = {}
    for metal in metal_binding_dict[0].keys():
        temps = []
        g_cos = []
        for i in range(len(temp_range)):
            temps.append(temp_range[i])
            g_cos.append(metal_binding_dict[i][metal]['G'])
        metals_data[metal] = {'temps': temps, 'g_cos': g_cos}

        
    # Plot line for each metal and add label next to line
    # Define vertical offsets for each metal relative to Fe (which is at 0)
    vertical_offsets = {}
    # Get g_cos values at middle temperature point for ranking
    mid_temp_idx = len(temp_range) // 2
    g_cos_values = {metal: data['g_cos'][mid_temp_idx] for metal, data in metals_data.items()}
    
    # Sort metals by g_cos values
    sorted_metals = sorted(g_cos_values.items(), key=lambda x: x[1])
    
    # Find Cu's rank
    cu_rank = next(i for i, (metal, _) in enumerate(sorted_metals) if metal == 'Cu')
    
    # Set offsets based on distance from Cu's rank
    for rank, (metal, _) in enumerate(sorted_metals):
        if rank < cu_rank:
            # Metals with more negative g_cos than Cu
            vertical_offsets[metal] = (rank - cu_rank) * 10
        elif rank > cu_rank:
            # Metals with more positive g_cos than Cu
            vertical_offsets[metal] = (rank - cu_rank) * 10
        else:
            # Cu itself
            vertical_offsets[metal] = 0

    # vertical_offsets = {
    #     'Pt': -15,  # Pt label below Fe
    #     'Ni': -10,  # Ni label below Fe
    #     'Fe': 0,    # Fe at center
    #     'Cu': 5,    # Cu label above Fe
    #     'Au': 10,   # Au label above Fe
    #     'Ag': 15    # Ag label above Fe ch
    # }
    print(vertical_offsets)
    horizontal_offsets = {'Pd': -1,
                           'Ni':3,
                           'Pt':-4,
                           'Rh':-9,
                           'Cu':3,
                           'Au':5,
                           'Ag':6,
                           'Zn':7,
                           }
    for metal, data in metals_data.items():
        line = ax1.plot(data['g_cos'], data['temps'], ':', linewidth=1.5, color='black')[0]
        # Add label at the middle point with vertical offset
        mid_idx = int(len(data['temps']) * 0.25)  # 25% point from bottom
        offset = vertical_offsets.get(metal, 0)  # Get offset for this metal, default 0
        horizontal_offset = horizontal_offsets.get(metal, 0)
        ax1.annotate(metal, 
                    xy=(data['g_cos'][mid_idx], data['temps'][mid_idx]),
                    xytext=(horizontal_offset, offset),  # 5 points right, variable vertical offset
                    textcoords='offset points',
                    va='center',
                    color='black')
                
    
    # Plot settings for Full Calculation
    log_scale = (-14, -2)
    ticks_num = log_scale[1] - log_scale[0] + 1   
    levels = np.logspace(log_scale[0], log_scale[1], 40)
    #RdBu_r


    palette = 'rocket'
    cmap = sns.color_palette(palette, as_cmap=True)
    cmap = 'RdBu_r'
    cmap = 'Reds'
    contour = ax1.contourf(X1, X2-273.15, Z1, levels=levels, norm=LogNorm(), cmap=cmap, alpha=1.0)
    
    # Add colorbar with integer ticks

    cbar = plt.colorbar(contour, ax=ax1, ticks=np.logspace(log_scale[0], log_scale[1], ticks_num//2))
    cbar.set_label('k(T)$_{COR}$ / k(T)$_{HER}$', rotation=270, labelpad=10, fontproperties=helvetica_bold_prop, fontsize=12)
    # cbar.ax.set_yticklabels(['$10^{-14}$', '$10^{-13}$', '$10^{-12}$', '$10^{-11}$', 
    #                         '$10^{-10}$', '$10^{-9}$', '$10^{-8}$', '$10^{-7}$',
    #                         '$10^{-6}$', '$10^{-5}$', '$10^{-4}$', '$10^{-3}$',
    #                         '$10^{-2}$', '$10^{-1}$', '$10^{0}$', '$10^{1}$',
    #                         '$10^{2}$', '$10^{3}$', '$10^{4}$', '$10^{5}$',
    #                         '$10^{6}$', '$10^{7}$'])
    # Add k_CO2R = k_HER line
    CS = ax1.contour(X1, X2-273.15, Z1, levels=[1.0], colors='black', 
                   linestyles='--', linewidths=2)
    # CS = ax1.contour(X1, X2-273.15, Z1, levels=[1e-4], colors='black', alpha=0.5, 
    #                linestyles='--', linewidths=1)
    # #ax1.clabel(CS, inline=True, fmt='k$_{CO2R}$ = k$_{HER}$')
    # ax1.clabel(CS, inline=True, fmt='1e-4')

    # CS2 = ax1.contour(X1, X2-273.15, Z1, levels=[1e-5], colors='black', alpha=0.5,
    #                linestyles='--', linewidths=1)
    # ax1.clabel(CS2, inline=True, fmt='1e-5')
    
    # Labels and title
    ax1.set_xlabel('ΔG$_{CO}$ (eV)', fontproperties=helvetica_bold_prop)
    ax1.set_ylabel('Temperature (°C)', fontproperties=helvetica_bold_prop)
    y_max = temp_range[-1]
    print(y_max)
    #ax1.set_title('Full Calculation')
    ax1.set_ylim(0, y_max)
    ax1.set_xlim(co_binding_range[0], co_binding_range[-1])
    ax1.set_xticks(np.arange(co_binding_range[0], co_binding_range[-1]+0.1, 0.4))
    ax1.set_xticks(np.arange(co_binding_range[0], co_binding_range[-1]+0.1, 0.2), minor=True)
    ax1.set_yticks(np.arange(0, y_max+1, 50))
    ax1.set_yticks(np.arange(0, y_max+1, 25), minor=True)
    ax1.tick_params(direction='in', which='both')
    ax1.tick_params(axis='x', which='minor', length=2)
    ax1.tick_params(axis='x', which='major', length=4)
    ax1.tick_params(axis='y', which='minor', length=2)
    ax1.tick_params(axis='y', which='major', length=4)
    # 테두리 설정
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)
    # 테두리(스파인) 설정
    ax1.spines['left'].set_visible(True)
    ax1.spines['right'].set_visible(True)
    ax1.spines['top'].set_visible(True)
    ax1.spines['bottom'].set_visible(True)
    plt.tight_layout()

    # save
    plt.savefig('volcano_dG_CO.png', dpi=300)
    return fig


# Create plots
#co_binding_range1 = np.linspace(-2.0, 0.5, 30)
# co_binding_range2 = np.linspace(-1.2, 0.2, 50)
# temp_range = np.linspace(0, 140, 50)
#temperatures = np.array([273, 298])  # K
# fig = create_reaction_rate_contours(co_binding_range1, temp_range, 
#                                   fugacity_dict, manual_solv, 
#                                   scaling_relation_100, E_CO_dict_100)
# fig = create_reaction_rate_contours_dE_CO(co_binding_range2, temp_range, 
#                                   fugacity_dict, manual_solv, 
#                                   scaling_relation_100, E_CO_dict_100)