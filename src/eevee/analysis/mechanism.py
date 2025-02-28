# src/eevee/analysis/mechanism.py
from typing import Dict, List
from catmap import analyze
import pandas as pd
from ..core.thermodynamics import ThermoCorrector
from ..io.readers.input_reader import InputReader, MKMReader
import matplotlib.pyplot as plt
from catmap import ReactionModel
from pathlib import Path
import tempfile
import shutil
import numpy as np

class MechanismAnalyzer:
    """Analyzer for reaction mechanisms"""
    
    def __init__(self, mkm_path: str, input_path: str, temperature: float = None):
        self.mkm_path = mkm_path
        self.model = None
        self.input_path = input_path
        # Extract temperature from input filename if not provided
        if temperature is None:
            self.base_temperature = self._extract_temperature_from_input()
        else:
            self.base_temperature = temperature
        self.setup_model()

    def _extract_temperature_from_input(self) -> float:
        """Extract temperature from input filename (e.g., Cu100_Gibbs_323K.txt -> 323)"""
        try:
            # First try to get temperature from input filename
            input_name = Path(self.input_path).stem  # Get filename without extension
            temp_k = float(input_name.split('_')[-1].rstrip('K'))
            return temp_k
        except (ValueError, IndexError):
            # If that fails, try to read from mkm file
            try:
                with open(self.mkm_path, 'r') as f:
                    mkm_content = f.read()
                    for line in mkm_content.split('\n'):
                        if 'input_file' in line:
                            input_file = line.split('=')[1].strip().strip("'").strip('"')
                            temp_k = float(input_file.split('_')[-1].rstrip('K.txt'))
                            self.input_path = input_file
                            return temp_k
            except (ValueError, IndexError, FileNotFoundError):
                # If both methods fail, return default temperature
                return 300.0

    def setup_model(self) -> None:
        """
        TODO: 
        - move this function to input reader

        Setup CatMAP model with specified input file path
        
        Parameters
        ----------
        input_path : str
            Path to input file to use for this analysis
        """
        # Create temporary mkm file with updated input_file path
        temp_mkm = self._create_temp_mkm()
        
        try:
            # Initialize model with temporary mkm file
            self.model = ReactionModel(setup_file=temp_mkm)
            # Additional model setup if needed
            self.model.adsorbate_interaction_model = None
            self.ma = analyze.MechanismAnalysis(self.model)
            self.rxn_mechanisms = self.model.rxn_mechanisms
            self._configure_analysis()
            self.get_all_rxn_steps()

        finally:
            # Clean up temporary file
            if temp_mkm:
                Path(temp_mkm).unlink()

    def _create_temp_mkm(self) -> str:
        """
        Create temporary mkm file with updated input_file path
        
        Parameters
        ----------
        input_path : str
            New input file path to use
        
        Returns
        -------
        str
            Path to temporary mkm file
        """
        # Read original mkm content
        with open(self.mkm_path, 'r') as f:
            mkm_content = f.read()
            
        # Update input_file path
        abs_input_path = str(Path(self.input_path).resolve())
        if "input_file" in mkm_content:
            # Replace existing input_file line
            mkm_content = self._replace_input_file_line(mkm_content, abs_input_path)
        else:
            # Add input_file line if not present
            mkm_content += f"\ninput_file = '{abs_input_path}'\n"
            
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.mkm',
            delete=False
        )
        temp_file.write(mkm_content)
        temp_file.close()
        
        return temp_file.name

    def _replace_input_file_line(self, content: str, new_path: str) -> str:
        """Replace input_file line in mkm content"""
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('input_file'):
                lines[i] = f"input_file = '{new_path}'"
                break
        return '\n'.join(lines)


    def _configure_analysis(self):
        """Configure analysis parameters"""
        self.ma.coverage_correction = True
        self.ma.pressure_correction = True
        self.ma.energy_type = 'free_energy'
        self.ma.include_labels = True
        self.ma.energy_lines = True
        self.ma.temperature = self.base_temperature
        

    def get_all_rxn_steps(self):
        len_rxn_expressions = len(self.model.rxn_expressions)
        all_rxn_steps = list(range(1, len_rxn_expressions+1))
        self.rxn_mechanisms['all_rxn_steps'] = all_rxn_steps

    def _get_mechanism_data(self, potential: float, temperature: float):
        """Get mechanism data at specified potential and temperature"""

        # Update model
        self.ma.descriptor_ranges = [[potential, potential], [temperature, temperature]]

        # Generate plot data
        fig = self.ma.plot(save=False, plot_variants=[potential])
        plt.close(fig)

        return self.ma.data_dict

    def _process_mechanism_data(self, 
                         mechanism_name: str,
                         delta_g_list: List[float],
                         ga_list: List[float]):
        """Analyze specific mechanism at given potentials"""

        # 1) 
        rxn_indices = self.rxn_mechanisms[mechanism_name]

        delta_g_list = delta_g_list[1:]

        # RDS
        max_ga = max(ga_list)

        # PDS
        max_delta_g = max(delta_g_list)

        # Initialize cumulative IS (intermediate state) energy
        IS = 0.0
        
        # List to store data for all steps
        steps_data = []

        for step_idx, (dg, ga) in enumerate(zip(delta_g_list, ga_list)):
            # Update intermediate state energy
            IS += dg
            
            # Get reaction index and expression
            rxn_idx = rxn_indices[step_idx]
            rxn_str = self.model.rxn_expressions[rxn_idx-1]
            
            # Round values
            ga_rounded = round(ga, 2)
            dg_rounded = round(dg, 2)
            
            # Determine if this step is RDS or PDS
            is_rds = (ga == max_ga)
            is_pds = (dg == max_delta_g)
            
            # Create dictionary for this step
            step_data = {
                'step_idx': step_idx + 1,  # 1-based indexing
                'rxn_idx': rxn_idx,
                'Reaction': rxn_str,
                'Ga': ga_rounded,
                'dG': dg_rounded,
                'IS': round(IS, 2),
                'is_RDS': is_rds,
                'is_PDS': is_pds
            }
            # Add to list of steps
            steps_data.append(step_data)
        
        return steps_data

    def mechanism_to_dataframe(self, potential: float = 0.0, temperature: float = None) -> pd.DataFrame:
        """
        Convert mechanism analysis data to DataFrame for a given potential and temperature
        """
        # Use instance temperature if not specified
        analysis_temp = temperature or self.base_temperature  
        data_dict = self._get_mechanism_data(potential, analysis_temp)

        if not hasattr(self.ma, 'data_dict') or not self.ma.data_dict:
            raise ValueError("ma.data_dict is empty. Run ma.plot() first.")
        
         

        all_steps_data = []
        for mechanism, (delta_g_list, ga_list) in data_dict.items():
            if not mechanism == 'all_rxn_steps':
                continue
                
            # Get data for all steps in this mechanism
            steps_data = self._process_mechanism_data(
                mechanism, delta_g_list, ga_list
            )
            all_steps_data.extend(steps_data)
        
        # Convert to DataFrame
        gibbs_all_df = pd.DataFrame(all_steps_data)
        
        # Specify column order
        col_order = ['step_idx', 'rxn_idx', 'Reaction', 
                    'Ga', 'dG', 'IS', 'is_RDS', 'is_PDS']
        gibbs_all_df = gibbs_all_df[col_order]
        
        return gibbs_all_df

# New plotting method
def plot_ga_vs_temperature(base_dir: str, potentials: List[float], steps_to_plot: List[int]):
    """
    Plot Ga vs temperature for different electrolyte conditions and potentials
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing ElecTemp folders
    potentials : list
        List of potential values to analyze (V vs RHE)
    steps_to_plot : list
        List of step indices to include in the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Define electrolyte temperatures and colors
    elec_temps = [0, 25, 70]  # °C
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Create subplots for each potential
    fig, axes = plt.subplots(1, len(potentials), figsize=(20, 6))
    
    for potential, ax in zip(potentials, axes):
        for elec_temp, color in zip(elec_temps, colors):
            temps = []
            ga_values = {step: [] for step in steps_to_plot}
            
            # Collect data from cathode temperature folders
            elec_dir = Path(base_dir) / f"ElecTemp_{elec_temp}K"
            for cath_dir in elec_dir.glob("CathodeTemp_*K"):
                # Extract temperature from folder name
                temp_k = int(cath_dir.name.split('_')[1].rstrip('K'))
                
                try:
                    # Initialize analyzer with temperature-specific input
                    
                    #input_path = 
                    mkm_path = cath_dir / "CO2R.mkm"
                    
                    analyzer = MechanismAnalyzer(
                        mkm_path=str(mkm_path),
                        input_path=str(input_path),
                        temperature=temp_k
                    )
                    
                    # Get mechanism data
                    df = analyzer.mechanism_to_dataframe(potential=potential)
                    
                    # Store data for selected steps
                    for step in steps_to_plot:
                        step_data = df[df['step_idx'] == step]
                        if not step_data.empty:
                            ga_values[step].append(step_data['Ga'].values[0])
                        else:
                            ga_values[step].append(np.nan)
                    
                    temps.append(temp_k)
                    
                except Exception as e:
                    print(f"Error processing {cath_dir}: {str(e)}")
                    continue

            # Plot each step
            for step in steps_to_plot:
                ax.plot(temps, ga_values[step], 
                        marker='o', linestyle='--', color=color,
                        label=f'Step {step} ({elec_temp}°C)' if potential == potentials[0] else '')
        
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Ga (eV)')
        ax.set_title(f'Potential: {potential} V')
        ax.grid(True, alpha=0.3)
    
    # Create unified legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{base_dir}/ga_vs_temperature.png', bbox_inches='tight', dpi=300)
    plt.close()



