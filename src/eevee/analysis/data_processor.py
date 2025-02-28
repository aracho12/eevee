# src/eevee/analysis/data_processor.py

from ..io.readers.input_reader import InputReader
from ..core.thermodynamics import ThermoCorrector
import pandas as pd
from pathlib import Path
from typing import List


def create_gibbs_table(
    input_path: str, 
    temperature: float = 300, 
    fugacity_dict: dict = None
):
    """
    Create a table of Gibbs free energies and thermodynamic corrections
    
    Parameters
    ----------
    input_path : str
        Path to input file
    temperature : float, optional
        Temperature in K, default 300K
    fugacity_dict : dict, optional
        Dictionary of fugacities for gas species (e.g., {'H2': 1e5, 'CO2': 1e5})
        If None, default fugacity of 1e5 Pa is used for all gas species
    
    Returns
    -------
    pd.DataFrame
        Extended DataFrame containing original data and Gibbs free energies
    """
    # Default fugacity dictionary
    if fugacity_dict is None:
        fugacity_dict = {}
    
    # 1) Load input file
    input_reader = InputReader(input_path)
    energy_df = input_reader.energy_df.copy()

    # 2) Calculate free energy for each species using ThermoCorrector
    def calculate_gibbs(row):
        species_name = row['species_name']
        # Get species-specific fugacity
        fugacity = fugacity_dict.get(species_name, 1e5)
        
        thermo_corrector = ThermoCorrector(row)
        status, ZPE, Cp, H, dS, TS, F = thermo_corrector.get_free_energy(temperature, fugacity)
        E = row['formation_energy']
        G = E + F

        # round 3
        return pd.Series({
            'status': status,
            'E': round(E, 3),
            'ZPE': round(ZPE, 3),
            'Cp': round(Cp, 3),
            'H': round(H, 3),
            'dS': round(dS, 3),
            'TS': round(TS, 3),
            'F': round(F, 3),
            'G': round(G, 3)
        })
    
    # Calculate Gibbs corrections and add to energy_df
    gibbs_corrections = energy_df.apply(calculate_gibbs, axis=1)
    extended_df = pd.concat([energy_df, gibbs_corrections], axis=1)
    
    return extended_df

def compare_multiple_gibbs_tables(
    input_paths: List[str],
    temperature: float = 298.15,
    fugacity: float = 1e5,
    threshold: float = 0.01,
    columns_to_compare: List[str] = ['E', 'F', 'G']
) -> pd.DataFrame:
    """
    Compare Gibbs free energy tables from multiple input files
    """
    # Store results for each input file
    all_results = {}
    all_species = set()
    
    # Process each input file
    for input_path in input_paths:
        input_name = Path(input_path).stem
        gibbs_df = create_gibbs_table(input_path, temperature, fugacity)
        
        # Round numeric columns
        numeric_cols = gibbs_df.select_dtypes(include=['float64']).columns
        gibbs_df[numeric_cols] = gibbs_df[numeric_cols].round(2)
        
        all_results[input_name] = gibbs_df
        all_species.update(gibbs_df['species_name'])
    
    # Create comparison DataFrame with MultiIndex columns
    comparison_data = []
    
    # Create column MultiIndex tuples
    columns = [
        ('Info', 'species_name'),
        ('Info', 'status')
    ]
    
    # Add columns for each input file (without diff columns)
    for input_path in input_paths:
        input_name = Path(input_path).stem
        for col in columns_to_compare:
            columns.append((input_name, col))
    
    # Add diff columns at the end
    for col in columns_to_compare:
        columns.append(('Diff', col))
    
    # Create MultiIndex columns
    column_index = pd.MultiIndex.from_tuples(
        columns,
        names=['Source', 'Value']
    )
    
    # For each species
    for species in sorted(all_species):
        species_data = {
            ('Info', 'species_name'): species
        }
        
        # Get status from first file that has this species
        for df in all_results.values():
            species_row = df[df['species_name'] == species]
            if not species_row.empty:
                species_data[('Info', 'status')] = species_row.iloc[0]['status']
                break
        
        # Get values from each input file
        for input_path in input_paths:
            input_name = Path(input_path).stem
            df = all_results[input_name]
            species_row = df[df['species_name'] == species]
            
            # Add values
            for col in columns_to_compare:
                value = species_row[col].iloc[0] if not species_row.empty else None
                species_data[(input_name, col)] = value
        
        # Calculate differences (second file - first file)
        first_name = Path(input_paths[0]).stem
        second_name = Path(input_paths[1]).stem
        
        for col in columns_to_compare:
            first_val = species_data.get((first_name, col))
            second_val = species_data.get((second_name, col))
            
            if first_val is not None and second_val is not None:
                diff = second_val - first_val
                species_data[('Diff', col)] = diff
            else:
                species_data[('Diff', col)] = None
        
        comparison_data.append(species_data)
    
    # Create DataFrame with MultiIndex columns
    comparison_df = pd.DataFrame(comparison_data, columns=column_index)
    
    # Round numeric columns
    numeric_cols = comparison_df.select_dtypes(include=['float64']).columns
    comparison_df[numeric_cols] = comparison_df[numeric_cols].round(2)
    
    # Add metadata for styling
    comparison_df.attrs['threshold'] = threshold
    
    return comparison_df

def style_differences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create styled DataFrame with highlighted differences
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from compare_multiple_gibbs_tables
        
    Returns
    -------
    pd.DataFrame
        Styled DataFrame with highlighted differences
    """
    threshold = df.attrs.get('threshold', 0.01)
    
    def highlight_diff(val, col):
        # Check if this is a difference column (under 'Diff' source)
        if col[0] == 'Diff':
            if pd.isna(val):
                return None
            if abs(val) > threshold:
                return 'background-color: #ffcdd2' if val > 0 else 'background-color: #c8e6c9'
        return None
    
    # Format numbers and apply highlighting
    styled = df.style.apply(lambda x: [highlight_diff(v, c) for v, c in zip(x, df.columns)], axis=1)
    
    # Format numeric columns to 2 decimal places
    numeric_cols = df.select_dtypes(include=['float64']).columns
    styled = styled.format("{:.2f}", subset=numeric_cols)
    
    return styled

def style_mechanism_differences(df: pd.DataFrame) -> pd.DataFrame:
    """
    스타일이 적용된 DataFrame을 반환합니다.
    
    Parameters
    ----------
    df : pd.DataFrame
        compare_mechanism_differences()에서 반환된 DataFrame
        
    Returns
    -------
    pd.DataFrame
        스타일이 적용된 DataFrame
    """
    threshold = df.attrs.get('threshold', 0.05)
    
    def highlight_diff(val, col):
        """단일 값에 대한 스타일링"""
        # Check if this is a difference column (under 'Diff' source)
        if col[0] == 'Diff':
            if pd.isna(val):
                return None
            if abs(val) > threshold:
                return 'background-color: #ffcdd2' if val > 0 else 'background-color: #c8e6c9'
        return None
    
    # Format numbers and apply highlighting
    styled = df.style.apply(lambda x: [highlight_diff(v, c) for v, c in zip(x, df.columns)], axis=1)
    
    # Format numeric columns to 2 decimal places
    numeric_cols = df.select_dtypes(include=['float64']).columns
    styled = styled.format("{:.2f}", subset=numeric_cols)
    
    return styled

def compare_multiple_inputs(
    input_paths: List[str],
    mkm_path: str,
    temperature: float = 300.0,
    potential: float = 0.0,
    threshold: float = 0.05
) -> pd.DataFrame:
    """
    Compare reaction mechanisms for multiple input files using the same MKM file
    
    Parameters
    ----------
    input_paths : list[str]
        List of paths to input files
    mkm_path : str
        Path to MKM file
    temperature : float, optional
        Temperature in K, default 300K
    potential : float, optional
        Applied potential (V vs. RHE), default 0.0V
    threshold : float, optional
        Minimum difference to highlight, by default 0.05 eV
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing comparison of dG and Ga for each input
    """
    from .mechanism import MechanismAnalyzer
    
    # Store results for each input file
    all_results = {}
    
    # Initialize analyzers with specified temperature
    for input_path in input_paths:
        input_name = Path(input_path).stem
        analyzer = MechanismAnalyzer(
            mkm_path=mkm_path,
            input_path=input_path,
            temperature=temperature  # Pass temperature to analyzer
        )
        
        # Get mechanism data
        df = analyzer.mechanism_to_dataframe(potential=potential)
        all_results[input_name] = df
    
    # Create comparison DataFrame
    comparison_data = []
    
    # Get reference input (first one)
    ref_input = list(all_results.keys())[0]
    ref_df = all_results[ref_input]
    
    # For each reaction step
    for idx, ref_row in ref_df.iterrows():
        step_data = {
            'step_idx': ref_row['step_idx'],
            'rxn_idx': ref_row['rxn_idx'],
            'Reaction': ref_row['Reaction'],
            f'{ref_input}_dG': ref_row['dG'],
            f'{ref_input}_Ga': ref_row['Ga'],
        }
        
        # Add data from other inputs
        for input_name, df in all_results.items():
            if input_name == ref_input:
                continue
                
            row = df[df['rxn_idx'] == ref_row['rxn_idx']].iloc[0]
            step_data[f'{input_name}_dG'] = row['dG']
            step_data[f'{input_name}_Ga'] = row['Ga']
            
            # Calculate differences
            step_data[f'{input_name}_dG_diff'] = row['dG'] - ref_row['dG']
            step_data[f'{input_name}_Ga_diff'] = row['Ga'] - ref_row['Ga']
        
        comparison_data.append(step_data)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Round numeric columns to 2 decimal places
    numeric_cols = comparison_df.select_dtypes(include=['float64']).columns
    comparison_df[numeric_cols] = comparison_df[numeric_cols].round(2)
    
    # Add metadata for styling
    comparison_df.attrs['threshold'] = threshold
    
    return comparison_df

def save_mechanism_comparison(
    df: pd.DataFrame,
    output_path: str,
    threshold: float = 0.05,
    index: bool = False
) -> None:
    """
    스타일이 적용된 메커니즘 비교 테이블을 HTML 파일로 저장합니다.
    
    Parameters
    ----------
    df : pd.DataFrame
        비교할 DataFrame
    output_path : str
        저장할 HTML 파일 경로
    threshold : float, optional
        강조 표시할 차이 값의 임계값, 기본값 0.05 eV
    index : bool, optional
        인덱스 포함 여부, 기본값 False
    """
    styled_df = style_mechanism_differences(df)
    
    # Convert output_path to Path object
    output_path = Path(output_path)
    
    # Ensure the path has .html extension
    if output_path.suffix != '.html':
        output_path = output_path.with_suffix('.html')
        
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to HTML
    if not index:
        styled_df = styled_df.hide(axis='index')
    
    styled_df.to_html(output_path)

def compare_temperatures(
    input_path: str,
    temperatures: List[float],
    fugacity: float = 1e5,
    threshold: float = 0.01,
    columns_to_compare: List[str] = ['E', 'F', 'G']
) -> pd.DataFrame:
    """
    Compare Gibbs free energy tables at different temperatures
    
    Parameters
    ----------
    input_path : str
        Path to input file
    temperatures : List[float]
        List of temperatures in K to compare
    fugacity : float, optional
        Fugacity in Pa, by default 1e5
    threshold : float, optional
        Minimum difference to highlight, by default 0.01 eV
    columns_to_compare : List[str], optional
        Columns to compare, by default ['E', 'F', 'G']
        
    Returns
    -------
    pd.DataFrame
        DataFrame comparing energies at different temperatures
    """
    # Calculate Gibbs energies for each temperature
    all_results = {}
    for temp in temperatures:
        gibbs_df = create_gibbs_table(
            input_path=input_path,
            temperature=temp,
            fugacity=fugacity
        )
        all_results[f"{temp}K"] = gibbs_df
        
    # Get all unique species
    all_species = set()
    for df in all_results.values():
        all_species.update(df['species_name'])
    
    # Prepare comparison data
    comparison_data = []
    ref_temp = temperatures[0]
    ref_name = f"{ref_temp}K"
    
    # For each species
    for species in sorted(all_species):
        species_data = {'species_name': species}
        
        # Get status from first temperature result
        for df in all_results.values():
            species_row = df[df['species_name'] == species]
            if not species_row.empty:
                species_data['status'] = species_row.iloc[0]['status']
                break
        
        # For each column to compare
        for col in columns_to_compare:
            # Add values from each temperature
            for temp_name, df in all_results.items():
                species_row = df[df['species_name'] == species]
                value = species_row[col].iloc[0] if not species_row.empty else None
                species_data[f"{temp_name}_{col}"] = value
                
                # Calculate differences from reference temperature
                if temp_name != ref_name:
                    ref_row = all_results[ref_name][all_results[ref_name]['species_name'] == species]
                    if not ref_row.empty and not species_row.empty:
                        diff = value - ref_row[col].iloc[0]
                        species_data[f"{temp_name}_{col}_diff"] = diff
        
        comparison_data.append(species_data)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Add metadata for styling
    comparison_df.attrs['columns_to_compare'] = columns_to_compare
    comparison_df.attrs['threshold'] = threshold
    comparison_df.attrs['reference_temp'] = ref_temp
    
    # Round numeric columns
    numeric_cols = comparison_df.select_dtypes(include=['float64']).columns
    comparison_df[numeric_cols] = comparison_df[numeric_cols].round(2)
    
    return comparison_df

def style_temperature_differences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create styled DataFrame with highlighted temperature differences
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from compare_temperatures
        
    Returns
    -------
    pd.DataFrame
        Styled DataFrame with highlighted differences
    """
    if 'columns_to_compare' not in df.attrs:
        raise ValueError("DataFrame must be from compare_temperatures()")
        
    threshold = df.attrs.get('threshold', 0.01)
    
    def highlight_diff(val, col):
        if not col.endswith('_diff'):
            return None
        if pd.isna(val):
            return None
        if abs(val) > threshold:
            if val > 0:
                return 'background-color: #ffcdd2'  # Red (positive)
            else:
                return 'background-color: #c8e6c9'  # Green (negative)
        return None
    
    # Format numbers and apply highlighting
    styled = df.style.apply(lambda x: [highlight_diff(v, c) for v, c in zip(x, df.columns)], axis=1)
    
    # Format numeric columns to 3 decimal places
    numeric_cols = df.select_dtypes(include=['float64']).columns
    styled = styled.format("{:.3f}", subset=numeric_cols)
    
    return styled

def save_temperature_comparison(
    df: pd.DataFrame,
    output_path: str,
    threshold: float = 0.01,
    index: bool = False
) -> None:
    """
    Save styled temperature comparison table to HTML file
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from compare_temperatures
    output_path : str
        Path to save the HTML file
    threshold : float, optional
        Minimum difference to highlight, by default 0.01 eV
    index : bool, optional
        Whether to include index in output, by default False
    """
    styled_df = style_temperature_differences(df)
    
    # Convert output_path to Path object
    output_path = Path(output_path)
    
    # Ensure the path has .html extension
    if output_path.suffix != '.html':
        output_path = output_path.with_suffix('.html')
        
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to HTML
    styled_df.hide(axis='index').to_html(output_path)

def compare_mechanism_temperatures(
    input_path: str,
    mkm_path: str,
    temperatures: List[float],
    potential: float = 0.0,
    threshold: float = 0.05
) -> pd.DataFrame:
    """
    Compare reaction mechanisms at different temperatures
    
    Parameters
    ----------
    input_path : str
        Path to input file
    mkm_path : str
        Path to MKM file
    temperatures : List[float]
        List of temperatures in K to compare
    potential : float, optional
        Applied potential (V vs. RHE), default 0.0V
    threshold : float, optional
        Minimum difference to highlight, by default 0.05 eV
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing mechanism comparison at different temperatures
    """
    from .mechanism import MechanismAnalyzer
    
    # Store results for each temperature
    all_results = {}
    
    # Analyze mechanism at each temperature
    for temp in temperatures:
        print(f"\nAnalyzing temperature: {temp}K")
        analyzer = MechanismAnalyzer(
            mkm_path=mkm_path,
            input_path=input_path,
            temperature=temp
        )
        df = analyzer.mechanism_to_dataframe(potential=potential)
        all_results[f"{temp}K"] = df
    
    # Create comparison DataFrame with MultiIndex columns
    comparison_data = []
    ref_temp = temperatures[0]
    ref_name = f"{ref_temp}K"
    ref_df = all_results[ref_name]
    
    # Create column MultiIndex tuples
    columns = [
        ('Info', 'step_idx'),
        ('Info', 'rxn_idx'),
        ('Info', 'Reaction')
    ]
    
    for temp in temperatures:
        temp_name = f"{temp}K"
        # Add base columns
        columns.extend([
            (temp_name, 'dG'),
            (temp_name, 'Ga'),
            (temp_name, 'IS')
        ])
        # Add diff columns for non-reference temperatures
        if temp != ref_temp:
            columns.extend([
                (temp_name, 'dG_diff'),
                (temp_name, 'Ga_diff'),
                (temp_name, 'IS_diff')
            ])
    
    # Create MultiIndex columns
    column_index = pd.MultiIndex.from_tuples(
        columns,
        names=['Temperature', 'Value']
    )
    
    # For each reaction step
    for idx, ref_row in ref_df.iterrows():
        row_data = {
            ('Info', 'step_idx'): ref_row['step_idx'],
            ('Info', 'rxn_idx'): ref_row['rxn_idx'],
            ('Info', 'Reaction'): ref_row['Reaction']
        }
        
        # Add data for each temperature
        for temp in temperatures:
            temp_name = f"{temp}K"
            df = all_results[temp_name]
            row = df[df['rxn_idx'] == ref_row['rxn_idx']].iloc[0]
            
            # Base values
            row_data[(temp_name, 'dG')] = row['dG']
            row_data[(temp_name, 'Ga')] = row['Ga']
            row_data[(temp_name, 'IS')] = row['IS']
            
            # Difference values (for non-reference temperatures)
            if temp != ref_temp:
                row_data[(temp_name, 'dG_diff')] = row['dG'] - ref_row['dG']
                row_data[(temp_name, 'Ga_diff')] = row['Ga'] - ref_row['Ga']
                row_data[(temp_name, 'IS_diff')] = row['IS'] - ref_row['IS']
        
        comparison_data.append(row_data)
    
    # Create DataFrame with MultiIndex columns
    comparison_df = pd.DataFrame(comparison_data, columns=column_index)
    
    # Round numeric columns
    numeric_cols = comparison_df.select_dtypes(include=['float64']).columns
    comparison_df[numeric_cols] = comparison_df[numeric_cols].round(2)
    
    # Add metadata for styling
    comparison_df.attrs['threshold'] = threshold
    comparison_df.attrs['reference_temp'] = ref_temp
    
    return comparison_df

def style_mechanism_temperature_differences(df: pd.DataFrame) -> pd.DataFrame:
    threshold = df.attrs.get('threshold', 0.05)
    
    def highlight_diff(val):
        """단일 값에 대한 스타일링"""
        try:
            # MultiIndex에서 현재 열 이름 가져오기
            col_name = highlight_diff.col_name
            if not col_name[1].endswith('_diff'):
                return None
            if pd.isna(val):
                return None
            if abs(val) > threshold:
                if val > 0:
                    return 'background-color: #ffcdd2'  # Red (positive)
                else:
                    return 'background-color: #c8e6c9'  # Green (negative)
        except:
            return None
        return None
    
    # Format numbers and apply highlighting
    styled = df.style
    
    # 각 열에 대해 개별적으로 스타일 적용
    for col in df.columns:
        if col[1].endswith('_diff'):
            highlight_diff.col_name = col  # 현재 열 이름 저장
            styled = styled.applymap(highlight_diff, subset=[col])
    
    # Format numeric columns to 2 decimal places
    numeric_cols = df.select_dtypes(include=['float64']).columns
    styled = styled.format("{:.2f}", subset=numeric_cols)
    
    return styled

def save_mechanism_temperature_comparison(
    df: pd.DataFrame,
    output_path: str,
    threshold: float = 0.05,
    index: bool = False
) -> None:
    """
    Save styled mechanism temperature comparison table to HTML file
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from compare_mechanism_temperatures
    output_path : str
        Path to save the HTML file
    threshold : float, optional
        Minimum difference to highlight, by default 0.05 eV
    index : bool, optional
        Whether to include index in output, by default False
    """
    styled_df = style_mechanism_temperature_differences(df)
    
    # Convert output_path to Path object
    output_path = Path(output_path)
    
    # Ensure the path has .html extension
    if output_path.suffix != '.html':
        output_path = output_path.with_suffix('.html')
        
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to HTML
    styled_df.hide(axis='index').to_html(output_path)

def compare_mechanism_differences(
    input_paths: List[str],
    mkm_path: str,
    temperature: float = 300,
    potential: float = 0.0,
    threshold: float = 0.05
) -> pd.DataFrame:
    """메커니즘 비교 함수"""
    from .mechanism import MechanismAnalyzer
    
    if not input_paths:
        raise ValueError("At least one input path must be provided")
    
    # 기준 메커니즘 분석
    base_path = input_paths[0]
    base_name = Path(base_path).stem
    base_analyzer = MechanismAnalyzer(
        mkm_path=mkm_path,
        input_path=base_path,
        temperature=temperature
    )
    base_df = base_analyzer.mechanism_to_dataframe(potential=potential)
    
    # Create column MultiIndex tuples
    columns = [
        ('Info', 'step_idx'),
        ('Info', 'rxn_idx'),
        ('Info', 'Reaction')
    ]
    
    # Add columns for base file
    columns.extend([
        (base_name, 'dG'),
        (base_name, 'Ga')
    ])
    
    # 비교할 메커니즘들 분석
    all_results = {}
    for comp_path in input_paths[1:]:  # Skip the first (base) path
        name = Path(comp_path).stem
        analyzer = MechanismAnalyzer(
            mkm_path=mkm_path,
            input_path=comp_path,
            temperature=temperature
        )
        df = analyzer.mechanism_to_dataframe(potential=potential)
        all_results[name] = df
        
        # Add columns for comparison files
        columns.extend([
            (name, 'dG'),
            (name, 'Ga')
        ])
    
    # Add diff columns at the end
    columns.extend([
        ('Diff', 'dG'),
        ('Diff', 'Ga')
    ])
    
    # Create MultiIndex columns
    column_index = pd.MultiIndex.from_tuples(
        columns,
        names=['Source', 'Value']
    )
    
    # Prepare comparison data
    comparison_data = []
    for idx, base_row in base_df.iterrows():
        row_data = {
            ('Info', 'step_idx'): base_row['step_idx'],
            ('Info', 'rxn_idx'): base_row['rxn_idx'],
            ('Info', 'Reaction'): base_row['Reaction'],
            (base_name, 'dG'): base_row['dG'],
            (base_name, 'Ga'): base_row['Ga']
        }
        
        # Compare with other mechanisms
        for name, comp_df in all_results.items():
            comp_row = comp_df[comp_df['rxn_idx'] == base_row['rxn_idx']].iloc[0]
            
            row_data[(name, 'dG')] = comp_row['dG']
            row_data[(name, 'Ga')] = comp_row['Ga']
            
            # Store differences under 'Diff' source
            if name == Path(input_paths[1]).stem:  # Only for the first comparison file
                row_data[('Diff', 'dG')] = comp_row['dG'] - base_row['dG']
                row_data[('Diff', 'Ga')] = comp_row['Ga'] - base_row['Ga']
        
        comparison_data.append(row_data)
    
    # Create DataFrame with MultiIndex columns
    comparison_df = pd.DataFrame(comparison_data, columns=column_index)
    
    # Round numeric columns
    numeric_cols = comparison_df.select_dtypes(include=['float64']).columns
    comparison_df[numeric_cols] = comparison_df[numeric_cols].round(2)
    
    # Add metadata for styling
    comparison_df.attrs['threshold'] = threshold
    comparison_df.attrs['base_name'] = base_name
    
    return comparison_df

def create_input_from_gibbs(
    extended_df_dl: pd.DataFrame,
    extended_df_gas: pd.DataFrame,
    dl_temperature: float,
    gas_temperature: float,
    output_path: str = None
) -> pd.DataFrame:
    """
    Create input format DataFrame using Gibbs free energies from two temperature corrections.

    For adsorbed species (i.e. rows whose site_name does not include 'gas'),
    Gibbs energies are taken from extended_df_dl computed with the dl_temperature
    (cathode temperature). For gas-phase species (i.e. rows whose site_name includes 'gas'),
    Gibbs energies are calculated using gas_temperature (electrolyte temperature).
    In addition, gas-phase rows are duplicated so that one copy retains the original site_name
    ('gas') while the duplicate has its site_name replaced with 'dl'.

    Parameters
    ----------
    extended_df_dl : pd.DataFrame
        Extended DataFrame from create_gibbs_table() computed with dl_temperature.
    extended_df_gas : pd.DataFrame
        Extended DataFrame from create_gibbs_table() computed with gas_temperature.
    dl_temperature : float
        Temperature used for dl calculations (cathode temperature).
    gas_temperature : float
        Temperature used for gas calculations (electrolyte temperature).
    output_path : str, optional
        Path to save the output file. If None, file is not saved.
        
    Returns
    -------
    pd.DataFrame
        Input format DataFrame with duplicated gas-phase rows.
    """
    # Process adsorbed species from extended_df_dl (i.e. not gas-phase)
    df_ads = extended_df_dl[~extended_df_dl['site_name'].str.contains('gas', case=False, na=False)].copy()
    df_gas_dl = extended_df_dl[extended_df_dl['site_name'].str.contains('gas', case=False, na=False)].copy()
    df_gas_gas = extended_df_gas[extended_df_gas['site_name'].str.contains('gas', case=False, na=False)].copy()


    # 가스 phase 항목들은 surface_name을 'None'으로 설정 (출력 예와 맞추기 위함).
    df_gas_gas.loc[:, 'surface_name'] = 'None'
    df_gas_dl.loc[:, 'surface_name'] = 'Cu'


    # Duplicate the gas-phase rows:
    #  - one copy retains the original site_name ("gas")
    #  - the duplicate changes site_name to "dl"
    
    df_gas_dl.loc[:, 'site_name'] = 'dl'

    # Combine adsorbed species and both sets of gas-phase rows
    combined_df = pd.concat([df_gas_gas, df_gas_dl, df_ads], ignore_index=True)

    # Build the output DataFrame by selecting the necessary columns
    output_df = combined_df[['surface_name', 'site_name', 'species_name', 'frequencies', 'reference']].copy()
    # Use the free energy column (G) as the formation energy
    output_df['formation_energy'] = combined_df['G']

    # If an output file is requested, convert frequencies to a string and save the file
    if output_path:
        output_df['frequencies'] = output_df['frequencies'].apply(lambda x: '[]')
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, sep='\t', index=False, na_rep="None")

    return output_df