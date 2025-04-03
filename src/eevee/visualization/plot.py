import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path
from ..plot_setting import set_scientific_paper_style
import matplotlib.font_manager as fm
from typing import List, Dict, Tuple
from catmap import analyze, ReactionModel
import catmap

# Print catmap version
#print(f"catmap version: {catmap.__version__}")

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
color_dict = {
    0: '#24AAE3',    # light blue
    25: '#047499',   # blue
    50: '#D13A7A',   # pink
    70: '#A6202E',   # red
    90: '#ff0000',   # dark red
    120: '#b22222',  # dark red
    140: '#8b0000'   # dark red
}
marker_dict = {
    0: 'o',
    25: 's',
    50: '^',
    70: 'v',
    90: 'D',
    120: 'X',
    140: 's'
}
elec_color_dict = {
    0: "#24AAE3",    # light blue
    25: "#047499",
    70: "#A6202E"
}

line_styles = [
    (0, (1, 1)),      # 촘촘한 점선
    (0, (3, 1)),      # 일반 점선 
    (0, (5, 1)),      # 듬성듬성한 점선
    (0, (1, 1, 3, 1)), # 점-대시
    (0, (3, 1, 1, 1)), # 대시-점
    (0, (5, 1, 1, 1)), # 긴대시-점
    (0, (3, 1, 3, 1)), # 대시-대시
    '--',             # 기본 점선
    ':',              # 점선
    '-.'              # 대시-점선
]
def create_rate_map_csv(base_dir, elec_temps, cath_temps):
    """
    Create rate_map_data.csv files for each temperature combination from model.rate_map data.
    Convert reaction labels using convert_reaction_label function.
    
    Parameters:
    -----------
    base_dir : str or Path
        Base directory containing the temperature folders
    elec_temps : list
        List of electrolyte temperatures
    cath_temps : list
        List of cathode temperatures
    """
    for elec_temp in elec_temps:
        for cath_temp in cath_temps:
            folder_path = Path(base_dir) / f"ElecTemp_{elec_temp}C" / f"CathodeTemp_{int(cath_temp+273)}K"
            
            if not folder_path.exists():
                print(f"Folder not found: {folder_path}")
                continue
            os.chdir(folder_path)
            try:
                # Load model from CO2R.log file
                model_file = folder_path / "CO2R.log"
                print(model_file)
                if not model_file.exists():
                    print(f"Model file not found: {model_file}")
                    continue
                    
                model = ReactionModel(setup_file=str(model_file))
                
                # Extract data from rate_map
                data = {
                    'Potential_vs_RHE(V)': [],
                    'Temperature(K)': []
                }
                
                # Convert reaction labels and initialize columns
                reaction_labels = []
                for rxn in model.output_labels['rate']:
                    converted_label = convert_reaction_label(rxn)
                    reaction_labels.append(converted_label)
                    data[converted_label] = []
                
                # Fill data
                for point in model.rate_map:
                    potential = point[0][0]
                    temp = point[0][1]
                    rates = [float(r) for r in point[1]]  # Convert mpf to float
                    
                    data['Potential_vs_RHE(V)'].append(potential)
                    data['Temperature(K)'].append(temp)
                    
                    # Add reaction rates
                    for label, rate in zip(reaction_labels, rates):
                        data[label].append(rate)
                
                # Create DataFrame and save to CSV
                df = pd.DataFrame(data)
                
                # Sort by potential
                df = df.sort_values('Potential_vs_RHE(V)')
                
                output_file = folder_path / "rate_map_data.csv"
                df.to_csv(output_file, index=False)
                print(f"Created rate map CSV file: {output_file}")
                
                # Print first few rows to verify
                # print(f"\nFirst few rows of {output_file}:")
                # print(df.head())
                # print("\nColumns:")
                # print(df.columns.tolist())
                
            except Exception as e:
                print(f"Error processing {folder_path}: {e}")
                raise e  # Re-raise the exception to see the full traceback
def extract_potentials(coverage_map):
    """
    coverage_map에서 potential 값들을 추출하는 함수
    
    Parameters:
    coverage_map (list): coverage_map 데이터
    
    Returns:
    list: 추출된 potential 값들의 리스트
    """
    potentials = [item[0][0] for item in coverage_map]
    return potentials
def get_all_rxn_steps(model):
    len_rxn_expressions = len(model.rxn_expressions)
    all_rxn_steps = list(range(1, len_rxn_expressions+1))
    model.rxn_mechanisms['all_rxn_steps'] = all_rxn_steps
    return model

def process_mechanism_data(model, mechanism_name: str,
                        delta_g_list: List[float],
                        ga_list: List[float]):
    """Analyze specific mechanism at given potentials"""

    # 1) 
    rxn_indices = model.rxn_mechanisms[mechanism_name]

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
        rxn_str = model.rxn_expressions[rxn_idx-1]
        
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

def mechanism_to_dataframe(ma, potential: float = 0.0, temperature: float = None) -> pd.DataFrame:
    """
    Convert mechanism analysis data to DataFrame for a given potential and temperature
    """
    # Use instance temperature if not specified
    fig = ma.plot(save=False, plot_variants=[potential])
    plt.close(fig)
    data_dict = ma.data_dict

    all_steps_data = []
    for mechanism, (delta_g_list, ga_list) in data_dict.items():
        if not mechanism == 'all_rxn_steps':
            continue
            
        # Get data for all steps in this mechanism
        steps_data = process_mechanism_data(ma,
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
def interpolate_point(x_points, y_points, x_target, tolerance=1e-10):
    """
    Performs linear interpolation between given data points.
    
    Parameters:
    -----------
    x_points : array-like
        x coordinate values
    y_points : array-like
        y coordinate values (2D array)
    x_target : float
        x value to interpolate at
    tolerance : float
        tolerance for floating point comparisons
    
    Returns:
    --------
    array : interpolated y values array
    """
    # Sort x_points and y_points based on x values
    sorted_indices = np.argsort(x_points)
    x_points = np.array(x_points)[sorted_indices]
    y_points = np.array(y_points)[sorted_indices]
    
    # Check if x_target is within data range
    if x_target < min(x_points) - tolerance or x_target > max(x_points) + tolerance:
        raise ValueError(f"Target x ({x_target}) is outside the data range [{min(x_points)}, {max(x_points)}]")

    # Find interval containing x_target
    for i in range(len(x_points)-1):
        x1, x2 = x_points[i], x_points[i+1]
        if x1 - tolerance <= x_target <= x2 + tolerance:
            # Convert mpf objects to float
            y1 = float(y_points[i])
            y2 = float(y_points[i+1])

            # Calculate slope: a = (y2-y1)/(x2-x1)
            a = (y2 - y1)/(x2 - x1)
            
            # Calculate y-intercept: b = y1 - a*x1
            b = y1 - a*x1
            
            # Calculate interpolated y value: y = ax + b
            y_interpolated = a*x_target + b
            
            return y_interpolated
            
    return None

def find_x_for_target_y(x_vals, y_vals, target, log=False):
    """
    Given arrays of x_vals and y_vals (assumed sorted by x),
    find the x-value corresponding to target y using linear interpolation.
    
    Parameters:
      x_vals: array-like of x-values
      y_vals: array-like of y-values (same length as x_vals)
      target: target y value
      log: if True, perform calculations in logarithmic scale (not used here)
      
    Returns:
      x_target: interpolated x-value where y equals target (or None if not found)
    """
    for i in range(len(x_vals)-1):
        y1 = y_vals[i]
        y2 = y_vals[i+1]
        # Check if target lies between y1 and y2 (or equals one endpoint)
        if (y1 - target) * (y2 - target) <= 0:
            x1 = x_vals[i]
            x2 = x_vals[i+1]
            if x2 != x1:
                m = (y2 - y1) / (x2 - x1)
                if m != 0:
                    x_target = x1 + (target - y1) / m
                    return x_target
    return None


def find_related_steps(product_idx, production_rates, elementary_rates, threshold=0.001):
    """
    각 전압에서 production rate와 유사한 값을 가지는 elementary step들을 찾습니다.
    """
    related_steps = set()
    
    for voltage_idx in range(len(production_rates)):
        prod_rate = np.abs(production_rates[voltage_idx, product_idx])
        if prod_rate > 0:  # production rate가 0이 아닌 경우만 고려
            for step_idx in range(elementary_rates.shape[1]):
                step_rate = np.abs(elementary_rates[voltage_idx, step_idx])
                # 두 rate의 비율이 threshold 범위 내에 있는지 확인
                if step_rate > 0 and 1-threshold <= (step_rate/prod_rate) <= 1+threshold:
                    related_steps.add(step_idx)
    
    return sorted(list(related_steps))

def convert_reaction_label(reaction):
    # 치환 규칙 정의
    replacements = {
        'H_g': 'H+',
        'ele_g': 'e-',
        '_t': '*',
        't': '*',
        'CO2_g': 'CO2',
        'H2O_g': 'H2O',
        'CO_g': 'CO',
        'HCOOH_g': 'HCOOH',
        'CH4_g': 'CH4',
        'O2_g': 'O2',
        'C2H4_g': 'C2H4',
        'H2_g': 'H2'
    }
    
    def simplify_species(species):
        """반복되는 화학종을 계수로 표현"""
        species_count = {}
        for s in species:
            if s in species_count:
                species_count[s] += 1
            else:
                species_count[s] = 1
        
        # 계수가 1인 경우는 계수를 표시하지 않음
        return ' + '.join(f'{count if count > 1 else ""}{species}' 
                        for species, count in species_count.items())
    
    def convert_species(species_list):
        # 화학종을 개별 항목으로 분리
        unique_species = []
        repeated_species = {'H+': 0, 'e-': 0}  # 반복될 수 있는 화학종 카운트
        
        for species in species_list:
            temp = species
            # 먼저 치환 규칙 적용
            for old, new in replacements.items():
                temp = temp.replace(old, new)
            
            # H+와 e-는 카운트
            if temp in repeated_species:
                repeated_species[temp] += 1
            else:
                unique_species.append(temp)
        
        # 반복되는 화학종 추가 (카운트가 0이 아닌 경우만)
        final_species = []
        for species, count in repeated_species.items():
            if count > 0:
                final_species.append(f"{count if count > 1 else ''}{species}")
        
        # 나머지 화학종 추가
        final_species.extend(unique_species)
        
        return ' + '.join(final_species)
    
    # IS와 FS만 변환 (TS 제외)
    is_species = convert_species(reaction[0])
    fs_species = convert_species(reaction[-1])
    
    return f"{is_species} $ \\rightarrow $ {fs_species}"


def plot_total_current_vs_potential(base_dir, target_current, elec_temps, cath_temps, ymax_current=1000):
    """
    For each electrolyte temperature (three subplots), plot total current and experimental data.
    
    Parameters:
      base_dir: Base directory where temperature-specific folders are stored.
      target_current: Target current value for interpolation.
    """
    # 실험 데이터 정의
    data_0C = {
        'Temperature': [0, 25, 50, 70, 90, 120, 140],
        'Average': [-1.006104077, -1.014636683, -1.04129524, -0.956481336, -1.036759643, -0.9724704, -0.917120021],
        'STD': [0.016578511, 0.022287844, 0.022210104, 0.069947976, 0.027417144, 0.041083994, 0.037165591]
    }

    data_25C = {
        'Temperature': [0, 25, 50, 70, 90, 120, 140],
        'Average': [-1.073473989, -1.067790195, -1.058116694, -0.973472696, -1.033715649, -1.009302003, -0.987633384],
        'STD': [0.019420577, 0.028049232, 0.021847092, 0.027631222, 0.024637091, 0.031442624, 0.02076341]
    }

    data_70C = {
        'Temperature': [0, 25, 50, 70, 90, 120, 140],
        'Average': [-1.055392753, -1.050983247, -1.056895426, -1.016039275, -1.022955809, -1.017724385, -0.959401295],
        'STD': [0.02824217, 0.033508477, 0.04037112, 0.05381293, 0.039480078, 0.048606313, 0.048606035]
    }

    
    color_dict = {
        0: '#00b7eb',    # light blue
        25: '#1f77b4',   # blue
        50: '#ff7f7f',   # pink
        70: '#ff4444',   # red
        90: '#ff0000',   # dark red
        120: '#b22222',  # dark red
        140: '#8b0000'   # dark red
    }
    marker_dict = {
        0: 'o',
        25: 's',
        50: '^',
        70: 'v',
        90: 'D',
        120: 'X',
        140: 's'
    }
    elec_color_dict = {
        0: "#24AAE3",    # light blue
        25: "#047499",
        70: "#A6202E"
    }
    products = ['H$_2$(g)', 'CO(g)', 'CH$_4$(g)', 'C$_2$H$_4$(g)', 'HCOOH(g)']
    target_potentials = {}
    
    # 그래프 설정
    fig = plt.figure(figsize=(15/1.5, 10/1.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1])
    axes_top = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axes_bottom = [fig.add_subplot(gs[1, i]) for i in range(3)]
    
    intersection_data = []
    
    for ax_top, ax_bottom, elec_temp in zip(axes_top, axes_bottom, elec_temps):
        # 상단 플롯 (기존 코드와 동일)
        ax_top.axhline(y=target_current, color='gray', linestyle='--', alpha=0.5)
        elec_intersections = []
        
        for cath_temp in cath_temps:
            cath_temp_K = cath_temp + 273
            folder_path = Path(base_dir) / f"ElecTemp_{elec_temp}C" / f"CathodeTemp_{int(cath_temp_K)}K"
            csv_file = folder_path / "current_density_data.csv"
            if not csv_file.exists():
                print(f"File not found: {csv_file}")
                continue
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                continue
            
            if "Potential_vs_RHE(V)" not in df.columns:
                print(f"'Potential_vs_RHE(V)' column not found in {csv_file}")
                continue
            
            potentials_arr = df["Potential_vs_RHE(V)"].values
            total_current = np.zeros_like(potentials_arr, dtype=float)
            for prod in products:
                if prod in df.columns:
                    total_current += df[prod].values
                else:
                    print(f"Column '{prod}' not found in {csv_file}. Using 0 for this product.")
            
            ax_top.plot(potentials_arr, total_current,
                    marker=marker_dict[cath_temp], linestyle='-',
                    color=color_dict[cath_temp],
                    label=f'{cath_temp}°C', markersize=6)
            
            interp_pot = find_x_for_target_y(potentials_arr, total_current, target_current, log=False)
            if interp_pot is not None:
                target_potentials[(elec_temp, cath_temp)] = interp_pot
                ax_top.plot(interp_pot, target_current, 'o', markersize=3, zorder=5, 
                          markerfacecolor='red', markeredgecolor='black')
                
                elec_intersections.append({
                    'Electrolyte_Temp': elec_temp,
                    'Cathode_Temp': cath_temp,
                    'Potential': interp_pot
                })
            else:
                print(f"Target current {target_current} not reached for Elec {elec_temp}°C, Cath {cath_temp}°C")
        
        # Configure top plot
        ax_top.set_xlabel("Potential vs RHE (V)", fontweight=900)
        ax_top.set_title(f"Electrolyte {elec_temp}°C", fontweight=900)
        #ax_top.xaxis.set_minor_locator(AutoMinorLocator(0))
        ax_top.tick_params(which='both', direction='in')
        ax_top.set_ylim(0, ymax_current)
        
        # 하단 플롯 설정
        if elec_intersections:
            df_intersect = pd.DataFrame(elec_intersections)
            intersection_data.append(df_intersect)
            
            # 실험 데이터 선택
            exp_data = {
                0: data_0C,
                25: data_25C,
                70: data_70C
            }[elec_temp]
            
            df_exp = pd.DataFrame(exp_data)
            
            # 실험값 플롯
            ax_bottom.errorbar(df_exp['Temperature'], df_exp['Average'], 
                             yerr=df_exp['STD'],
                             fmt='o-', capsize=5, capthick=1, 
                             elinewidth=1, markersize=6,
                             color=elec_color_dict[elec_temp], 
                             label='Experiment',
                             zorder=2)
            
            # 시뮬레이션 결과 플롯
            ax_bottom.scatter(df_intersect['Cathode_Temp'], 
                            df_intersect['Potential'],
                            marker='s', s=60,
                            color=elec_color_dict[elec_temp],
                            alpha=0.7, label='Simulation',
                            zorder=3)
            
            # 값 라벨 추가
            for x, y in zip(df_intersect['Cathode_Temp'], df_intersect['Potential']):
                ax_bottom.text(x, y, f'{y:.2f}',
                             ha='center', va='bottom',
                             fontsize=8, color='black')
            
            # 축 설정
            ax_bottom.set_xlabel('Cathode Temperature (°C)', fontweight=900)
            ax_bottom.set_ylabel('Potential (V)', fontweight=900)
            ax_bottom.set_title(f'Electrolyte {elec_temp}°C', fontweight=900)
            ax_bottom.set_xticks([0, 25, 50, 70, 90, 120, 140], minor=False)
            ax_bottom.set_ylim(-1.2, -0.7)
            ax_bottom.tick_params(direction='in', which='both')
            ax_bottom.set_yticks(np.arange(-1.2, -0.69, 0.1))
            ax_bottom.set_yticks(np.arange(-1.2, -0.69, 0.05), minor=True)
            
            ax_bottom.tick_params(axis='y', which='minor', length=2)
            ax_bottom.tick_params(axis='y', which='major', length=4)
            # 메이저 틱 사이에 하나의 마이너 틱 추가
            minor_ticks = []
            major_ticks = [0, 25, 50, 70, 90, 120, 140]
            for i in range(len(major_ticks)-1):
                minor_ticks.append((major_ticks[i] + major_ticks[i+1]) / 2)  # 중간값 계산

            ax_bottom.set_xticks(minor_ticks, minor=True)  # 마이너 틱 설정
            ax_bottom.tick_params(axis='x', which='minor', length=2)
            ax_bottom.tick_params(axis='x', which='major', length=4)
            #ax_bottom.grid(True, alpha=0.3)
            ax_bottom.legend(fontsize=8)
    
    # 공통 레이블 설정
    axes_top[0].set_ylabel("Total Current (A/cm²)", fontweight=900)
    axes_top[2].legend(loc='upper right')
    # 각 그래프에 스타일 적용
    for ax in axes_top + axes_bottom:
        set_scientific_paper_style(ax)

    plt.tight_layout()
    
    # 저장
    output_file = Path(base_dir) / "total_current_vs_potential.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    if intersection_data:
        pd.concat(intersection_data).to_csv(
            Path(base_dir)/'current_intersections.csv',
            index=False,
            float_format='%.2f'
        )

    
    return target_potentials


def plot_faradaic_efficiency_at_target(base_dir, target_potentials, elec_temps, cath_temps):
    """
    interpolate the faradaic efficiency
    For each electrolyte temperature and each cathode temperature, interpolate the faradaic efficiency
    at the target potential and produce a stacked bar plot.
    """
    # Define products and colors for the bar plot
    products = {
        'H$_2$(g)': '#5CB2E4',       # Blue
        'CO(g)': '#4081A2',          # Teal
        'CH$_4$(g)': '#CA5281',      # Pink
        'C$_2$H$_4$(g)': '#A74245',  # Red-orange
        'HCOOH(g)': '#468451',       # Green
    }
    bar_width = 10
    x_ticks = cath_temps

    # Create one subplot per electrolyte temperature
    fig, axes = plt.subplots(1, len(elec_temps), figsize=(15/1.5, 5/1.5), sharey=True)
    
    for ax, elec_temp in zip(axes, elec_temps):
        cathode_data = []
        for cath_temp in cath_temps:
            key = (elec_temp, cath_temp)
            data_point = {"cathode_temp": cath_temp}
            
            if key not in target_potentials:
                print(f"No target potential available for Elec {elec_temp}°C, Cath {cath_temp}°C; skipping.")
                for prod in products.keys():
                    data_point[prod] = 0
                cathode_data.append(data_point)
                continue
                
            target_pot = target_potentials[key]
            
            # Load faradaic efficiency data
            cath_dir = Path(base_dir) / f"ElecTemp_{elec_temp}C" / f"CathodeTemp_{int(cath_temp+273)}K"
            csv_file = cath_dir / "faradaic_efficiency_data.csv"
            
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file)
                    if "Potential_vs_RHE(V)" not in df.columns:
                        raise ValueError(f"'Potential_vs_RHE(V)' not in {csv_file}")
                        
                    # Get potential array and FE values for interpolation
                    potentials = df["Potential_vs_RHE(V)"].values
                    
                    # Interpolate FE for each product
                    for prod in products.keys():
                        if prod in df.columns:
                            fe_values = df[prod].values
                            # Interpolate and convert to percentage
                            fe_interpolated = interpolate_point(potentials, fe_values, target_pot)
                            data_point[prod] = float(fe_interpolated) * 100 if fe_interpolated is not None else 0
                        else:
                            data_point[prod] = 0
                            
                except Exception as e:
                    print(f"Error processing {csv_file}: {e}")
                    for prod in products.keys():
                        data_point[prod] = 0
                        
            else:
                print(f"Faradaic efficiency file not found: {csv_file}")
                for prod in products.keys():
                    data_point[prod] = 0
                    
            cathode_data.append(data_point)
        
        # Create stacked bar plot
        temp_df = pd.DataFrame(cathode_data).sort_values('cathode_temp')
        bottom = np.zeros(len(temp_df))
        
        # 바 플롯 생성 부분 수정
        for prod, color in products.items():
            values = temp_df[prod].values if prod in temp_df.columns else np.zeros(len(temp_df))
            prod_name = prod.replace('(g)', '')
            ax.bar(temp_df["cathode_temp"], values, bottom=bottom,
                label=prod_name if elec_temp == elec_temps[-1] else "",
                color=color, width=bar_width, 
                edgecolor='black', linewidth=0.5)  # 검정색 테두리 추가
            bottom += values

        # 축 설정 부분 수정
        ax.set_xlabel("Cathode Temperature (°C)")
        ax.set_title(f"Electrolyte {elec_temp}°C")
        ax.set_ylim(0, 102)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks)

        # y축 눈금 설정
        ax.set_yticks(np.arange(0, 101, 20))  # 20마다 major tick
        ax.set_yticks(np.arange(0, 101, 10), minor=True)  # 10마다 minor tick
        ax.tick_params(axis='y', which='minor', length=2)  # minor tick 길이 설정
        ax.tick_params(axis='y', which='major', length=4)  # major tick 길이 설정
        minor_ticks = []
        major_ticks = [0, 25, 50, 70, 90, 120, 140]
        for i in range(len(major_ticks)-1):
            minor_ticks.append((major_ticks[i] + major_ticks[i+1]) / 2)  # 중간값 계산

        ax.set_xticks(minor_ticks, minor=True)  # 마이너 틱 설정
        ax.tick_params(axis='x', which='minor', length=2)
        ax.tick_params(axis='x', which='major', length=4)
        ax.grid(True, alpha=0.3)
        set_scientific_paper_style(ax)
            
    axes[0].set_ylabel("Faradaic Efficiency (%)")
    #fig.suptitle("Faradaic Efficiency at fixed current", y=1.05)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8
    # 모든 축의 legend 제거 (마지막에 하나만 추가할 것이므로)
    for ax in axes:
        ax.get_legend().remove() if ax.get_legend() else None

    # 그래프 위에 하나의 row로 legend 추가
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, 
            loc='upper center', 
            bbox_to_anchor=(0.5, 1.05),
            ncol=len(products),  # 모든 항목을 한 줄에 표시
            fontsize=10)   

    plt.tight_layout()
    output_file = Path(base_dir) / "faradaic_efficiency_at_target.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

def plot_coverage_vs_temperature(base_dir, target_potentials, elec_temps, cath_temps, threshold=0.005):
    """
    Plot surface coverage for each temperature combination at target potentials.
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing calculation results
    target_potentials : dict
        Dictionary mapping (elec_temp, cath_temp) to target potential
    elec_temps : list
        List of electrolyte temperatures (°C)
    cath_temps : list
        List of cathode temperatures (°C)
    threshold : float
        Minimum coverage value to include in the plot (default: 0.005)
    """
    # Create figure
    plt.figure(figsize=(3.5*1.2, 2.8*1.2))
    
    # Dictionary to store coverage data for all species
    all_coverage_data = {}
    
    # First pass: collect all species with coverage above threshold
    significant_species = set()
    
    for elec_temp in elec_temps:
        for cath_temp in cath_temps:
            if (elec_temp, cath_temp) not in target_potentials:
                continue
                
            target_pot = target_potentials[(elec_temp, cath_temp)]
            folder_path = Path(base_dir) / f"ElecTemp_{elec_temp}C" / f"CathodeTemp_{int(cath_temp+273)}K"
            coverage_file = folder_path / "coverage_map_data.csv"
            
            if coverage_file.exists():
                try:
                    cov_df = pd.read_csv(coverage_file)
                    potentials = cov_df['Potential_vs_RHE(V)'].values
                    
                    # Get all species columns
                    species_columns = [col for col in cov_df.columns if col != 'Potential_vs_RHE(V)']
                    
                    # Interpolate coverage at target potential
                    for species in species_columns:
                        coverage = cov_df[species].values
                        interpolated_coverage = interpolate_point(potentials, coverage, target_pot)
                        
                        if interpolated_coverage is not None and interpolated_coverage > threshold:
                            significant_species.add(species)
                            
                except Exception as e:
                    print(f"Error reading {coverage_file}: {e}")
                    continue
    
    print(f"Species with significant coverage (>{threshold}): {sorted(significant_species)}")
    
    # Create a colormap for species
    color_dict = {
    0: '#24AAE3',    # light blue
    25: '#047499',   # blue
    50: '#D13A7A',   # pink
    70: '#A6202E',   # red
    90: '#ff0000',   # dark red
    120: '#b22222',  # dark red
    140: '#8b0000'   # dark red
    #'#CA5281' pink
    }
    
    species_colors = plt.cm.tab20(np.linspace(0, 1, len(significant_species)))
    species_color_dict = {species: color for species, color in zip(sorted(significant_species), species_colors)}
    species_color_dict['CH*'] = '#CF3030'
    species_color_dict['CO*'] = '#F78536'
    species_color_dict['COH*'] = '#009975'
    species_color_dict['H*'] = '#5CB2E4'
    species_color_dict['OCC*'] = '#22559C'
    #species_color_dict['*'] = color_dict[140]
    # Create a list for legend entries
    legend_entries = []
    
    # Second pass: collect and plot data for significant species
    for species_idx, species in enumerate(sorted(significant_species)):
        # Get color for this species
        species_color = species_color_dict[species]
        
        # Add species to legend only once
        legend_entries.append(plt.Line2D([0], [0], color=species_color, lw=2, label=species))
        
        for i, elec_temp in enumerate(elec_temps):
            coverages = []
            cath_temps_available = []
            
            for cath_temp in cath_temps:
                if (elec_temp, cath_temp) not in target_potentials:
                    continue
                    
                target_pot = target_potentials[(elec_temp, cath_temp)]
                folder_path = Path(base_dir) / f"ElecTemp_{elec_temp}C" / f"CathodeTemp_{int(cath_temp+273)}K"
                coverage_file = folder_path / "coverage_map_data.csv"
                
                if coverage_file.exists():
                    try:
                        cov_df = pd.read_csv(coverage_file)
                        potentials = cov_df['Potential_vs_RHE(V)'].values
                        
                        if species in cov_df.columns:
                            coverage = cov_df[species].values
                            interpolated_coverage = interpolate_point(potentials, coverage, target_pot)
                            
                            if interpolated_coverage is not None:
                                coverages.append(float(interpolated_coverage))
                                cath_temps_available.append(cath_temp)
                                
                    except Exception as e:
                        print(f"Error processing {coverage_file} for {species}: {e}")
                        continue
            
            if coverages:
                # Calculate alpha based on electrolyte temperature
                # Higher electrolyte temperature = higher alpha (more opaque)
                alpha = 0.3 + 0.7 * (i / (len(elec_temps) - 1)) if len(elec_temps) > 1 else 1.0
                
                # Plot coverage vs cathode temperature
                plt.plot(cath_temps_available, coverages, '-o', 
                        color=species_color,
                        alpha=alpha,
                        linewidth=2,
                        markersize=6)
                
                # # Add small text annotation for electrolyte temperature
                # for x, y in zip(cath_temps_available, coverages):
                #     plt.annotate(f'{elec_temp}°C', 
                #                 xy=(x, y), 
                #                 xytext=(2, 2),
                #                 textcoords='offset points',
                #                 fontsize=6,
                #                 alpha=alpha)
                
                # Store data for table
                if species not in all_coverage_data:
                    all_coverage_data[species] = {}
                all_coverage_data[species][elec_temp] = {temp: cov for temp, cov in zip(cath_temps_available, coverages)}
    
    # Add electrolyte temperature explanation
    elec_temp_legend = []
    for i, temp in enumerate(elec_temps):
        alpha = 0.3 + 0.7 * (i / (len(elec_temps) - 1)) if len(elec_temps) > 1 else 1.0
        elec_temp_legend.append(plt.Line2D([0], [0], color='gray', alpha=alpha, lw=2, label=f'Elec {temp}°C'))
    
    # Customize plot
    #plt.xlabel('Cathode Temperature (°C)', font)
    plt.ylabel('Coverage', fontproperties=helvetica_bold_prop)
    #plt.title('Coverage vs. Temperature at Target Potential')
    #plt.grid(True, alpha=0.3)
    # 축 설정 부분 수정
    plt.xlabel("Cathode Temperature (°C)",  fontproperties=helvetica_bold_prop)
    plt.ylim(0,0.41)
    plt.xlim(0, 143)
    # plt.set_xticks(x_ticks)
    # plt.set_xticklabels(x_ticks)

    # y축 눈금 설정
    plt.yticks(np.arange(0, 0.41, 0.1))  # 20마다 major tick
    plt.yticks(np.arange(0, 0.41, 0.05), minor=True)  # 10마다 minor tick
    plt.tick_params(axis='y', which='minor', length=2)  # minor tick 길이 설정
    plt.tick_params(axis='y', which='major', length=4)  # major tick 길이 설정
    #minor_ticks = []
    #major_ticks = [0, 25, 50, 70, 90, 120, 140]
    # for i in range(len(major_ticks)-1):
    #     minor_ticks.append((major_ticks[i] + major_ticks[i+1]) / 2)  # 중간값 계산
    plt.xticks(np.arange(0, 141, 20))
    plt.xticks(np.arange(0, 141, 10), minor=True)  # 마이너 틱 설정
    plt.tick_params(axis='x', which='minor', length=2)
    plt.tick_params(axis='x', which='major', length=4)
    #plt.grid(True, alpha=0.3)
    #set_scientific_paper_style(plt)    
    # Create two-part legend
    #first_legend = plt.legend(handles=legend_entries, loc='upper left', bbox_to_anchor=(1.05, 1))
    first_legend = plt.legend(handles=legend_entries, loc='lower right')
    plt.gca().add_artist(first_legend)
    #plt.legend(handles=elec_temp_legend, title="Electrolyte Temp", loc='upper left', bbox_to_anchor=(1.05, 0.5))
    plt.legend(handles=elec_temp_legend, loc='upper left')
    plt.tight_layout()
    
    # Save figure
    output_file = Path(base_dir) / "coverage_vs_temperature.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a DataFrame for the coverage data
    data_rows = []
    for species in sorted(all_coverage_data.keys()):
        for elec_temp in elec_temps:
            if elec_temp in all_coverage_data[species]:
                for cath_temp, coverage in all_coverage_data[species][elec_temp].items():
                    data_rows.append({
                        'Species': species,
                        'Electrolyte_Temperature(C)': elec_temp,
                        'Cathode_Temperature(C)': cath_temp,
                        'Coverage': coverage,
                        'Potential_vs_RHE(V)': target_potentials[(elec_temp, cath_temp)]
                    })
    
    coverage_df = pd.DataFrame(data_rows)
    
    # Save to CSV
    csv_file = Path(base_dir) / "coverage_vs_temperature_data.csv"
    coverage_df.to_csv(csv_file, index=False)
    print(f"Coverage data saved to {csv_file}")
    
    return coverage_df


def plot_rates_vs_temperature(base_dir, target_potentials, elec_temps, cath_temps):
    """
    Plot production rates and related elementary steps for each temperature combination.
    Saves raw data as CSV files.
    """
    # Define line styles for related steps
    line_styles = [
        (0, (1, 1)),      # 촘촘한 점선
        (0, (3, 1)),      # 일반 점선 
        (0, (5, 1)),      # 듬성듬성한 점선
        (0, (1, 1, 3, 1)), # 점-대시
        (0, (3, 1, 1, 1)), # 대시-점
        (0, (5, 1, 1, 1)), # 긴대시-점
        (0, (3, 1, 3, 1)), # 대시-대시
        '--',             # 기본 점선
        ':',              # 점선
        '-.'              # 대시-점선
    ]

    # 전해질 온도별 색상 정의
    elec_color_dict = {
        0: "#24AAE3",    # light blue
        25: "#047499",
        70: "#A6202E"
    }

    # 표시할 제품 목록 지정
    selected_products = ['H$_2$(g)', 'CO(g)', 'CH$_4$(g)', 'C$_2$H$_4$(g)']
    exclude_products = ['CO$_2$(g)', 'H$_2$O(g)', 'H(g)', 'O$_2$(g)', 'ele(g)', 'HCOOH(g)']  # HCOOH 제외
    
    # Dictionary to store data for CSV export
    csv_data = {
        'Electrolyte_Temperature(C)': [],
        'Cathode_Temperature(C)': [],
        'Product': [],
        'Production_Rate': [],
        'Elementary_Step': [],
        'Elementary_Step_Rate': [],
        'Potential_vs_RHE(V)': []
    }
    
    # 제품 데이터 확인
    products = None
    for elec_temp in elec_temps:
        for cath_temp in cath_temps:
            prod_file = Path(base_dir) / f"ElecTemp_{elec_temp}C" / f"CathodeTemp_{int(cath_temp+273)}K" / "production_rate_map_data.csv"
            if prod_file.exists():
                df = pd.read_csv(prod_file)
                all_products = [col for col in df.columns if col not in ['Potential_vs_RHE(V)'] + exclude_products]
                # 선택된 제품만 필터링
                products = [p for p in all_products if p in selected_products]
                break
        if products is not None:
            print(f"Products to analyze: {products}")
            break
    
    if not products:
        print("No products found or none of the selected products are available.")
        return
            
    # Collect related steps for each product across all temperatures
    product_related_steps = {product: set() for product in products}
    
    # First pass: 각 제품별 모든 related steps 수집
    for product in products:
        for elec_temp in elec_temps:
            for cath_temp in cath_temps:
                if (elec_temp, cath_temp) not in target_potentials:
                    continue
                    
                folder_path = Path(base_dir) / f"ElecTemp_{elec_temp}C" / f"CathodeTemp_{int(cath_temp+273)}K"
                prod_file = folder_path / "production_rate_map_data.csv"
                rate_file = folder_path / "rate_map_data.csv"
                
                if prod_file.exists() and rate_file.exists():
                    try:
                        prod_df = pd.read_csv(prod_file)
                        rate_df = pd.read_csv(rate_file)
                        
                        if product not in prod_df.columns:
                            continue
                            
                        production_rates = np.zeros((len(prod_df), len(products)))
                        for p_idx, p in enumerate(products):
                            if p in prod_df.columns:
                                production_rates[:, p_idx] = prod_df[p].values
                        
                        elementary_steps = [col for col in rate_df.columns 
                                         if col not in ['Potential_vs_RHE(V)', 'Temperature(K)']]
                        elementary_rates = np.zeros((len(rate_df), len(elementary_steps)))
                        for e_idx, step in enumerate(elementary_steps):
                            elementary_rates[:, e_idx] = rate_df[step].values
                        
                        try:
                            related_indices = find_related_steps(products.index(product), 
                                                              production_rates, 
                                                              elementary_rates,
                                                              threshold=0.00001)
                            
                            for idx in related_indices:
                                product_related_steps[product].add(elementary_steps[idx])
                        except Exception as e:
                            print(f"Error finding related steps for {product}: {e}")
                            continue
                            
                    except Exception as e:
                        print(f"Error in first pass for {folder_path}: {e}")
                        continue

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(12/1.3, 20/1.2))
    axes = axes.flatten()  # 2D 배열을 1D로 변환
    
    for prod_idx, product in enumerate(products):
        if prod_idx >= len(axes):  # 안전 장치
            break
            
        ax = axes[prod_idx]
        
        # 각 플롯마다 별도의 legend 항목을 위한 핸들과 라벨 리스트
        elec_handles = []
        elec_labels = []
        step_handles = []
        step_labels = []
        
        for elec_temp in elec_temps:
            rates = []
            related_steps_rates = {}
            cath_temps_available = []
            
            # 전해질 온도에 해당하는 색상 가져오기
            elec_color = elec_color_dict.get(elec_temp, '#333333')  # 기본값은 회색
            
            for cath_temp in cath_temps:
                if (elec_temp, cath_temp) not in target_potentials:
                    continue
                    
                target_pot = target_potentials[(elec_temp, cath_temp)]
                folder_path = Path(base_dir) / f"ElecTemp_{elec_temp}C" / f"CathodeTemp_{int(cath_temp+273)}K"
                prod_file = folder_path / "production_rate_map_data.csv"
                rate_file = folder_path / "rate_map_data.csv"
                
                if prod_file.exists() and rate_file.exists():
                    try:
                        prod_df = pd.read_csv(prod_file)
                        rate_df = pd.read_csv(rate_file)
                        
                        potentials = prod_df['Potential_vs_RHE(V)'].values
                        
                        if product in prod_df.columns:
                            prod_rates = prod_df[product].values
                            interpolated_rate = interpolate_point(potentials, prod_rates, target_pot)
                            
                            if interpolated_rate is not None:
                                rates.append(float(interpolated_rate))
                                cath_temps_available.append(cath_temp)
                                
                                # Store product rate data for CSV
                                csv_data['Electrolyte_Temperature(C)'].append(elec_temp)
                                csv_data['Cathode_Temperature(C)'].append(cath_temp)
                                csv_data['Product'].append(product)
                                csv_data['Production_Rate'].append(float(interpolated_rate))
                                csv_data['Elementary_Step'].append(None)
                                csv_data['Elementary_Step_Rate'].append(None)
                                csv_data['Potential_vs_RHE(V)'].append(target_pot)
                                
                                for step in product_related_steps[product]:
                                    if step in rate_df.columns:
                                        step_rates = rate_df[step].values
                                        step_rate = interpolate_point(potentials, step_rates, target_pot)
                                        if step_rate is not None:
                                            if step not in related_steps_rates:
                                                related_steps_rates[step] = []
                                            related_steps_rates[step].append(float(step_rate))
                                            
                                            # Store elementary step rate data for CSV
                                            csv_data['Electrolyte_Temperature(C)'].append(elec_temp)
                                            csv_data['Cathode_Temperature(C)'].append(cath_temp)
                                            csv_data['Product'].append(product)
                                            csv_data['Production_Rate'].append(None)
                                            csv_data['Elementary_Step'].append(step)
                                            csv_data['Elementary_Step_Rate'].append(float(step_rate))
                                            csv_data['Potential_vs_RHE(V)'].append(target_pot)
                                        
                    except Exception as e:
                        print(f"Error in second pass for {folder_path}: {e}")
                        continue
            
            if rates:
                # Plot production rate
                line, = ax.semilogy(cath_temps_available, rates, '-o', 
                          color=elec_color, 
                          linewidth=4, alpha=0.3)
                
                # 전해질 온도 legend에 추가
                elec_handles.append(line)
                elec_labels.append(f'Elec {elec_temp}°C')
                
                # Plot related steps
                for step_idx, (step, step_rates) in enumerate(related_steps_rates.items()):
                    if len(step_rates) == len(cath_temps_available):
                        line_style = line_styles[step_idx % len(line_styles)]  # 순환적으로 스타일 선택
                        step_line, = ax.semilogy(cath_temps_available, step_rates,
                                  linestyle=line_style,
                                  color=elec_color,
                                  alpha=1.0,
                                  linewidth=1.0)
                        
                        # 첫 번째 전해질 온도에서만 step legend 추가 (중복 방지)
                        if elec_temp == elec_temps[0]:
                            step_handles.append(step_line)
                            step_labels.append(step)
        
        # 제품 이름에서 (g) 제거하여 타이틀 설정
        clean_product_name = product.replace('(g)', '')
        ax.set_title(f'{clean_product_name}')
        #ax.grid(True, alpha=0.3)
        
        # y축 레이블 설정
        ax.set_ylabel(f'Rate (s$^{{-1}}$)')
        
        # x축 레이블은 아래쪽 서브플롯에만 표시
        #if prod_idx >= 2:  # 아래쪽 서브플롯
        ax.set_xlabel('Cathode Temperature (°C)')
        
        # 각 서브플롯 아래에 legend 배치
        # 전해질 온도 legend
        elec_legend = ax.legend(elec_handles, elec_labels, 
                              #title="Electrolyte Temperature",
                              loc='lower center', 
                              bbox_to_anchor=(0.5, 1.05),
                              ncol=len(elec_temps),
                              fontsize=8)
        ax.add_artist(elec_legend)
        
        # 반응 단계 legend는 너무 많을 수 있으므로 생략하거나 필요시 추가
        # 필요하다면 아래 주석을 해제하여 사용
        
        if step_handles:
            step_legend = ax.legend(step_handles, step_labels, 
                                  #title="Elementary Steps",
                                  bbox_to_anchor=(0.5, -0.35), 
                                  loc='lower center', 
                                  ncol=min(2, len(step_handles)),
                                  fontsize=7)
            ax.add_artist(step_legend)
        
    
    plt.tight_layout(h_pad=10, w_pad=2.0)  # 서브플롯 간 간격 조정
    #plt.tight_layout()
    plt.subplots_adjust(bottom=0.4, top=0.8, hspace=0.5)   # legend를 위한 하단 여백 추가
    #plt.subplots_adjust(hspace=0.4 )   # legend를 위한 하단 여백 추가
    output_file = Path(base_dir) / "rates_vs_temperature_2x2.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.show()
    
    # Save data to CSV
    csv_df = pd.DataFrame(csv_data)
    csv_output_file = Path(base_dir) / "rates_vs_temperature_data.csv"
    csv_df.to_csv(csv_output_file, index=False, float_format='%.6e')
    print(f"Raw data saved to {csv_output_file}")

def analyze_temperature_data(base_dir: str, elec_temps: List[int], cath_temps: List[int]):
    """
    Generate and save coverage map and mechanism data for each temperature condition
    
    Parameters:
    -----------
    base_dir : str
        Base directory path
    elec_temps : List[int]
        List of electrolyte temperatures (e.g. [0, 25, 70])
    cath_temps : List[int]
        List of cathode temperatures (e.g. [273, 298, 323, 343])
    """
    for elec_temp in elec_temps:
        for temp_k in cath_temps:
            try:
                # Create current working directory path
                cath_dir = Path(base_dir) / f"ElecTemp_{elec_temp}C/CathodeTemp_{temp_k}K"
                
                if not cath_dir.exists():
                    print(f"Directory not found: {cath_dir}")
                    continue
                    
                print(f"\nProcessing: ElecTemp_{elec_temp}C, CathodeTemp_{temp_k}K")
                
                # Change to current directory
                os.chdir(str(cath_dir))  # Convert Path to string
                
                # Initialize model - convert mkm file path to string
                mkm_path = str(cath_dir / "CO2R.log")  # Convert to string
                model = ReactionModel(setup_file=mkm_path)
                ma = analyze.MechanismAnalysis(model)
                ma.coverage_correction = True
                ma.pressure_correction = True
                ma.energy_type = 'free_energy'
                ma.include_labels = True
                
                # Extract potential list
                potential_list = extract_potentials(model.coverage_map)
                #print(potential_list)
                # # Generate coverage map data
                # all_coverage_df = None
                # for potential in potential_list:
                #     df = create_coverage_dataframe(model.coverage_map, model)
                #     df['potential'] = potential
                #     df['temperature'] = model.temperature
                    
                #     if all_coverage_df is None:
                #         all_coverage_df = df
                #     else:
                #         all_coverage_df = pd.concat([all_coverage_df, df], ignore_index=True)
                
                # # Save coverage map
                # coverage_filename = f"coverage_map_data_elec{elec_temp}K_cath{temp_k}K.csv"
                # all_coverage_df.to_csv(coverage_filename, index=False)
                # print(f"Saved: {coverage_filename}")
                
                # Generate mechanism data
                model = get_all_rxn_steps(model)
                print(model.rxn_mechanisms)
                all_df = None
                
                for potential in potential_list:
                    df = mechanism_to_dataframe(ma, potential=potential)
                    df['potential'] = potential
                    df['temperature'] = model.temperature
                    df = df[['potential', 'temperature', 'step_idx', 'rxn_idx', 
                            'Reaction', 'Ga', 'dG', 'IS', 'is_RDS', 'is_PDS']]
                    
                    if all_df is None:
                        all_df = df
                    else:
                        all_df = pd.concat([all_df, df], ignore_index=True)

                        
                
                # Save mechanism data
                mechanism_filename = f"mechanism_data.csv"
                all_df.to_csv(mechanism_filename, index=False)
                print(f"Saved: {mechanism_filename}")
                
            except Exception as e:
                print(f"Error processing {cath_dir}: {str(e)}")
                continue

def analyze_rds_at_fixed_current(base_dir, target_current, rxn_mechanisms, elec_temps, cath_temps):
    """
    Analyze RDS at fixed current for different temperature combinations
    
    Parameters:
        base_dir (str): Base directory containing the data folders
        target_current (float): Target current density (A/cm²)
        elec_temps (list): List of electrolyte temperatures in Celsius
        cath_temps (list): List of cathode temperatures in Celsius
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from pathlib import Path

    
    # Product-mechanism mapping
    product_mechanisms = {
        'H$_2$(g)': ['HER_Heyrovsky', 'HER_Tafel'],
        'CO(g)': ['CO2R_CO'],
        'CH$_4$(g)': ['CO2R_C1_via_CO-H-ele', 'CO2R_C1_via_H-CO'],
        'C$_2$H$_4$(g)': ['CO2R_C2_via_OCCOH', 'CO2R_C2_via_C-CO', 'CO2R_C2_via_CH-CO']
    }   
    # Define reaction labels
    step_labels = {
        1: 'H$^+$ + e$^-$ + * $\\rightarrow$ H*',
        2: 'H* + H$^+$ + e$^-$ $\\rightarrow$ H$_2$',
        3: 'H* + H* $\\rightarrow$ H$_2$',
        5: 'CO$_2$ + * + H$^+$ + e$^-$ $\\rightarrow$ COOH*',
        6: 'COOH* + H$^+$ + e$^-$ $\\rightarrow$ CO* + H$_2$O',
        7: 'CO* $\\rightarrow$ CO$_g$ + *',
        14: 'CH$_3$* + H$^+$ + e$^-$ $\\rightarrow$ CH$_4$',
        15: 'CO* + H$^+$ + e$^-$ $\\rightarrow$ COH*',
        16: 'COH* + H$^+$ + e$^-$ $\\rightarrow$ C* + H$_2$O',
        18: 'CO* + H* $\\rightarrow$ CHO* + *',
        25: 'CO* + C* $\\rightarrow$ OCC*',
        26: 'CO$_g$ + C* $\\rightarrow$ OCC* + *',
        29: '2CH$_2$* $\\rightarrow$ C$_2$H$_4$$_g$ + 2*',
        30: '2CO* $\\rightarrow$ OCCO*',
        31: 'OCCO* + H$^+$ + e$^-$ $\\rightarrow$ OCCOH*',
        32: 'OCCOH* + H$^+$ + e$^-$ $\\rightarrow$ OCC* + H$_2$O$_g$',
        33: 'CO* + CH* $\\rightarrow$ OCCH* + *',
        34: 'CO* + H$^+$ + e$^-$ $\\rightarrow$ CHO*'
    }
    
    # Initialize data storage
    all_data = []
    
    # Process data for each temperature combination
    for elec_temp in elec_temps:
        for cath_temp in cath_temps:
            cath_temp_K = cath_temp + 273
            folder_path = Path(base_dir) / f"ElecTemp_{elec_temp}C" / f"CathodeTemp_{cath_temp_K}K"
            
            try:
                # Find potential at target current
                current_file = folder_path / "current_density_data.csv"
                if not current_file.exists():
                    continue
                    
                current_df = pd.read_csv(current_file)
                total_current = np.zeros_like(current_df["Potential_vs_RHE(V)"].values)
                for col in current_df.columns:
                    if col != "Potential_vs_RHE(V)":
                        total_current += current_df[col].values
                
                target_pot = find_x_for_target_y(
                    current_df["Potential_vs_RHE(V)"].values,
                    total_current,
                    target_current
                )
                
                print(elec_temp, cath_temp, target_pot)
                if target_pot is None:
                    continue
                
                # Read mechanism data
                mech_file = folder_path / "mechanism_data.csv"
                if not mech_file.exists():
                    print(f"Mechanism file not found for {folder_path}")
                    continue
                    
                mech_df = pd.read_csv(mech_file)
                #display(mech_df)
                
                # Find closest potential in mechanism data
                closest_pot = mech_df['potential'].iloc[(mech_df['potential'] - target_pot).abs().argsort()[:1]]
                #print(closest_pot)
                # Select all data from mech_df where potential is closest to closest_pot
                mech_data = mech_df[abs(mech_df['potential'] - closest_pot.iloc[0]) < 0.0001]
                #print("mech_data", mech_data)
            
                
                # Analyze RDS for each product and its mechanisms
                for product, mechanisms in product_mechanisms.items():
                    for mechanism in mechanisms:
                        mechanism_steps = rxn_mechanisms[mechanism] # [1,2]
                        #print(mechanism_steps)
                        mech_steps_data = mech_data[mech_data['rxn_idx'].isin(mechanism_steps)] # if rxn_idx is 1 or 2, then select the data
                        #print(mech_steps_data
                        
                        if not mech_steps_data.empty:
                            rds_row = mech_steps_data.loc[mech_steps_data['Ga'].idxmax()]
                            step_num = int(rds_row['rxn_idx'])
                            
                            all_data.append({
                                'Product': product,
                                'Mechanism': mechanism,
                                'Electrolyte_Temp': elec_temp,
                                'Cathode_Temp': cath_temp,
                                'Potential_vs_RHE': target_pot,
                                'RDS_Step': step_num,
                                'RDS_Reaction': step_labels.get(step_num, f'Step {step_num}'),
                                'Ga': rds_row['Ga'],
                                'dG': rds_row['dG']
                            })
                            
            except Exception as e:
                print(f"Error processing {folder_path}: {str(e)}")
                continue
    
    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(all_data)
    csv_file = Path(base_dir) / f"rds_analysis_current_{int(target_current)}Acm2.csv"
    results_df.to_csv(csv_file, index=False)
    
    # Create plots
    plot_rds_temperature_trends(results_df, base_dir, target_current, step_labels)
    
    return results_df


def plot_rds_temperature_trends(df, base_dir, target_current, step_labels):
    """Create a combined plot showing RDS trends for all products"""
    # Define step colors
    step_colors = {
        1: '#047499',   # blue
        2: '#a9629f',   # purple
        3: '#009975',   # 초록색 
        5: '#e04555',   # 빨간색 CO2->COOH
        6: '#9467bd',   # 보라색
        7: '#d34467',   # pink (CO* -> COg)
        14: '#eb949e',  # 분홍색 CH3* + H -> CH4
        15: '#7f7f7f',  # 회색
        16: '#bcbd22',  # 연두색
        18: '#b53656',  # 청록색 CO + H -> CHO 17becf
        25: '#e04555',  # 진한 분홍색 ff1493
        26: '#00aeac',  # 연두색 COg + C -> OCC
        29: '#4b0082',  # 남색
        30: '#ffa500',  # 밝은 주황색
        31: '#f2b995',  # 노란색 OCCO->OCCOH
        32: '#8b4513',  # 새들브라운
        33: '#136371',  # 로얄블루
        34: '#e04555'   # 핑크색
    }
    
    # Define alpha values for electrolyte temperatures
    elec_temp_alpha = {
        0: 0.3,    # 원래 색상
        25: 0.6,   # 60% 투명도
        70: 1.0    # 30% 투명도
    }

    # Define markers for different electrolyte temperatures
    elec_markers = {
        0: 'o',     # circle
        25: 's',    # square
        70: 'D'     # diamond
    }

    fig, axes = plt.subplots(2, 2, figsize=(13/1.5, 12/1.5))
    axes = axes.ravel()
    
    products = df['Product'].unique()
    elec_temps = sorted(df['Electrolyte_Temp'].unique())

    
    # Store electrolyte temperature legend handles
    elec_legend_handles = []
    for et in elec_temps:
        elec_legend_handles.append(
            plt.Line2D([0], [0], 
                      marker=elec_markers[et],
                      color='gray',
                      linestyle='None',
                      markersize=7,
                      label=f'Elec. {et}°C')
        )
    
    # Create legend handles with alpha values for electrolyte temperature
    elec_legend_handles_with_alpha = []
    for temp, marker in elec_markers.items():
        handle = plt.Line2D([0], [0],
                           marker=marker,
                           color='black',
                           linestyle='None',
                           markersize=8,
                           alpha=elec_temp_alpha[temp],
                           label=f'Elec {temp}°C',)
        elec_legend_handles_with_alpha.append(handle)
    
    # Track the CO graph axis for later use
    co_ax = None
    
    for idx, product in enumerate(products):
        ax = axes[idx]
        product_data = df[df['Product'] == product]
        
        # Store the CO graph axis
        if product == "CO(g)":
            co_ax = ax
        
        # Get unique RDS steps for this product
        unique_steps = sorted(product_data['RDS_Step'].unique())
        
        # Store legend handles and labels for this subplot
        legend_handles = []
        legend_labels = []
        
        # Plot for each step
        for step in unique_steps:
            step_data = product_data[product_data['RDS_Step'] == step]
            step_color = step_colors.get(step, '#000000')
            
            # Get mechanism name for this step
            mechanism = step_data['Mechanism'].iloc[0]
            step_label = step_labels.get(step, f'Step {step}')
            #full_label = f'{mechanism}: {step_label}'
            full_label = step_label
            
            # Plot for each electrolyte temperature
            for elec_temp in elec_temps:
                temp_data = step_data[step_data['Electrolyte_Temp'] == elec_temp]
                if not temp_data.empty:
                    temp_data = temp_data.sort_values('Cathode_Temp')
                    
                    # Plot line and points
                    ax.plot(temp_data['Cathode_Temp'], temp_data['Ga'],
                           color=step_color,
                           alpha=elec_temp_alpha[elec_temp],
                           linestyle='-')
                    
                    ax.scatter(temp_data['Cathode_Temp'], temp_data['Ga'],
                             color=step_color,
                             marker=elec_markers[elec_temp],
                             alpha=elec_temp_alpha[elec_temp],
                             s=100)
            
            # Add legend handle for this step (only once)
            legend_handles.append(plt.Line2D([0], [0], 
                                          marker='o',
                                          color=step_color,
                                          linestyle='None',
                                          markersize=7,
                                          label=full_label))
       
        ax.set_xlabel('Cathode Temperature (°C)', fontproperties=helvetica_bold_prop)
        ax.set_ylabel('Activation Energy (eV)', fontproperties=helvetica_bold_prop)
        ax.set_title(product[:-3], fontproperties=helvetica_bold_prop)
        #ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.63)
        ax.set_xlim(0, 145)
        ax.set_xticks(np.arange(0, 141, 20))
        ax.set_xticks(np.arange(0, 141, 10), minor=True)
        ax.set_yticks(np.arange(0, 1.61, 0.4))
        ax.set_yticks(np.arange(0, 1.61, 0.2), minor=True)
        ax.tick_params(axis='both', which='major', length=4)
        ax.tick_params(axis='both', which='minor', length=2)
        
        # Add legend for steps under each subplot
        ax.legend(handles=legend_handles,
                 loc='center',
                 bbox_to_anchor=(0.5, -0.3),
                 ncol=2,
                 fontsize=8)
    
    # Add electrolyte temperature legend to the CO graph only if it exists
    if co_ax is not None:
        # Create a second y-axis that shares the same x-axis
        # This is just a trick to add a second legend
        ax2 = co_ax.twinx()
        # Make the second y-axis invisible
        ax2.set_yticks([])
        ax2.set_yticklabels([])
        # Add the electrolyte temperature legend to the second y-axis
        ax2.legend(handles=elec_legend_handles_with_alpha,
                  loc='upper right',
                  fontsize=10)
    
    # Remove the global electrolyte temperature legend
    # fig.legend(handles=elec_legend_handles_with_alpha,
    #           loc='upper right', 
    #           #bbox_to_anchor=(0.99, 0.9),
    #           #title='Electrolyte\nTemperature',
    #           fontsize=10)
    
    #plt.suptitle(f'RDS Analysis at {target_current} A/cm²', y=1.02, fontsize=14, fontproperties=helvetica_bold_prop)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make room for legends
    
    # Save plot
    plt.savefig(Path(base_dir) / f"rds_temperature_trends_{int(target_current)}Acm2.png",
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# rxn_mechanisms = {
#     'HER_Heyrovsky': [1, 2],
#     'HER_Tafel': [1, 3],
#     'CO2R_CO': [5,6,7],
#     'CO2R_HCOOH': [8,9],
#     'CO2R_C1_via_CO-H-ele': [5,6,15,16,17,12,13,14],
#     'CO2R_C1_via_H-CO': [5,6,1,18,10,11,12,13,14],
#     'CO2R_C2_via_OCCOH': [5,6,5,6,30,31,32,27,28],
#     'CO2R_C2_via_C-CO': [5,6,15,16,26,27,28],
#     'CO2R_C2_via_CH-CO': [5,6,15,16,17,5,6,33,28]
# }


# # Example usage:



# cal_dir = base_dir + '/13_Cu100'
# target_current = 100  # A/cm²
# elec_temps = [0, 25, 70]  # °C
# cath_temps = [0, 25, 50, 70, 90, 120, 140]  # °C
# cath_temps_K = [273, 298, 323, 343]
# #analyze_temperature_data(cal_dir, elec_temps, cath_temps_K)
# #results = analyze_rds_at_fixed_current(cal_dir, target_current, rxn_mechanisms, elec_temps, cath_temps)

def plot_reaction_steps_vs_temperature(base_dir, target_potentials, elec_temps, cath_temps, step_indices=None, steps=None, same_y_scale=False):
    """
    Plot rates of specific reaction steps for each temperature combination.
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing temperature data folders
    target_potentials : dict
        Dictionary mapping (elec_temp, cath_temp) to the target potential value
    elec_temps : list
        List of electrolyte temperatures to analyze
    cath_temps : list
        List of cathode temperatures to analyze
    step_indices : list, optional
        List of reaction step indices to plot (1-indexed, excludes Potential_vs_RHE(V) and Temperature(K) columns)
    steps : list, optional
        List of reaction step names to plot (alternative to step_indices)
    same_y_scale : bool, optional
        If True, all subplots will use the same y-axis scale (default: False)
    """
    if step_indices is None and steps is None:
        raise ValueError("Either step_indices or steps must be provided")
    
    # Define line styles for different steps
    line_styles = [
        '-',              # 실선
        '--',             # 대시
        ':',              # 점선
        '-.',             # 대시-점선
        (0, (1, 1)),      # 촘촘한 점선
        (0, (3, 1)),      # 일반 점선 
        (0, (5, 1)),      # 듬성듬성한 점선
        (0, (1, 1, 3, 1)), # 점-대시
        (0, (3, 1, 1, 1)), # 대시-점
        (0, (5, 1, 1, 1)), # 긴대시-점
    ]

    # 전해질 온도별 색상 정의
    elec_color_dict = {
        0: "#24AAE3",    # light blue
        25: "#047499",
        70: "#A6202E"
    }
    
    # Dictionary to store data for CSV export
    csv_data = {
        'Electrolyte_Temperature(C)': [],
        'Cathode_Temperature(C)': [],
        'Elementary_Step': [],
        'Elementary_Step_Rate': [],
        'Potential_vs_RHE(V)': []
    }
    
    # If step_indices is provided, get the corresponding step names
    if step_indices and not steps:
        # First, get all available steps from any dataset
        all_steps = []
        for elec_temp in elec_temps:
            for cath_temp in cath_temps:
                # if (elec_temp, cath_temp) not in target_potentials:
                #     continue
                    
                folder_path = Path(base_dir) / f"ElecTemp_{elec_temp}C" / f"CathodeTemp_{int(cath_temp+273)}K"
                rate_file = folder_path / "rate_map_data.csv"
                
                if rate_file.exists():
                    try:
                        rate_df = pd.read_csv(rate_file)
                        #print(f"Rate DataFrame columns: {rate_df.columns}")
                        # 컬럼 중 Potential_vs_RHE(V)와 Temperature(K)를 제외한 나머지 컬럼이 반응 스텝
                        available_steps = [col for col in rate_df.columns if col not in ['Potential_vs_RHE(V)', 'Temperature(K)']]
                        if available_steps:
                            all_steps = available_steps
                            break
                    except Exception as e:
                        print(f"Error reading steps from {rate_file}: {e}")
                        continue
            
            if all_steps:
                break
        
        if not all_steps:
            raise ValueError("Could not find any reaction steps in the data files")
        
        
        # Map indices to step names (인덱스는 1부터 시작)
        steps = []
        for idx in step_indices:
            # 1부터 시작하는 인덱스를 0부터 시작하는 인덱스로 변환
            idx_zero_based = idx - 1
            if 0 <= idx_zero_based < len(all_steps):
                steps.append(all_steps[idx_zero_based])
            else:
                print(f"Warning: Step index {idx} is out of range (1-{len(all_steps)}), skipping")
        
        # 단계 인덱스와 이름을 출력하여 확인
        #print("Available reaction steps:")
        #for i, step in enumerate(all_steps):
        #    print(f"Index {i+1}: {step}")  # 1부터 시작하는 인덱스로 출력
        
        #print("\nSelected steps for plotting:")
        #for i, step in enumerate(steps):
        #    if i < len(step_indices):
        #        print(f"Step {step_indices[i]}: {step}")
    
    # Verify specified steps exist in at least one dataset
    steps_found = False
    for elec_temp in elec_temps:
        for cath_temp in cath_temps:
            if (elec_temp, cath_temp) not in target_potentials:
                continue
                
            folder_path = Path(base_dir) / f"ElecTemp_{elec_temp}C" / f"CathodeTemp_{int(cath_temp+273)}K"
            rate_file = folder_path / "rate_map_data.csv"
            
            if rate_file.exists():
                try:
                    rate_df = pd.read_csv(rate_file)
                    available_steps = [col for col in rate_df.columns if col not in ['Potential_vs_RHE(V)', 'Temperature(K)']]
                    
                    for step in steps:
                        if step in available_steps:
                            steps_found = True
                            break
                    
                    if steps_found:
                        break
                except Exception as e:
                    print(f"Error checking steps in {rate_file}: {e}")
            
        if steps_found:
            break
            
    if not steps_found:
        print(f"None of the specified steps {steps} found in any dataset.")
        return
        
    # Collect data for all steps first
    all_data = {}
    min_rate = float('inf')
    max_rate = float('-inf')
    
    for step in steps:
        all_data[step] = {}
        for elec_temp in elec_temps:
            all_data[step][elec_temp] = {'cath_temps': [], 'rates': []}
            
            for cath_temp in cath_temps:
                if (elec_temp, cath_temp) not in target_potentials:
                    continue
                    
                target_pot = target_potentials[(elec_temp, cath_temp)]
                folder_path = Path(base_dir) / f"ElecTemp_{elec_temp}C" / f"CathodeTemp_{int(cath_temp+273)}K"
                rate_file = folder_path / "rate_map_data.csv"
                
                if rate_file.exists():
                    try:
                        rate_df = pd.read_csv(rate_file)
                        
                        potentials = rate_df['Potential_vs_RHE(V)'].values
                        
                        if step in rate_df.columns:
                            step_data = rate_df[step].values
                            interpolated_rate = interpolate_point(potentials, step_data, target_pot)
                            
                            if interpolated_rate is not None:
                                rate_value = float(interpolated_rate)
                                all_data[step][elec_temp]['cath_temps'].append(cath_temp)
                                all_data[step][elec_temp]['rates'].append(rate_value)
                                
                                # Update min and max rates for y-axis scaling
                                if rate_value > 0:  # Only consider positive values for log scale
                                    min_rate = min(min_rate, rate_value)
                                    max_rate = max(max_rate, rate_value)
                                
                                # Store elementary step rate data for CSV
                                csv_data['Electrolyte_Temperature(C)'].append(elec_temp)
                                csv_data['Cathode_Temperature(C)'].append(cath_temp)
                                csv_data['Elementary_Step'].append(step)
                                csv_data['Elementary_Step_Rate'].append(rate_value)
                                csv_data['Potential_vs_RHE(V)'].append(target_pot)
                    except Exception as e:
                        print(f"Error processing {rate_file} for step {step}: {e}")
                        continue
    
    # 인덱스 개수에 맞춰 서브플롯 생성
    if step_indices:
        num_plots = len(step_indices)
    else:
        num_plots = len(steps)
    
    rows = int(np.ceil(num_plots / 2))
    cols = min(2, num_plots)
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6*rows))
    if num_plots == 1:
        axes = np.array([axes])  # Ensure axes is always array-like
    axes = axes.flatten()  # 2D 배열을 1D로 변환
    
    # Apply padding to y-axis limits for better visualization
    if same_y_scale and min_rate < float('inf') and max_rate > float('-inf'):
        y_min = min_rate * 0.5  # Add some padding
        y_max = max_rate * 2    # Add some padding
    
    # Plot data
    for plot_idx in range(num_plots):
        if plot_idx >= len(axes):  # 안전 장치
            break
            
        ax = axes[plot_idx]
        
        # 반응 단계 결정
        if plot_idx < len(steps):
            step = steps[plot_idx]
        else:
            # 인덱스는 있지만 해당 스텝이 없는 경우 (예: 건너뛴 경우)
            ax.set_title(f'Step {step_indices[plot_idx]}: Not Found')
            continue
        
        # 각 플롯마다 레전드 항목을 위한 핸들과 라벨 리스트
        handles = []
        labels = []
        
        for elec_temp in elec_temps:
            if elec_temp in all_data[step] and all_data[step][elec_temp]['rates']:
                cath_temps_available = all_data[step][elec_temp]['cath_temps']
                step_rates = all_data[step][elec_temp]['rates']
                
                # 전해질 온도에 해당하는 색상 가져오기
                elec_color = elec_color_dict.get(elec_temp, '#333333')  # 기본값은 회색
                
                # Plot step rate with semilogy for better visualization
                line, = ax.semilogy(cath_temps_available, step_rates, '-o', 
                          color=elec_color, 
                          linewidth=2)
                
                # Add to legend
                handles.append(line)
                labels.append(f'Elec {elec_temp}°C')
        
        # Add original index if step_indices was used
        if step_indices and plot_idx < len(step_indices):
            # Find the index of this step
            step_index = step_indices[plot_idx]
            ax.set_title(f'Step {step_index}: {step}')
        else:
            ax.set_title(f'{step}')
        
        # Set axis labels
        ax.set_ylabel(f'Rate (s$^{{-1}}$)')
        ax.set_xlabel('Cathode Temperature (°C)')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits if same_y_scale is True
        if same_y_scale and min_rate < float('inf') and max_rate > float('-inf'):
            ax.set_ylim(y_min, y_max)
        
        # Add legend
        if handles:
            ax.legend(handles, labels, 
                     loc='best', 
                     fontsize=8)
    
    plt.tight_layout()
    
    # Save CSV data
    csv_df = pd.DataFrame(csv_data)
    csv_file_path = Path(base_dir) / f"reaction_steps_vs_temperature_data.csv"
    csv_df.to_csv(csv_file_path, index=False)
    print(f"Saved data to {csv_file_path}")
    
    # Save plot
    fig_file_path = Path(base_dir) / f"reaction_steps_vs_temperature.png"
    plt.savefig(fig_file_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {fig_file_path}")
    
    return fig