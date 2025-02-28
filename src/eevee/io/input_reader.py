import pandas as pd
from typing import Dict, Any
from ..core.thermodynamics import safe_eval_frequencies

class InputReader:
    def __init__(self, input_file: str):
        self.input_file = input_file
        
    def read_data(self) -> pd.DataFrame:
        """Read and process input file"""
        df = pd.read_csv(self.input_file, sep='\t')
        
        # Verify required columns
        required_cols = ['species_name', 'site_name', 
                        'formation_energy', 'frequencies']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert frequencies to list format
        df['frequencies'] = df['frequencies'].apply(safe_eval_frequencies)
        
        return df
