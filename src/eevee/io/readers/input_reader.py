import pandas as pd
from typing import Dict, Any
from ...core.thermodynamics import safe_eval_frequencies
from catmap import ReactionModel

class InputReader:
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.energy_df = self.read_energy_data()


    def __str__(self):
        return f"InputReader(input_file={self.input_file})"
    
    def read_energy_data(self) -> pd.DataFrame:
        """Read and process input file"""
        df = pd.read_csv(self.input_file, sep='\t')
        
        # Convert frequencies to list format
        df['frequencies'] = df['frequencies'].apply(safe_eval_frequencies)
        
        return df


class MKMReader:
    def __init__(self, mkm_file: str):
        self.mkm_file = mkm_file
        self.mkm_model = ReactionModel(setup_file=self.mkm_file)

    def __str__(self):
        return f"MkmReader(mkm_file={self.mkm_file})"
    
    def get_reaction_model(self) -> ReactionModel:
        return self.mkm_model

    def get_mechanism(self):
        try: 
            return self.mkm_model.rxn_mechanisms.keys()
        except Exception as e:
            print(f"Error getting mechanism: {e}")
            return None
    