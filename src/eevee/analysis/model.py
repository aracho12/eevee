# src/eevee/analysis/model.py
from typing import Dict, List, Optional
from catmap import ReactionModel

class ReactionModelWrapper:
    """Wrapper class for CatMAP ReactionModel with additional functionality"""
    
    def __init__(self, setup_file: str):
        self.model = ReactionModel(setup_file=setup_file)
        self.disable_interactions()
    
    def disable_interactions(self):
        """Disable adsorbate interactions"""
        self.model.adsorbate_interaction_model = None
        if hasattr(self.model.thermodynamics, 'adsorbate_interactions'):
            self.model.thermodynamics.adsorbate_interactions = None
            
    def add_mechanism(self, name: str, indices: List[int]):
        """Add a new reaction mechanism"""
        self.model.rxn_mechanisms[name] = indices