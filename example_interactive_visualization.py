#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script demonstrating how to create an interactive HTML visualization
of reaction steps vs temperature data with advanced filtering options.
"""

import os
import sys
from eevee.visualization.plot import interactive_reaction_steps_vs_temperature
from eevee.visualization.plot import rxn_mechanisms, product_to_mechanisms

def main():
    # Define the base directory where your data is stored
    # Update this to match your actual data directory
    base_dir = "."
    
    # If a command-line argument is provided for base_dir, use that instead
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    
    # Check if the directory exists
    if not os.path.exists(base_dir):
        print(f"Directory does not exist: {base_dir}")
        print("Please specify the correct path to the directory containing reaction_steps_vs_temperature_data.csv")
        sys.exit(1)
    
    # Define the electrolyte temperatures to include in the visualization
    elec_temps = [0, 25, 70]
    
    # Define step reactions with correct indices
    step_reactions = {
        1: "H+ + e- + * $ \\rightarrow $ H*",
        2: "H+ + e- + H* $ \\rightarrow $ H2 + *",
        3: "H* + H* $ \\rightarrow $ H2 + * + *",
        4: "H+ + e- + OH* $ \\rightarrow $ H2O + *",
        5: "H+ + e- + CO2 + * + * $ \\rightarrow $ COOH*",
        6: "H+ + e- + COOH* $ \\rightarrow $ CO* + H2O + *",
        7: "CO* $ \\rightarrow $ CO + *",
        8: "e- + CO2 + H* + * $ \\rightarrow $ HCOO*",
        9: "H+ + e- + HCOO* $ \\rightarrow $ HCOOH+ + * + *",
        10: "H+ + e- + CHO* $ \\rightarrow $ CHOH*",
        11: "H+ + e- + CHOH* $ \\rightarrow $ CH* + H2O",
        12: "H+ + e- + CH* $ \\rightarrow $ CH2*",
        13: "H+ + e- + CH2* $ \\rightarrow $ CH3*",
        14: "H+ + e- + CH3* $ \\rightarrow $ CH4 + *",
        15: "H+ + e- + CO* $ \\rightarrow $ COH*",
        16: "H+ + e- + COH* $ \\rightarrow $ C* + H2O",
        17: "H+ + e- + C* $ \\rightarrow $ CH*",
        18: "CO* + H* $ \\rightarrow $ CHO* + *",
        19: "COH* + H* $ \\rightarrow $ CHOH* + *",
        20: "C* + H* $ \\rightarrow $ CH* + *",
        21: "CH* + H* $ \\rightarrow $ CH2* + *",
        22: "CH2* + H* $ \\rightarrow $ CH3* + *",
        23: "CH3* + H* $ \\rightarrow $ CH4 + * + *",
        24: "O2 + * $ \\rightarrow $ O2*",
        25: "CO* + C* $ \\rightarrow $ OCC* + *",
        26: "CO + C* $ \\rightarrow $ OCC*",
        27: "H+ + e- + OCC* $ \\rightarrow $ OCCH*",
        28: "5H+ + 5e- + OCCH* $ \\rightarrow $ C2H4 + H2O + *",
        29: "CH2* + CH2* $ \\rightarrow $ C2H4 + * + *",
        30: "CO* + CO* $ \\rightarrow $ OCCO*",
        31: "H+ + e- + OCCO* $ \\rightarrow $ OCCOH*",
        32: "H+ + e- + OCCOH* $ \\rightarrow $ OCC* + H2O + *",
        33: "CO* + CH* $ \\rightarrow $ OCCH* + *",
        34: "H+ + e- + CO* $ \\rightarrow $ CHO*"
    }
    
    # Build product-to-steps mapping based on mechanisms
    product_steps_mapping = {}
    for product, mechanisms in product_to_mechanisms.items():
        product_steps = []
        for mech in mechanisms:
            if mech in rxn_mechanisms:
                product_steps.extend(rxn_mechanisms[mech])
        product_steps_mapping[product] = sorted(list(set(product_steps)))
    
    # Create the interactive visualization
    print(f"Creating interactive visualization from data in {base_dir}...")
    print(f"Including electrolyte temperatures: {elec_temps}")
    print(f"With 4 filtering options: by mechanism, by product, by intermediate species, and by reaction type")
    
    fig = interactive_reaction_steps_vs_temperature(
        base_dir=base_dir,
        csv_filename="reaction_steps_vs_temperature_data.csv",
        elec_temps=elec_temps,
        height=900,
        width=1400,
        product_steps_mapping=product_steps_mapping,
        step_reactions=step_reactions
    )
    
    print("Done!")
    print("The interactive visualization has been saved as 'interactive_reaction_steps.html' in the specified directory.")
    print("Open this file in a web browser to view and interact with the visualization.")
    print("\nFilter options include:")
    print("  1. By mechanism - to see steps involved in specific reaction mechanisms")
    print("  2. By product - to see steps contributing to specific products (H2, CO, HCOOH, CH4, C2H4)")
    print("  3. By intermediate - to see steps involving specific surface intermediates")
    print("  4. By reaction type - to distinguish between electrochemical and thermochemical reactions")
    print("  5. Y-axis scale - to switch between linear and logarithmic scaling")

if __name__ == "__main__":
    main() 