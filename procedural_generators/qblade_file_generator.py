import os
import math
import numpy as np
from datetime import datetime

def generate_qblade_file(naca_profile, twist_angle, solidity):
    """
    Generates a single QBlade definition file based on the given parameters.
    """
    
    # Calculate chord length based on solidity
    # Assuming a vertical axis wind turbine (VAWT) with 3 blades, 
    # Radius = 0.5m (from example). 
    # Solidity = (N * c) / R  => c = (Solidity * R) / N
    # Actually, standard VAWT solidity is often defined as sigma = (N * c) / (2 * R) or similar.
    # Looking at the example: Sol=0.75, Chord=0.125, N=3, R=0.5
    # (3 * 0.125) / 0.5 = 0.75. So, Formula: c = (Solidity * R) / N
    
    radius = 0.50000
    num_blades = 3
    chord = (solidity * radius) / num_blades
    
    # Ensure folders exist
    folder_name = f"NACA_{naca_profile}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    # File naming convention based on example
    # Example: Blade_NACA_0015_Circ80_Sol1.20.bld
    # Formatting to keep it clean (e.g., Sol0.75 instead of Sol0.750000000)
    filename = f"Blade_NACA_{naca_profile}_Circ{twist_angle}_Sol{solidity:.2f}.bld"
    filepath = os.path.join(folder_name, filename)
    
    # Date/Time for header
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    current_date = now.strftime("%d.%m.%Y")
    
    # Pre-calculated Heights based on example (10 sections)
    heights = [0.00000, 0.11111, 0.22222, 0.33333, 0.44444, 0.55556, 0.66667, 0.77778, 0.88889, 1.00000]
    
    # Calculate Circular Angles (linear distribution from 0 to max twist_angle)
    # Looking at example: Circ80 gives values from 0.00 to 80.00
    circ_angles = np.linspace(0, twist_angle, len(heights))
    
    with open(filepath, 'w') as f:
        # Header
        f.write("----------------------------------------QBlade Blade Definition File------------------------------------------------\n")
        f.write("Generated with : QBlade CE v2.0.9.7b_beta windows\n")
        f.write("Archive Format: 310043\n")
        f.write(f"Time : {current_time}\n")
        f.write(f"Date : {current_date}\n\n")
        
        # Object Name
        f.write("----------------------------------------Object Name-----------------------------------------------------------------\n")
        object_name = filename.replace('.bld', '')
        f.write(f"{object_name:<50} OBJECTNAME         - the name of the blade object\n\n")
        
        # Parameters
        f.write("----------------------------------------Parameters------------------------------------------------------------------\n")
        f.write(f"{'VAWT':<50} ROTORTYPE          - the rotor type\n")
        f.write(f"{'false':<50} INVERTEDFOILS      - invert the airfoils? (only VAWT) [bool]\n")
        f.write(f"{'3':<50} NUMBLADES          - number of blades\n\n")
        
        # Blade Data Header
        f.write("----------------------------------------Blade Data------------------------------------------------------------------\n")
        header_cols = ["HEIGHT_[m]", "CHORD_[m]", "RADIUS_[m]", "OFFSET_Y_[m]", "TWIST_[deg]", "CIRCANGLE_[deg]", "P_AXIS_Y_[-]", "POLAR_FILE"]
        
        # Custom formatting string to match the example column widths
        header_format = "{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}\n"
        row_format = "{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.2f}{:<20.2f}{:<20.5f}{:<20}\n"
        
        f.write(header_format.format(*header_cols))
        
        # Write rows
        polar_file = f"Polars/NACA_{naca_profile}_MultiRePolar.plr"
        for i in range(len(heights)):
            h = heights[i]
            c_angle = circ_angles[i]
            f.write(row_format.format(
                h,              # HEIGHT
                chord,          # CHORD
                radius,         # RADIUS
                0.0,            # OFFSET_Y
                0.0,            # TWIST (constant 0.0 in examples)
                c_angle,        # CIRCANGLE
                0.5,            # P_AXIS_Y
                polar_file      # POLAR_FILE
            ))

def generate_all_blades():
    """
    Generates QBlade files for all requested NACA profiles, 
    solidities (0.25 to 2.0 in steps of 0.05), and 
    circular twist angles (0 to 120 in steps of 5).
    """
    profiles = ["0015", "0018", "0021", "0013"]
    twists = range(0, 125, 5) # 0 to 120 inclusive
    
    # Generate solidity ranges, avoiding floating point issues with round()
    # 0.25 to 2.0 inclusive in steps of 0.05
    solidities = [round(x * 0.05, 2) for x in range(int(0.25/0.05), int(2.00/0.05) + 1)]
    
    count = 0
    for profile in profiles:
        for solidity in solidities:
            for twist in twists:
                generate_qblade_file(profile, twist, solidity)
                count += 1
    print(f"Generated {count} total files for NACA {', '.join(profiles)}.")

if __name__ == "__main__":
    print("Starting QBlade file generation...")
    generate_all_blades()
    print("All files generated successfully.")