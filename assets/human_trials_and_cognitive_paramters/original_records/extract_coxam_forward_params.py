#!/usr/bin/env python3
"""
Extract per-participant cognitive parameters from forward_simulation_comparison_v0.4.xlsx
Filters for unique Participant IDs and extracts cognitive model parameters.
"""
import pandas as pd
import os

def extract_coxam_forward_params():
    # Input and output paths
    input_file = 'assets/param_config/forward_simulation_comparison_v0.4.xlsx'
    output_file = 'assets/param_config/CoXAM_forward_simulation_cog_param.csv'
    
    # Parameters to extract
    required_params = [
        'Participant Id',
        'AppId',
        'Complexity',
        'T_enc',
        'T_op',
        'chi_value',
        'ddm_a',
        'ddm_s',
        'lapse',
        'latency_factor',
        'retrieval_threshold'
    ]
    
    # Read Excel file
    print(f"Reading {input_file}...")
    df = pd.read_excel(input_file, sheet_name=0)
    
    print(f"Loaded {len(df)} rows")
    
    # Keep only first row per participant
    df_unique = df.drop_duplicates(subset=['Participant Id'], keep='first')
    
    print(f"Extracted {len(df_unique)} unique participants")
    
    # Select only parameter columns
    df_output = df_unique[required_params].copy()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    df_output.to_csv(output_file, index=False)
    
    print(f"✓ Saved to {output_file}")
    print(f"\nDataset info:")
    print(f"  Rows: {len(df_output)}")
    print(f"  Columns: {len(df_output.columns)}")
    print(f"\nColumns saved:")
    for col in df_output.columns:
        print(f"  - {col}")
    
    print(f"\nFirst participant:")
    print(df_output.iloc[0])

if __name__ == '__main__':
    extract_coxam_forward_params()
