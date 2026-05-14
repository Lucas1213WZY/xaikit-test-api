#!/usr/bin/env python3
"""
Extract per-participant cognitive parameters from rl_fit_trials.csv (CSV-only, no pandas)
"""
import csv
import os

# Read input CSV
input_file = 'code_for_papers/old/coxam/rl_fit_trials.csv'
output_file = 'assets/param_config/CoXAM_counterfactual_simulation_cog_param.csv'

# Create directory if needed
os.makedirs('assets/param_config', exist_ok=True)

# Read and process CSV
seen_participants = {}
participant_params = []

with open(input_file, 'r') as f:
    reader = csv.DictReader(f)
    
    for row in reader:
        pid = row['Participant Id']
        
        # Only process first occurrence of each participant
        if pid not in seen_participants:
            seen_participants[pid] = True
            
            param_dict = {
                'Participant Id': pid,
                'Best NLL': row['Best NLL'],
                'Best MAE': row['Best MAE'],
                'Best time': row['Best time'],
                'Best retrieval_threshold': row['Best retrieval_threshold'],
                'Best over_margin': row['Best over_margin'],
                'Best chi': row['Best chi'],
                'app_id': row['app_id'],
                'model': row['model'],
                'complexity': row['complexity'],
                'condition': row['condition'],
            }
            participant_params.append(param_dict)

# Write output CSV
if participant_params:
    fieldnames = participant_params[0].keys()
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(participant_params)
    
    print(f"✓ Extracted {len(participant_params)} unique participants")
    print(f"✓ Saved to {output_file}")
    print(f"\nFirst participant:")
    print(participant_params[0])
else:
    print("ERROR: No participants found")
