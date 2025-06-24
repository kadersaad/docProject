import pandas as pd
import numpy as np
import os

def count_total_accepted(decision_files_paths):
    """
    Counts total accepted decisions across multiple decision files.
    Each file is expected to have an 'ID_ZONE' and 'Decision' column.

    Args:
        decision_files_paths (list): List of absolute file paths for CSV files
                                     (e.g., decision_final_decideur_X.csv).

    Returns:
        tuple: (total_accepted_count, accepted_decisions_df)
               total_accepted_count (int): Total number of 'Accepted' decisions.
               accepted_decisions_df (pd.DataFrame): DataFrame containing unique
                                                     ID_ZONEs that were accepted.
    """
    all_accepted_decisions = []
    total_count = 0
    for file_path in decision_files_paths:
        try:
            # Read CSV and clean column names
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip().str.replace(' ', '_').str.upper()

            # Check if 'DECISION' column exists and filter for 'Accepted'
            if 'DECISION' in df.columns:
                accepted_df = df[df['DECISION'] == 'Accepted']
                total_count += len(accepted_df)
                all_accepted_decisions.append(accepted_df)
            else:
                print(f"Warning: 'DECISION' column not found in {file_path}. Skipping file for acceptance count.")
        except FileNotFoundError:
            print(f"Warning: Decision file not found at {file_path}. Skipping.")
            continue
        except Exception as e:
            print(f"Error processing {file_path}: {e}. Skipping file.")
            continue

    if all_accepted_decisions:
        # Concatenate all accepted decisions and remove duplicates based on 'ID_ZONE'
        # This assumes each unique ID_ZONE accepted by any decider is counted once for negotiation.
        return total_count, pd.concat(all_accepted_decisions).drop_duplicates(subset=['ID_ZONE'])
    print("No accepted decisions found in any file. Returning empty DataFrame.")
    return 0, pd.DataFrame()

def compute_best_action(decision_files_paths, accepted_data_df, decision_weights):
    """
    Computes the best action based on weighted ranks from multiple decider files.
    This function expects 'ID_ZONE' and 'RANK' columns in the decision files.

    Args:
        decision_files_paths (list): List of absolute file paths for decision_final_decideur_X.csv.
        accepted_data_df (pd.DataFrame): DataFrame of zones accepted by at least one decider (must contain 'ID_ZONE').
        decision_weights (dict): Dictionary of weights for each decider (e.g., {1: 0.1, 2: 0.1, ...}).
                                 Keys should be decider IDs (integers), values are their weights (floats).

    Returns:
        dict: Best action details (a dictionary representation of a pandas Series) or None if no valid action found.
    """
    if accepted_data_df.empty:
        print("No accepted zones provided for best action computation.")
        return None

    # Dictionary to store DataFrames with 'ID_ZONE' and renamed 'RANK' columns
    dfs_ranks = {}
    for i, file_path in enumerate(decision_files_paths):
        # Extract decider ID from file name, assuming 'decision_final_decideur_X.csv' pattern
        try:
            dec_id = int(os.path.basename(file_path).split('_')[-1].replace('.csv', ''))
        except (ValueError, IndexError):
            print(f"Warning: Could not parse decider ID from file name: {file_path}. Skipping.")
            continue

        try:
            df = pd.read_csv(file_path)
            # Clean column names
            df.columns = df.columns.str.strip().str.replace(' ', '_').str.upper()

            # Ensure necessary columns exist
            if 'ID_ZONE' not in df.columns or 'RANK' not in df.columns:
                print(f"Warning: Missing 'ID_ZONE' or 'RANK' in {file_path}. Skipping file for best action.")
                continue

            # Store only ID_ZONE and RANK, renaming RANK column for clarity
            dfs_ranks[dec_id] = df[['ID_ZONE', 'RANK']].rename(columns={'RANK': f'RANK_DECIDEUR_{dec_id}'})
            
            # Also merge the original 'Decision' column for display purposes in HTML
            if 'DECISION' in df.columns:
                dfs_ranks[dec_id] = pd.merge(dfs_ranks[dec_id], 
                                             df[['ID_ZONE', 'DECISION']].rename(columns={'DECISION': f'DECISION_DECIDEUR_{dec_id}'}),
                                             on='ID_ZONE', how='left')
        except FileNotFoundError:
            print(f"Warning: Decision file not found at {file_path}. Skipping.")
            continue
        except Exception as e:
            print(f"Error loading {file_path}: {e}. Skipping file.")
            continue

    if not dfs_ranks:
        print("No valid decider rank files found for best action computation.")
        return None

    # Start merging with the DataFrame of accepted zones
    merged_df = accepted_data_df[['ID_ZONE']].copy()

    # Merge all individual decider rank and decision DataFrames into a single DataFrame
    for dec_id, df_info in dfs_ranks.items():
        merged_df = pd.merge(merged_df, df_info, on='ID_ZONE', how='left')

    # Get the list of rank columns that were successfully merged
    rank_cols = [f'RANK_DECIDEUR_{i}' for i in sorted(dfs_ranks.keys())] # Ensure consistent order

    # Fill NaN ranks with a large number. This makes unranked zones "bad" for that decider.
    for col in rank_cols:
        if col not in merged_df.columns:
            # If a rank column doesn't exist for a decider, it means no data was processed for them.
            # Assign a very high rank (low preference) for missing deciders too.
            merged_df[col] = 9999.0
        merged_df[col] = merged_df[col].fillna(9999.0).astype(float) # Ensure numeric type

    # Calculate Weighted Rank
    # Ensure weights are applied correctly based on the decider ID
    merged_df['Weighted_Rank'] = 0.0
    valid_weights = {}
    for dec_id, weight_value in decision_weights.items():
        if f'RANK_DECIDEUR_{dec_id}' in merged_df.columns:
            merged_df['Weighted_Rank'] += merged_df[f'RANK_DECIDEUR_{dec_id}'] * weight_value
            valid_weights[dec_id] = weight_value # Store actual weights used

    if merged_df.empty or 'Weighted_Rank' not in merged_df.columns:
        print("Could not compute weighted rank. Merged DataFrame is empty or missing 'Weighted_Rank'.")
        return None

    # Select the best action (lowest weighted rank)
    best_action = merged_df.sort_values(by='Weighted_Rank').iloc[0]

    # Add the actual weights used as part of the result for clarity
    for dec_id, weight_value in valid_weights.items():
        best_action[f'Weight_decideur_{dec_id}'] = weight_value

    return best_action.to_dict() # Convert Series to dictionary for easy JSON serialization
    return best_action.to_dict() # Convert Series to dictionary for easy JSON serialization
