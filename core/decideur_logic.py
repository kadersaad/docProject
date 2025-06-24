import pandas as pd
import numpy as np
import math
import matplotlib
matplotlib.use('Agg') # Use the Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import os
from io import StringIO

# Ensure the 'static/images' directory exists for plots
PLOTS_DIR = os.path.join(os.getcwd(), 'static', 'images')
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- PROMETHEE II Functions (Keep the previously corrected versions) ---

def preference_function(d, p, q):
    if d <= q:
        return 0
    elif d >= p:
        return 1
    else:
        return (d - q) / (p - q)

def compute_preference_matrix(data, w, p, q, ids, maximize):
    n_alternatives, n_criteria = data.shape
    preference_matrix = np.zeros((n_alternatives, n_alternatives))

    for i in range(n_alternatives):
        for j in range(n_alternatives):
            if i != j:
                aggregate_preference = 0
                for k in range(n_criteria):
                    d = data[i][k] - data[j][k]
                    if maximize[k]:
                        d = -d
                    pref = preference_function(d, p[k], q[k])
                    weighted_pref = w[k] * pref
                    aggregate_preference += weighted_pref
                preference_matrix[i][j] = aggregate_preference
    return preference_matrix

def compute_discordance_matrix(data, p, q, maximize):
    n_alternatives, n_criteria = data.shape
    discordance_matrix = np.zeros((n_alternatives, n_alternatives))

    for i in range(n_alternatives):
        for j in range(n_alternatives):
            if i == j:
                continue
            
            max_individual_discordance = 0
            for k in range(n_criteria):
                diff_jk_ik = data[j][k] - data[i][k]
                
                adjusted_diff = diff_jk_ik if maximize[k] else -diff_jk_ik
                
                if adjusted_diff > 0:
                    individual_discordance = preference_function(adjusted_diff, p[k], q[k])
                    max_individual_discordance = max(max_individual_discordance, individual_discordance)
                
            discordance_matrix[i][j] = max_individual_discordance
                
    return discordance_matrix


def compute_flows(preference_matrix, ids):
    n = preference_matrix.shape[0]
    phi_plus = preference_matrix.sum(axis=1) / (n - 1)
    phi_minus = preference_matrix.sum(axis=0) / (n - 1)
    phi = phi_plus - phi_minus

    print("\n=== Flows ===")
    for i in range(n):
        print(f"{ids[i]}: Phi+ = {phi_plus[i]:.4f}, Phi- = {phi_minus[i]:.4f}, Phi = {phi[i]:.4f}")
    
    return phi, phi_plus, phi_minus

def promethee_ii(columns, data_np, ids, w, p, q, maximize):
    print("\n=== Input Data Matrix (for PROMETHEE II) ===")
    print(pd.DataFrame(data_np, index=ids, columns=columns))

    preference_matrix_np = compute_preference_matrix(data_np, w, p, q, ids, maximize)
    discordance_matrix_np = compute_discordance_matrix(data_np, p, q, maximize)

    phi, phi_plus, phi_minus = compute_flows(preference_matrix_np, ids)

    result = pd.DataFrame({
        'ID_ZONE': ids,
        'Phi+ (Positive Flow)': phi_plus,
        'Phi- (Negative Flow)': phi_minus,
        'Net Flow (Phi)': phi
    })

    result = result.sort_values(by='Net Flow (Phi)', ascending=False).reset_index(drop=True)
    result['Rank'] = result.index + 1

    print("\n=== PROMETHEE II Final Ranking ===")
    print(result)
    
    preference_matrix_df = pd.DataFrame(preference_matrix_np, index=ids, columns=ids)
    discordance_matrix_df = pd.DataFrame(discordance_matrix_np, index=ids, columns=ids)

    print("\n=== Preference Matrix ===")
    print(preference_matrix_df)
    print("\n=== Discordance Matrix ===")
    print(discordance_matrix_df)

    return result, preference_matrix_df, discordance_matrix_df

# --- SWOT Analysis Function ---

def calculate_swot_status(row, criteria_cols):
    swot_scores = {
        'F': 0, 'W': 0, 'O': 0, 'T': 0
    }

    if row['NUISANCES'] <= 0.5:
        swot_scores['O'] += 1
    else:
        swot_scores['T'] += 1

    if row['BRUIT'] <= 0.5:
        swot_scores['O'] += 1
    else:
        swot_scores['T'] += 1

    if row['IMPACTS'] <= 3:
        swot_scores['O'] += 1
    else:
        swot_scores['T'] += 1

    if row['GEOTECHNIQ'] <= 3:
        swot_scores['O'] += 1
    else:
        swot_scores['T'] += 1

    if row['EQUIPEMENT'] <= 1122:
        swot_scores['F'] += 1
    else:
        swot_scores['W'] += 1

    if row['ACCESSIBIL'] <= 7.5:
        swot_scores['F'] += 1
    else:
        swot_scores['W'] += 1

    if row['CLIMAT'] >= 0.5:
        swot_scores['O'] += 1
    else:
        swot_scores['T'] += 1
    
    positive_score = swot_scores['F'] + swot_scores['O']
    negative_score = swot_scores['W'] + swot_scores['T']
    net_score = positive_score - negative_score

    row['F'] = swot_scores['F']
    row['O'] = swot_scores['O']
    row['W'] = swot_scores['W']
    row['T'] = swot_scores['T']
    row['Net_SWOT_Score'] = net_score

    if net_score >= 0:
        row['SWOT_Status'] = 'Positive SWOT'
    else:
        row['SWOT_Status'] = 'Negative SWOT'

    row['X_SWOT'] = row['O'] - row['T']
    row['Y_SWOT'] = row['F'] - row['W']

    return row

# --- Main Flask Logic Function ---

def run_decideur_analysis(decideur_id, new_data_matrix_str, poids_str, seuil_p_str, seuil_q_str):
    try:
        try:
            poids = np.array([float(x.strip()) for x in poids_str.split(',') if x.strip()])
            seuil_p = np.array([float(x.strip()) for x in seuil_p_str.split(',') if x.strip()])
            seuil_q = np.array([float(x.strip()) for x in seuil_q_str.split(',') if x.strip()])
        except ValueError:
            raise ValueError("Weights or thresholds (p, q) are badly formatted. Please use comma-separated numbers.")
        
        if poids.size == 0 or seuil_p.size == 0 or seuil_q.size == 0:
            raise ValueError("Weight or threshold (p, q) lists cannot be empty. Please provide values.")

        criteria_cols = ['NUISANCES', 'BRUIT', 'IMPACTS', 'GEOTECHNIQ', 'EQUIPEMENT', 'ACCESSIBIL', 'CLIMAT']
        
        if not (len(poids) == len(criteria_cols) and 
                len(seuil_p) == len(criteria_cols) and 
                len(seuil_q) == len(criteria_cols)):
            raise ValueError(f"Inconsistency: The number of criteria ({len(criteria_cols)}) does not match the number of weights ({len(poids)}), preference thresholds ({len(seuil_p)}), or indifference thresholds ({len(seuil_q)}) provided. Please check your inputs.")

        selected_params = {
            'poids': poids.tolist(),
            'seuil_p': seuil_p.tolist(),
            'seuil_q': seuil_q.tolist()
        }

        df_input = pd.DataFrame()

        if not new_data_matrix_str:
            nearest_points_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'nearest_points_to_centroids.csv')
            if not os.path.exists(nearest_points_path):
                raise FileNotFoundError(f"The file '{os.path.basename(nearest_points_path)}' is missing. It is generated by the K-Means analysis in the 'Negotiator Analysis' page.")
            
            try:
                df_input = pd.read_csv(nearest_points_path)
                df_input.columns = df_input.columns.str.strip()
                if 'ID_ZONE' not in df_input.columns:
                     raise KeyError(f"The file '{os.path.basename(nearest_points_path)}' must contain a column 'ID_ZONE'.")
                if not all(col in df_input.columns for col in criteria_cols):
                    missing_cols = [col for col in criteria_cols if col not in df_input.columns]
                    raise KeyError(f"The file '{os.path.basename(nearest_points_path)}' does not contain all columns needed for PROMETHEE II analysis (missing: {', '.join(missing_cols)}).")
                df_input = df_input[['ID_ZONE'] + criteria_cols].copy()
                print(f"Data loaded from {os.path.basename(nearest_points_path)}")
            except Exception as e:
                raise ValueError(f"Error loading or processing the file '{os.path.basename(nearest_points_path)}': {e}")
        else:
            new_data_io = StringIO(new_data_matrix_str)
            try:
                df_input = pd.read_csv(new_data_io, header=None, names=['ID_ZONE'] + criteria_cols)
            except Exception as e:
                raise ValueError(f"Error reading the provided Data Matrix: {e}. Make sure the format is correct (CSV type, no header, comma-separated columns) and the number of columns is correct.")
            
            df_input['ID_ZONE'] = df_input['ID_ZONE'].astype(int)
            for col in criteria_cols:
                df_input[col] = pd.to_numeric(df_input[col], errors='coerce')
            df_input.dropna(subset=criteria_cols, inplace=True)
            print("Data loaded from the text area input.")

        if df_input.empty:
            raise ValueError("No valid data could be loaded for analysis.")

        ids = df_input['ID_ZONE'].values
        X = df_input[criteria_cols].values

        promethee_maximize_flags = [True, True, True, True, True, True, False]
        
        ranking, preference_matrix_df, discordance_matrix_df = promethee_ii(criteria_cols, X, ids, poids, seuil_p, seuil_q, promethee_maximize_flags)

        df_swot_calculated = df_input.apply(lambda row: calculate_swot_status(row, criteria_cols), axis=1)
        
        swot_columns_to_merge = ['ID_ZONE', 'F', 'W', 'O', 'T', 'Net_SWOT_Score', 'SWOT_Status', 'X_SWOT', 'Y_SWOT']
        df_swot_final_for_plot_and_display = df_swot_calculated[swot_columns_to_merge].copy()

        ranking_with_swot = pd.merge(ranking, df_swot_final_for_plot_and_display, on='ID_ZONE', how='left')

        print("\n=== PROMETHEE II Ranking with SWOT Status ===")
        print(ranking_with_swot)

        total_actions = len(ranking_with_swot)
        top_n = math.floor(0.5 * total_actions)

        def apply_decision(row, top_n_limit):
            if row['Rank'] <= top_n_limit:
                return "Accepted" if row['SWOT_Status'] == "Positive SWOT" else "Denied"
            else:
                return "Denied"

        ranking_with_swot['Decision'] = ranking_with_swot.apply(lambda row: apply_decision(row, top_n), axis=1)

        final_result_for_display = ranking_with_swot[['ID_ZONE', 'Rank', 'SWOT_Status', 'Decision', 'X_SWOT', 'Y_SWOT']].copy()
        print("\n=== Final Decision (All actions, top 50% evaluated by SWOT) ===")
        print(final_result_for_display)
        
        # Prepare data for CSV file (EXCLUDE X_SWOT, Y_SWOT)
        final_result_for_save = ranking_with_swot[['ID_ZONE', 'Rank', 'SWOT_Status', 'Decision']].copy()
        
        decision_filename = f"decision_final_decideur_{decideur_id}.csv"
        decision_file_path = os.path.join(os.path.dirname(__file__), '..', 'data', decision_filename)
        final_result_for_save.to_csv(decision_file_path, index=False, encoding='utf-8-sig')
        print(f"\nDecision file '{decision_filename}' saved for the negotiator.")

        # --- Generate SWOT Plot ---
        plt.figure(figsize=(10, 8))
        
        max_coord = max(final_result_for_display['X_SWOT'].abs().max(), final_result_for_display['Y_SWOT'].abs().max()) + 1
        plot_limit = max(6, int(max_coord))
        x_line = np.linspace(-plot_limit, plot_limit, 100)
        plt.plot(x_line, -x_line, color='blue', linestyle='--', label='Neutral SWOT Line (Net Score = 0)')

        colors_map = {'Accepted': 'green', 'Denied': 'red'}

        if not final_result_for_display.empty:
            accepted_points = final_result_for_display[final_result_for_display['Decision'] == 'Accepted']
            refused_points = final_result_for_display[final_result_for_display['Decision'] == 'Denied']

            if not accepted_points.empty:
                plt.scatter(accepted_points['X_SWOT'], accepted_points['Y_SWOT'], 
                            color=colors_map['Accepted'], s=80, alpha=0.7, label='Accepted')
            if not refused_points.empty:
                plt.scatter(refused_points['X_SWOT'], refused_points['Y_SWOT'], 
                            color=colors_map['Denied'], s=80, alpha=0.7, label='Denied')

            for i, row in final_result_for_display.iterrows():
                plt.text(row['X_SWOT'] + 0.1, row['Y_SWOT'] + 0.1, str(int(row['ID_ZONE'])), fontsize=9, color='black')

        plt.title(f"SWOT 2D Plot: Zones Positioning (Decider {decideur_id})")
        plt.xlabel("Opportunities (O) - Threats (T) → (External Factors)")
        plt.ylabel("Strengths (F) - Weaknesses (W) → (Internal Factors)")

        plt.xlim(-plot_limit, plot_limit)
        plt.ylim(-plot_limit, plot_limit)
        
        plt.axvline(0, color='grey', linestyle=':', linewidth=0.8)
        plt.axhline(0, color='grey', linestyle=':', linewidth=0.8)
        
        plt.text(0.75 * plot_limit, 0.75 * plot_limit, "Favorable (F+O)", fontsize=9, color='darkgreen', ha='right', va='top')
        plt.text(0.75 * plot_limit, -0.75 * plot_limit, "Unfavorable (F+T)", fontsize=9, color='darkred', ha='right', va='bottom')
        plt.text(-0.75 * plot_limit, 0.75 * plot_limit, "Unfavorable (W+O)", fontsize=9, color='darkred', ha='left', va='top')
        plt.text(-0.75 * plot_limit, -0.75 * plot_limit, "Unfavorable (W+T)", fontsize=9, color='darkred', ha='left', va='bottom')

        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()

        plot_filename = f'swot_decideur_{decideur_id}.png' 
        plot_path = os.path.join(PLOTS_DIR, plot_filename)
        plt.savefig(plot_path)
        plt.close()

        # Prepare data for return as list of dictionaries for easier HTML table rendering
        sub_matrix_processed_display = df_input[['ID_ZONE'] + criteria_cols].to_dict(orient='records')
        final_df_swot_display = df_swot_final_for_plot_and_display.to_dict(orient='records')
        preference_matrix_display = preference_matrix_df.reset_index().rename(columns={'index': 'ID_ZONE'}).to_dict(orient='records')
        discordance_matrix_display = discordance_matrix_df.reset_index().rename(columns={'index': 'ID_ZONE'}).to_dict(orient='records')

        # NEW: Create a version of final_result_for_display *without* X_SWOT and Y_SWOT for the HTML table
        final_decision_table_data_for_html = final_result_for_display[['ID_ZONE', 'Rank', 'SWOT_Status', 'Decision']].to_dict(orient='records')

        return {
            'decideur_id': decideur_id,
            'decideur_params': selected_params,
            'sub_matrix_processed': sub_matrix_processed_display,
            'final_df_swot': final_df_swot_display,
            'preference_matrix': preference_matrix_display,
            'discordance_matrix': discordance_matrix_display,
            'ranking_table': ranking.to_html(classes='table table-striped table-bordered', index=False),
            'final_decision_data': final_decision_table_data_for_html, 
            'swot_plot_path': '/static/images/' + plot_filename
        }

    except Exception as e:
        print(f"Error in run_decideur_analysis: {e}")
        raise e