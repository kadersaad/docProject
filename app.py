# app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import pandas as pd 
from werkzeug.utils import secure_filename # For secure file name handling
import uuid # For unique temporary file names
import json # For converting lists to JSON strings for display
import csv

from core.decideur_logic import run_decideur_analysis
from core.negociateur_logic import count_total_accepted, compute_best_action
from core.clustering_analysis import run_kmeans_analysis

app = Flask(__name__)

# Base directory for data files (assuming 'data' is a sibling folder to 'app.py')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
# Directory for temporarily uploaded files
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'tmp_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # Create upload folder if it doesn't exist

# Directory for static images (plots)
PLOTS_DIR = os.path.join(os.getcwd(), 'static', 'images')
os.makedirs(PLOTS_DIR, exist_ok=True) # Create images folder if it doesn't exist

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Max upload size: 16 MB

ALLOWED_EXTENSIONS = {'csv'}

# Ensure the data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"Created data directory: {DATA_DIR}")

def load_decision_weights(file_path):
    """
    Loads decision weights from a CSV file.
    Expected CSV format: 'Decideur', 'Weight'
    """
    try:
        df = pd.read_csv(file_path)
        # Convert to dictionary {Decideur: Weight}
        # Ensure 'Decideur' is treated as integer keys
        decision_weights = dict(zip(df['Decideur'].astype(int), df['Weight'].astype(float)))
        return decision_weights
    except FileNotFoundError:
        print(f"Error: decision_weights.csv not found at {file_path}. Using default weights or handling error.")
        return None
    except Exception as e:
        print(f"Error loading decision weights from {file_path}: {e}")
        return None

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# DECIDER PRESET PARAMETERS
DECIDER_PRESET_PARAMS = {
    1: {  # Environmental Activist
        'poids':    [20, 5, 15, 5, 5, 5, 45],        # Strong emphasis on CLIMAT, NUISANCES, IMPACTS
        'seuil_p':  [0.4, 0.4, 0.5, 100, 10, 1, 0.2],
        'seuil_q':  [0.2, 0.2, 0.25, 50, 5, 0.5, 0.1]
    },
    2: {  # Urban Planner
        'poids':    [5, 5, 5, 25, 25, 25, 10],       # Emphasis on GEOTECHNIQ, EQUIPEMENT, ACCESSIBIL
        'seuil_p':  [0.6, 0.6, 0.4, 90, 5, 1, 0.5],
        'seuil_q':  [0.3, 0.3, 0.2, 45, 2.5, 0.5, 0.25]
    },
    3: {  # Health & Safety Inspector
        'poids':    [30, 25, 20, 5, 5, 5, 10],       # Focused on NUISANCES, BRUIT, IMPACTS
        'seuil_p':  [0.3, 0.3, 0.3, 100, 10, 1, 0.6],
        'seuil_q':  [0.15, 0.15, 0.15, 50, 5, 0.5, 0.3]
    },
    4: {  # Accessibility Analyst
        'poids':    [5, 5, 5, 25, 20, 30, 10],       # Strong on ACCESSIBIL, GEOTECHNIQ, EQUIPEMENT
        'seuil_p':  [0.6, 0.6, 0.6, 80, 10, 1, 0.5],
        'seuil_q':  [0.3, 0.3, 0.3, 40, 5, 0.5, 0.25]
    }
}


# Global PROMETHEE maximize flags (default)
PROMETHEE_MINIMIZE_FLAGS = [True, True, True, True, True, True, False]

# Utility function for cleaning up files
def cleanup_decision_files():
    """Removes decision files and the nearest_points_to_centroids.csv file."""
    files_to_remove = [os.path.join(DATA_DIR, f"decision_final_decideur_{i}.csv") for i in range(1, 5)]
    files_to_remove.append(os.path.join(DATA_DIR, 'nearest_points_to_centroids.csv'))
    files_to_remove.append(os.path.join(DATA_DIR, 'decision_weights.csv'))
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"File removed: {file_path}")
            except OSError as e:
                print(f"Error removing file {file_path}: {e}")


@app.route('/')
def index():
    """
    Renders the home page, displaying decider information (weights and file existence)
    and negotiation options.
    """
    decider_info = []
    can_compute_best_action = True # Assume true, set to false if any required file is missing

    # Load decision weights
    weights_file_path = os.path.join(DATA_DIR, 'decision_weights.csv')
    decision_weights = load_decision_weights(weights_file_path)

    if decision_weights is None:
        print("Warning: Could not load decision weights. Displaying without weights.")
        decision_weights = {} # Use an empty dict if weights couldn't be loaded

    # Check for existence of each decider's file and gather info
    for i in range(1, 5): # Assuming deciders 1 to 4
        file_path = os.path.join(DATA_DIR, f"decision_final_decideur_{i}.csv")
        file_exists = os.path.exists(file_path)
        
        # If any of the required files for best action computation is missing,
        # set can_compute_best_action to False.
        if not file_exists:
            can_compute_best_action = False

        decider_info.append({
            'id': i,
            'file_exists': file_exists,
            'weight': decision_weights.get(i, 'N/A') # Get weight, or 'N/A' if not found
        })

    # Check if nearest_points_to_centroids.csv exists
    nearest_points_exists = os.path.exists(os.path.join(DATA_DIR, 'nearest_points_to_centroids.csv'))

    return render_template(
        'index.html',
        decider_info=decider_info,
        can_compute_best_action=can_compute_best_action,
        nearest_points_exists=nearest_points_exists
    )

@app.route('/decideur', methods=['GET', 'POST'])
def decideur_page():
    """Handles the Decider Analysis page, displaying the input form and results."""
    results = None
    error = None
    kmeans_file_warning = None # Specific warning if nearest_points.csv is missing
    
    # Get the decider ID from query parameters for prefill, if available
    prefill_decideur_id = request.args.get('decideur_id', type=int)
    if prefill_decideur_id not in [1, 2, 3, 4]:
        prefill_decideur_id = 4 # Default value if invalid or not provided

    # Set initial values for form fields (GET request or if POST fails validation)
    selected_preset = DECIDER_PRESET_PARAMS.get(prefill_decideur_id, DECIDER_PRESET_PARAMS[4]) # Default to decider 4
    # Convert lists to comma-separated strings for HTML input fields
    prefill_poids_str = ','.join(map(str, selected_preset['poids']))
    prefill_seuil_p_str = ','.join(map(str, selected_preset['seuil_p']))
    prefill_seuil_q_str = ','.join(map(str, selected_preset['seuil_q']))

    # Check for existence of nearest_points_to_centroids.csv on GET or if POST does not provide new data
    nearest_points_path = os.path.join(DATA_DIR, 'nearest_points_to_centroids.csv')
    if not os.path.exists(nearest_points_path):
        kmeans_file_warning = "The file 'nearest_points_to_centroids.csv' is missing. Please run the K-Means analysis on the 'Negotiator Analysis' page to generate it if you want to use the auto-load option."

    if request.method == 'POST':
        try:
            # Get the decider ID from the form input (prioritize form over query param, update prefill for display)
            decideur_id = int(request.form.get('decideur_id', prefill_decideur_id))
            prefill_decideur_id = decideur_id # Update for potential rendering

            # Get the new data matrix as a raw string (CSV format)
            new_data_matrix_str = request.form['new_data_matrix'].strip() # Remove whitespace

            # Get the weights and thresholds editable by the user from the form
            poids_input_str = request.form['poids_input'].strip()
            seuil_p_input_str = request.form['seuil_p_input'].strip()
            seuil_q_input_str = request.form['seuil_q_input'].strip()

            # Update prefill values based on user input (for rendering if an error occurs)
            prefill_poids_str = poids_input_str
            prefill_seuil_p_str = seuil_p_input_str
            prefill_seuil_q_str = seuil_q_input_str
            
            # Call the core decider logic function with user-provided parameters
            results = run_decideur_analysis(
                decideur_id, 
                new_data_matrix_str, 
                poids_input_str, 
                seuil_p_input_str, 
                seuil_q_input_str,
                PROMETHEE_MINIMIZE_FLAGS
            )
            
            # Pass the plot URL for HTML display
            results['swot_plot_url'] = results['swot_plot_path']

        except FileNotFoundError as e: # Specifically catch if nearest_points.csv is missing
            error = f"{e}. Please run the K-Means analysis on the 'Negotiator Analysis' page to generate this file."
            print(f"Decider analysis FileNotFoundError: {e}")
        except ValueError as e:
            error = f"Validation error: {e}. Make sure all fields are correctly filled (integers, floats, CSV format)."
            print(f"Decider analysis ValueError: {e}")
        except Exception as e:
            # Catch all other unexpected errors during analysis
            error = f"An unexpected error occurred during decider analysis: {e}"
            print(f"Decider analysis general error: {e}") # Log error for debugging

    # Render the decider page, passing results, errors, prefill ID and parameters
    return render_template('decideur.html', 
                           results=results, 
                           error=error, 
                           prefill_decideur_id=prefill_decideur_id,
                           prefill_poids_str=prefill_poids_str,
                           prefill_seuil_p_str=prefill_seuil_p_str,
                           prefill_seuil_q_str=prefill_seuil_q_str,
                           kmeans_file_warning=kmeans_file_warning)

@app.route('/negociateur', methods=['GET', 'POST'])
def negociateur_page():
    global PROMETHEE_MINIMIZE_FLAGS
    best_action_result = None
    kmeans_results = None
    error = None
    kmeans_data_path_used = None

    uploaded_file_path = None
    try:
        if request.method == 'POST' and 'data_file' in request.files:
            file = request.files['data_file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                unique_filename = str(uuid.uuid4()) + '_' + filename
                uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(uploaded_file_path)
                kmeans_data_path_used = uploaded_file_path
                print(f"Uploaded file saved as: {uploaded_file_path}")

                # --- Only set PROMETHEE_MINIMIZE_FLAGS if a line starts with 'promethee_minimize' ---
                try:
                    with open(uploaded_file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line.lower().startswith('promethee_minimize'):
                                parts = [v.strip().lower() for v in line.split(',')[1:]]
                                bool_map = {'true': True, 'false': False, '1': True, '0': False}
                                if all(v in bool_map for v in parts) and len(parts) >= 7:
                                    PROMETHEE_MINIMIZE_FLAGS = [bool_map[v] for v in parts[:7]]
                                    print(f"PROMETHEE_MINIMIZE_FLAGS set from CSV: {PROMETHEE_MINIMIZE_FLAGS}")
                                else:
                                    PROMETHEE_MINIMIZE_FLAGS = [True, True, True, True, True, True, False]
                                    print("PROMETHEE_MINIMIZE_FLAGS set to default (invalid or not found in CSV).")
                                break
                        else:
                            PROMETHEE_MINIMIZE_FLAGS = [True, True, True, True, True, True, False]
                except Exception as e:
                    PROMETHEE_MINIMIZE_FLAGS = [True, True, True, True, True, True, False]
                    print(f"Error reading PROMETHEE_MINIMIZE_FLAGS from CSV: {e}")
            elif file and not allowed_file(file.filename):
                error = "File type not allowed. Please upload a CSV file."
        # If no file is uploaded or it's invalid, try to use the default data_complet.csv file
        if not kmeans_data_path_used:
            default_data_path = os.path.join(DATA_DIR, 'data_complet.csv')
            if os.path.exists(default_data_path):
                kmeans_data_path_used = default_data_path
                PROMETHEE_MINIMIZE_FLAGS = [True, True, True, True, True, True, False]
            else:
                if not error:
                    error = "File 'data_complet.csv' not found in the 'data/' folder. Please upload or place it in the 'data/' folder."

        # --- K-Means Analysis (do NOT use promethee_minimize line in data) ---
        if kmeans_data_path_used:
            # Only use the data rows, skip any 'promethee_minimize' line for KMeans
            # But since run_kmeans_analysis expects a file path, and the function itself should not care about the special line,
            # we do not change the file or function here.
            kmeans_results = run_kmeans_analysis(kmeans_data_path_used)
            if 'kmeans_plot_path' in kmeans_results:
                kmeans_results['kmeans_plot_url'] = '/static/images/' + os.path.basename(kmeans_results['kmeans_plot_path'])
        else:
            if not error:
                error = "No data file is available for K-Means analysis."
    except Exception as e:
        error = f"Error during K-Means analysis: {e}"
        print(f"K-Means error: {e}") # Log error for debugging
    finally:
        # Clean up the temporary uploaded file after analysis
        if uploaded_file_path and os.path.exists(uploaded_file_path):
            try:
                os.remove(uploaded_file_path)
                print(f"Temporary file cleaned up: {uploaded_file_path}")
            except OSError as e:
                print(f"Error removing temporary file {uploaded_file_path}: {e}")

    if request.method == 'POST':
        try:
            # Get individual weights from the four separate input fields
            decision_weights = {
                1: float(request.form.get('weight_decideur_1', 0)),
                2: float(request.form.get('weight_decideur_2', 0)),
                3: float(request.form.get('weight_decideur_3', 0)),
                4: float(request.form.get('weight_decideur_4', 0))
            }

            # Ensure at least one weight is provided (and not zero)
            if not any(weight > 0 for weight in decision_weights.values()):
                raise ValueError("Please provide at least one non-zero weight for a decider to start negotiation.")


            # --- NEW: Save decision_weights to a CSV file ---
            weights_file_path = os.path.join(DATA_DIR, 'decision_weights.csv')
            with open(weights_file_path, 'w', newline='') as csvfile:
                fieldnames = ['Decideur', 'Weight']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for decideur, weight in decision_weights.items():
                    writer.writerow({'Decideur': decideur, 'Weight': weight})
            # --- END NEW ---

            # Prepare the list of decision file paths for the relevant deciders
            files = [os.path.join(DATA_DIR, f"decision_final_decideur_{i}.csv")
                    for i in decision_weights.keys()]

            # Step 1: Count total accepted decisions from the relevant files
            accepted_count, accepted_data = count_total_accepted(files)

            # Step 2: Compute the best action based on accepted data and weights
            best_action_result = compute_best_action(files, accepted_data, decision_weights)

        except ValueError as e:
            error = f"Weight or decision file validation error: {e}"
        except Exception as e:
            # Catch and log any other unexpected error during negotiation
            error = f"An unexpected error occurred during negotiator analysis: {e}"
            print(f"Negotiator error: {e}") # Log error for debugging

    # Render the negotiator page, passing negotiation results, K-Means results, and any errors
    return render_template('negociateur.html', 
                           best_action=best_action_result, 
                           kmeans_results=kmeans_results, 
                           error=error,
                           kmeans_data_path_used=os.path.basename(kmeans_data_path_used) if kmeans_data_path_used else 'N/A')


@app.route('/best_action')
def best_action_page():
    """Handles the 'Best Selected Action' page, performs calculation and cleanup."""
    best_action_result = None
    error = None
    
    try:
        # Define the path to the decision_weights.csv file
        weights_file_path = os.path.join(DATA_DIR, 'decision_weights.csv')
        decision_weights = load_decision_weights(weights_file_path)

        if decision_weights is None:
            error = "Decider weights could not be loaded. Please check the file 'decision_weights.csv'."
            print("Decision weights could not be loaded, returning early.")
            return render_template('best_action.html', best_action=best_action_result, error=error)
            
        # Construct file paths for decision files
        files = [os.path.join(DATA_DIR, f"decision_final_decideur_{i}.csv") for i in range(1, 5)]

        # Execute negotiation logic using the loaded weights
        accepted_count, accepted_data = count_total_accepted(files)
        
        if accepted_data.empty:
            error = "No 'Accepted' zone found in the decision files. Please check the file contents."
            print("No 'Accepted' zones found, returning early from best_action_page.")
        else:
            best_action_result = compute_best_action(files, accepted_data, decision_weights)

            # After successful calculation, perform file cleanup (uncomment if needed)
            cleanup_decision_files() # You would need to implement this function

    except Exception as e:
        error = f"An error occurred while calculating the best action: {e}"
        print(f"Error in best_action_page: {e}")

    return render_template('best_action.html', best_action=best_action_result, error=error)

# --- API Endpoints (unchanged) ---
@app.route('/api/decideur', methods=['POST'])
def api_decideur():
    """API endpoint for Decider Analysis (JSON input/output)."""
    try:
        data = request.get_json() # Expects a JSON payload
        decideur_id = data.get('decideur_id')
        new_data_matrix_str = data.get('new_data_matrix') # Expects a CSV-like string
        poids_str = data.get('poids_input')
        seuil_p_str = data.get('seuil_p_input')
        seuil_q_str = data.get('seuil_q_input')

        if not decideur_id or not new_data_matrix_str or not poids_str or not seuil_p_str or not seuil_q_str:
            return jsonify({'error': 'Missing required parameters for decider analysis'}), 400

        results = run_decideur_analysis(decideur_id, new_data_matrix_str, poids_str, seuil_p_str, seuil_q_str)
        return jsonify(results), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Internal server error: {e}'}), 500

@app.route('/api/negociateur', methods=['POST'])
def api_negociateur():
    """API endpoint for Negotiator Analysis (JSON input/output)."""
    try:
        data = request.get_json()
        decision_weights = data.get('decision_weights') # Expects a dict: {"1": 0.1, "2": 0.1, ...}

        if not decision_weights:
            return jsonify({'error': 'Missing decision weights'}), 400
        
        # Convert keys to int if they arrived as strings from JSON
        decision_weights_int_keys = {int(k): v for k, v in decision_weights.items()}

        files = [os.path.join(DATA_DIR, f"decision_final_decideur_{i}.csv") for i in decision_weights_int_keys.keys()]

        accepted_count, accepted_data = count_total_accepted(files)
        best_action_result = compute_best_action(files, accepted_data, decision_weights_int_keys)

        return jsonify(best_action_result), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Internal server error: {e}'}), 500


if __name__ == '__main__':
    app.run(debug=True)