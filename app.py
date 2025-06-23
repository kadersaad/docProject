# app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import pandas as pd 
from werkzeug.utils import secure_filename # Pour une gestion sécurisée des noms de fichiers
import uuid # Pour des noms de fichiers temporaires uniques
import json # Pour convertir des listes en chaînes JSON pour l'affichage
import csv

from core.decideur_logic import run_decideur_analysis
from core.negociateur_logic import count_total_accepted, compute_best_action
from core.clustering_analysis import run_kmeans_analysis

app = Flask(__name__)

# Répertoire de base pour les fichiers de données (supposant que 'data' est un dossier frère de 'app.py')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
# Répertoire pour les fichiers téléchargés temporairement
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'tmp_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # Crée le dossier de téléchargement s'il n'existe pas

# Répertoire pour les images statiques (plots)
PLOTS_DIR = os.path.join(os.getcwd(), 'static', 'images')
os.makedirs(PLOTS_DIR, exist_ok=True) # Crée le dossier d'images s'il n'existe pas

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Taille maximale de téléchargement : 16 Mo

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
    """Vérifie si l'extension du fichier est autorisée."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# PARAMÈTRES PRÉRÉGLÉS DU DÉCIDEUR (extraits de decideur_logic.py pour la gestion du pré-remplissage par app.py)
DECIDER_PRESET_PARAMS = {
    1: {'poids': [4.96, 7.08, 17.31, 18.93, 18.93, 17.52, 15.27],
        'seuil_p': [0.7, 0.7, 0.6, 100, 8, 1, 0.7],
        'seuil_q': [0.35, 0.35, 0.3, 50, 4, 0.5, 0.35]},
    2: {'poids': [7.51, 13.63, 13.63, 13.63, 17.2, 17.2, 17.2],
        'seuil_p': [0.6, 0.6, 0, 110, 10, 0.6, 0.6],
        'seuil_q': [0.3, 0.3, 0, 55, 5, 0.3, 0.3]},
    3: {'poids': [6.15, 19.57, 13.79, 13.79, 13.79, 16.45, 16.45],
        'seuil_p': [0.4, 0.4, 0.2, 60, 4, 0.6, 0.4],
        'seuil_q': [0.2, 0.2, 0.1, 30, 2, 0.15, 0.2]},
    4: {'poids': [17.38, 29.4, 6.16, 6.16, 6.16, 17.38, 17.38],
        'seuil_p': [0.5, 0.6, 0.3, 90, 6, 0.5, 0.5],
        'seuil_q': [0.25, 0.3, 0.15, 45, 3, 0.25, 0.25]},
}

# Fonction utilitaire pour le nettoyage des fichiers
def cleanup_decision_files():
    """Supprime les fichiers de décision et le fichier nearest_points_to_centroids.csv."""
    files_to_remove = [os.path.join(DATA_DIR, f"decision_final_decideur_{i}.csv") for i in range(1, 5)]
    files_to_remove.append(os.path.join(DATA_DIR, 'nearest_points_to_centroids.csv'))
    files_to_remove.append(os.path.join(DATA_DIR, 'decision_weights.csv'))
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Fichier supprimé : {file_path}")
            except OSError as e:
                print(f"Erreur lors de la suppression du fichier {file_path} : {e}")


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

    return render_template('index.html', 
                           decider_info=decider_info, 
                           can_compute_best_action=can_compute_best_action)

@app.route('/decideur', methods=['GET', 'POST'])
def decideur_page():
    """Gère la page d'analyse du Décideur, affichant le formulaire de saisie et les résultats."""
    results = None
    error = None
    kmeans_file_warning = None # Avertissement spécifique si nearest_points.csv est manquant
    
    # Récupère l'ID du décideur à partir des paramètres de requête pour le pré-remplissage, si disponible
    prefill_decideur_id = request.args.get('decideur_id', type=int)
    if prefill_decideur_id not in [1, 2, 3, 4]:
        prefill_decideur_id = 4 # Valeur par défaut si invalide ou non fournie

    # Définit les valeurs initiales pour les champs du formulaire (requête GET ou si POST échoue la validation)
    selected_preset = DECIDER_PRESET_PARAMS.get(prefill_decideur_id, DECIDER_PRESET_PARAMS[4]) # Par défaut, décideur 4
    # Convertit les listes en chaînes séparées par des virgules pour les champs de saisie HTML
    prefill_poids_str = ','.join(map(str, selected_preset['poids']))
    prefill_seuil_p_str = ','.join(map(str, selected_preset['seuil_p']))
    prefill_seuil_q_str = ','.join(map(str, selected_preset['seuil_q']))

    # Vérifie l'existence de nearest_points_to_centroids.csv lors d'une requête GET ou si POST ne fournit pas de nouvelles données
    nearest_points_path = os.path.join(DATA_DIR, 'nearest_points_to_centroids.csv')
    if not os.path.exists(nearest_points_path):
        kmeans_file_warning = "Le fichier 'nearest_points_to_centroids.csv' est introuvable. Veuillez exécuter l'analyse K-Means dans la page 'Analyse Négociateur' pour le générer si vous souhaitez utiliser l'option de chargement automatique."

    if request.method == 'POST':
        try:
            # Récupère l'ID du décideur à partir de l'entrée du formulaire (priorise le formulaire sur le paramètre de requête, met à jour le pré-remplissage pour l'affichage)
            decideur_id = int(request.form.get('decideur_id', prefill_decideur_id))
            prefill_decideur_id = decideur_id # Met à jour pour un rendu potentiel

            # Récupère la nouvelle matrice de données sous forme de chaîne brute (format CSV)
            new_data_matrix_str = request.form['new_data_matrix'].strip() # Supprime les espaces blancs

            # Récupère les poids et les seuils modifiables par l'utilisateur à partir du formulaire
            poids_input_str = request.form['poids_input'].strip()
            seuil_p_input_str = request.form['seuil_p_input'].strip()
            seuil_q_input_str = request.form['seuil_q_input'].strip()

            # Met à jour les valeurs de pré-remplissage en fonction de ce que l'utilisateur a saisi (pour le rendu si une erreur se produit)
            prefill_poids_str = poids_input_str
            prefill_seuil_p_str = seuil_p_input_str
            prefill_seuil_q_str = seuil_q_input_str
            
            # Appelle la fonction de logique de base du décideur avec les paramètres fournis par l'utilisateur
            results = run_decideur_analysis(
                decideur_id, 
                new_data_matrix_str, 
                poids_input_str, 
                seuil_p_input_str, 
                seuil_q_input_str
            )
            
            # Passe l'URL du plot pour l'affichage en HTML
            results['swot_plot_url'] = results['swot_plot_path']

        except FileNotFoundError as e: # Capture spécifiquement si nearest_points.csv est manquant
            error = f"{e}. Veuillez exécuter l'analyse K-Means dans la page 'Analyse Négociateur' pour générer ce fichier."
            print(f"Decideur analysis FileNotFoundError: {e}")
        except ValueError as e:
            error = f"Erreur de validation: {e}. Assurez-vous que tous les champs sont correctement remplis (nombres entiers, flottants, format CSV)."
            print(f"Decideur analysis ValueError: {e}")
        except Exception as e:
            # Capture toutes les autres erreurs inattendues pendant l'analyse
            error = f"Une erreur inattendue est survenue lors de l'analyse du décideur : {e}"
            print(f"Decideur analysis general error: {e}") # Journalise l'erreur pour le débogage

    # Rend la page du décideur, en passant les résultats, les erreurs, l'ID de pré-remplissage et les paramètres de pré-remplissage
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
    """Gère la page d'analyse du Négociateur, affichant les résultats K-Means et de négociation."""
    best_action_result = None
    kmeans_results = None
    error = None
    kmeans_data_path_used = None # Pour informer l'utilisateur quel fichier a été utilisé pour K-Means

    # Détermine quel fichier utiliser pour l'analyse K-Means
    uploaded_file_path = None
    try:
        # Vérifie si un fichier a été téléchargé via une requête POST
        if request.method == 'POST' and 'data_file' in request.files:
            file = request.files['data_file']
            if file and allowed_file(file.filename):
                # Crée un nom de fichier unique et sécurisé pour le fichier téléchargé
                filename = secure_filename(file.filename)
                unique_filename = str(uuid.uuid4()) + '_' + filename
                uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(uploaded_file_path)
                kmeans_data_path_used = uploaded_file_path # K-Means utilisera ce fichier
                print(f"Fichier téléchargé enregistré sous : {uploaded_file_path}")
            elif file and not allowed_file(file.filename):
                error = "Type de fichier non autorisé. Veuillez télécharger un fichier CSV."
        
        # Si aucun fichier n'est téléchargé ou s'il est invalide, essaie d'utiliser le fichier data_complet.csv par défaut
        if not kmeans_data_path_used:
            default_data_path = os.path.join(DATA_DIR, 'data_complet.csv')
            if os.path.exists(default_data_path):
                kmeans_data_path_used = default_data_path
            else:
                if not error: # Ne définit l'erreur que si elle n'est pas déjà définie par un problème de téléchargement de fichier
                    error = "Fichier 'data_complet.csv' non trouvé dans le dossier 'data/'. Veuillez le télécharger ou le placer dans le dossier 'data/'."

        # --- Analyse K-Means (s'exécute si un chemin de données valide est disponible) ---
        if kmeans_data_path_used:
            kmeans_results = run_kmeans_analysis(kmeans_data_path_used)
            # Ajoute l'URL du plot aux résultats pour l'affichage HTML
            if 'kmeans_plot_path' in kmeans_results:
                kmeans_results['kmeans_plot_url'] = '/static/images/' + os.path.basename(kmeans_results['kmeans_plot_path'])
        else:
            if not error:
                error = "Aucun fichier de données n'est disponible pour l'analyse K-Means."

    except Exception as e:
        error = f"Erreur lors de l'exécution de l'analyse K-Means : {e}"
        print(f"Erreur K-Means : {e}") # Journalise l'erreur pour le débogage
    finally:
        # Nettoie le fichier temporaire téléchargé après l'analyse
        if uploaded_file_path and os.path.exists(uploaded_file_path):
            try:
                os.remove(uploaded_file_path)
                print(f"Fichier temporaire nettoyé : {uploaded_file_path}")
            except OSError as e:
                print(f"Erreur lors de la suppression du fichier temporaire {uploaded_file_path} : {e}")

    if request.method == 'POST':
        try:
            # Récupère les poids individuels des quatre champs de saisie séparés
            decision_weights = {
                1: float(request.form.get('weight_decideur_1', 0)),
                2: float(request.form.get('weight_decideur_2', 0)),
                3: float(request.form.get('weight_decideur_3', 0)),
                4: float(request.form.get('weight_decideur_4', 0))
            }

            # S'assure qu'au moins un poids est fourni (et non nul)
            if not any(weight > 0 for weight in decision_weights.values()):
                raise ValueError("Veuillez fournir au moins un poids non nul pour un décideur pour lancer la négociation.")


            # --- NEW: Save decision_weights to a CSV file ---
            weights_file_path = os.path.join(DATA_DIR, 'decision_weights.csv')
            with open(weights_file_path, 'w', newline='') as csvfile:
                fieldnames = ['Decideur', 'Weight']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for decideur, weight in decision_weights.items():
                    writer.writerow({'Decideur': decideur, 'Weight': weight})
            # --- END NEW ---

            # Prépare la liste des chemins des fichiers de décision pour les décideurs pertinents
            files = [os.path.join(DATA_DIR, f"decision_final_decideur_{i}.csv")
                    for i in decision_weights.keys()]

            # Étape 1 : Compte le total des décisions acceptées à partir des fichiers pertinents
            accepted_count, accepted_data = count_total_accepted(files)

            # Étape 2 : Calcule la meilleure action en fonction des données acceptées et des poids
            best_action_result = compute_best_action(files, accepted_data, decision_weights)

        except ValueError as e:
            error = f"Erreur de validation des poids ou des fichiers de décision : {e}"
        except Exception as e:
            # Capture et journalise toute autre erreur inattendue pendant la négociation
            error = f"Une erreur inattendue est survenue lors de l'analyse du négociateur : {e}"
            print(f"Erreur Négociateur : {e}") # Journalise l'erreur pour le débogage

    # Rend la page du négociateur, en passant les résultats de la négociation, les résultats K-Means et toutes les erreurs
    return render_template('negociateur.html', 
                           best_action=best_action_result, 
                           kmeans_results=kmeans_results, 
                           error=error,
                           kmeans_data_path_used=os.path.basename(kmeans_data_path_used) if kmeans_data_path_used else 'N/A')


@app.route('/best_action')
def best_action_page():
    """Gère la page 'Meilleure action sélectionnée', effectue le calcul et le nettoyage."""
    best_action_result = None
    error = None
    
    try:
        # Define the path to the decision_weights.csv file
        weights_file_path = os.path.join(DATA_DIR, 'decision_weights.csv')
        decision_weights = load_decision_weights(weights_file_path)

        if decision_weights is None:
            error = "Les poids des décideurs n'ont pas pu être chargés. Veuillez vérifier le fichier 'decision_weights.csv'."
            print("Decision weights could not be loaded, returning early.")
            return render_template('best_action.html', best_action=best_action_result, error=error)
            
        # Construct file paths for decision files
        files = [os.path.join(DATA_DIR, f"decision_final_decideur_{i}.csv") for i in range(1, 5)]

        # Execute negotiation logic using the loaded weights
        accepted_count, accepted_data = count_total_accepted(files)
        
        if accepted_data.empty:
            error = "Aucune zone 'Accepté' trouvée dans les fichiers de décision. Veuillez vérifier le contenu des fichiers."
            print("No 'Accepté' zones found, returning early from best_action_page.")
        else:
            best_action_result = compute_best_action(files, accepted_data, decision_weights)

            # After successful calculation, perform file cleanup (uncomment if needed)
            cleanup_decision_files() # You would need to implement this function

    except Exception as e:
        error = f"Une erreur est survenue lors du calcul de la meilleure action : {e}"
        print(f"Erreur dans best_action_page : {e}")

    return render_template('best_action.html', best_action=best_action_result, error=error)

# --- Points d'API (inchangés) ---
@app.route('/api/decideur', methods=['POST'])
def api_decideur():
    """Point d'API pour l'analyse du Décideur (entrée/sortie JSON)."""
    try:
        data = request.get_json() # Attend une charge utile JSON
        decideur_id = data.get('decideur_id')
        new_data_matrix_str = data.get('new_data_matrix') # Attend une chaîne de caractères de type CSV
        poids_str = data.get('poids_input')
        seuil_p_str = data.get('seuil_p_input')
        seuil_q_str = data.get('seuil_q_input')

        if not decideur_id or not new_data_matrix_str or not poids_str or not seuil_p_str or not seuil_q_str:
            return jsonify({'error': 'Paramètres requis manquants pour l\'analyse du décideur'}), 400

        results = run_decideur_analysis(decideur_id, new_data_matrix_str, poids_str, seuil_p_str, seuil_q_str)
        return jsonify(results), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Erreur interne du serveur : {e}'}), 500

@app.route('/api/negociateur', methods=['POST'])
def api_negociateur():
    """Point d'API pour l'analyse du Négociateur (entrée/sortie JSON)."""
    try:
        data = request.get_json()
        decision_weights = data.get('decision_weights') # Attend un dictionnaire : {"1": 0.1, "2": 0.1, ...}

        if not decision_weights:
            return jsonify({'error': 'Poids de décision manquants'}), 400
        
        # Convertit les clés en entier si elles sont arrivées en tant que chaînes depuis JSON
        decision_weights_int_keys = {int(k): v for k, v in decision_weights.items()}

        files = [os.path.join(DATA_DIR, f"decision_final_decideur_{i}.csv") for i in decision_weights_int_keys.keys()]

        accepted_count, accepted_data = count_total_accepted(files)
        best_action_result = compute_best_action(files, accepted_data, decision_weights_int_keys)

        return jsonify(best_action_result), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Erreur interne du serveur : {e}'}), 500


if __name__ == '__main__':
    app.run(debug=True)