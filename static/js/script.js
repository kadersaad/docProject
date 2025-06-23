// This file would contain JavaScript if you wanted to make AJAX calls
// instead of traditional form submissions (e.g., using fetch API).
// For the provided HTML and Flask app, direct form submissions are used,
// so this file is optional unless you want more advanced client-side interactivity.

document.addEventListener('DOMContentLoaded', () => {
    // Example: If you had a button to fetch decider data without page reload
    const decideurForm = document.getElementById('decideurForm'); // Assuming you add an ID to your form

    if (decideurForm) {
        decideurForm.addEventListener('submit', async (event) => {
            event.preventDefault(); // Prevent default form submission

            const decideurId = document.getElementById('decideur_id').value;
            const newDataMatrix = document.getElementById('new_data_matrix').value;

            try {
                const response = await fetch('/api/decideur', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        decideur_id: parseInt(decideurId),
                        new_data_matrix: newDataMatrix
                    }),
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || 'Server error');
                }

                const result = await response.json();
                console.log('Decider API Result:', result);
                // Now, dynamically update your HTML with the `result` data
                // e.g., create new elements, update text content, display plot
                const resultsDiv = document.querySelector('.results');
                if (resultsDiv) {
                    resultsDiv.innerHTML = `
                        <h2>Résultats de l'Analyse Décideur ${result.decideur_id} (via API)</h2>
                        <h3>Plot SWOT :</h3>
                        <img src="${result.swot_plot_path}" alt="SWOT Plot" class="swot-plot">
                        <pre>${JSON.stringify(result.final_df_swot, null, 2)}</pre>
                        `;
                }

            } catch (error) {
                console.error('Error fetching decider data:', error);
                const errorDiv = document.querySelector('.error-message');
                if (errorDiv) {
                    errorDiv.innerHTML = `<p>Erreur: ${error.message}</p>`;
                } else {
                    alert(`Erreur: ${error.message}`);
                }
            }
        });
    }
});