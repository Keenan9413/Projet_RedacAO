# Assistant de Rédaction d’Appels d’Offres (AO)

Ce projet propose un assistant intelligent pour analyser et générer automatiquement des appels d’offres, en s’appuyant sur un corpus existant. L'application permet de visualiser, structurer, et enrichir les AO grâce au NLP et au clustering, tout en offrant la possibilité de générer de nouveaux documents via un LLM.

##  Objectifs

- Analyser automatiquement un corpus d’appels d’offres (AO)
- Extraire les sections typiques : *objectif*, *prestations attendues*, *critères de sélection*, etc.
- Regrouper les AO similaires (clustering)
- Générer de nouveaux AO via un modèle de langage (LLM)
- Interface utilisateur interactive avec Streamlit

##  Tech Stack

- **Langage :** Python
- **NLP :** spaCy (`fr_core_news_sm`)
- **ML :** scikit-learn (TF-IDF, KMeans, PCA)
- **Frontend :** Streamlit + Plotly
- **LLM (optionnel) :** OpenAI GPT-4 via API

## Démarrage rapide

1. **Cloner le dépôt**

```bash
git clone https://github.com/<ton_utilisateur>/ao-assistant.git
cd ao-assistant
