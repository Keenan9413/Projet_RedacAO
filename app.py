import streamlit as st
import json
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import spacy

# Charger le modèle spaCy français
nlp = spacy.load("fr_core_news_sm")

# Nettoyage et lemmatisation du texte avec spaCy
def nettoyer_texte_spacy(text):
    if not text:
        return ""
    doc = nlp(text.lower())
    stopwords = spacy.lang.fr.stop_words.STOP_WORDS
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in stopwords]
    return " ".join(tokens)

# Extraction automatique de sections depuis la description (basé sur regex simple)
def extraire_sections(description):
    if not description:
        return {}

    sections = {}
    pattern = re.compile(
        r"(objectif[s]?|prestations attendues|critères de sélection|contraintes|description|contexte|modalités|conditions)[\s:\n]+", 
        re.IGNORECASE
    )
    # Trouver les positions des titres dans la description
    matches = list(pattern.finditer(description))
    if not matches:
        sections["description"] = description.strip()
        return sections
    
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i+1].start() if i + 1 < len(matches) else len(description)
        titre = match.group(1).lower()
        contenu = description[start:end].strip()
        sections[titre] = contenu
    
    return sections

def construire_corpus_boamp(records):
    corpus = []
    for rec in records:
        desc = rec.get("description", "") or ""
        sections = extraire_sections(desc)
        corpus.append({
            "titre": rec.get("title", "N/A"),
            "procedure": rec.get("procedure_type", "N/A"),
            "organisme": rec.get("organism", "N/A") or "N/A",
            "lieu": rec.get("location", "N/A") or "N/A",
            "description": desc,
            "sections": sections,
        })
    return corpus

# Clustering sur texte enrichi (titre + sections importantes)
def clustering_aos(corpus, n_clusters=3):
    documents = []
    for item in corpus:
        texte = item["titre"].lower()
        
        for key in ["objectif", "prestations attendues", "critères de sélection"]:
            texte += " " + item["sections"].get(key, "")
        documents.append(nettoyer_texte_spacy(texte))
    
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(documents)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_

    for i, label in enumerate(labels):
        corpus[i]["cluster"] = label

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X.toarray())

    return corpus, X_reduced, labels

def afficher_mots_cles_par_cluster(corpus, n_clusters):
    mots_cles = {}
    for cluster_id in range(n_clusters):
        textes_cluster = []
        for item in corpus:
            if item.get("cluster") == cluster_id:
                # On combine titre + sections pour plus de richesse
                txt = item["titre"].lower() + " "
                txt += " ".join(item["sections"].values())
                textes_cluster.append(nettoyer_texte_spacy(txt))
        mots = []
        for texte in textes_cluster:
            mots.extend(texte.split())
        counter = Counter(mots)
        mots_cles[cluster_id] = counter.most_common(10)
    return mots_cles

@st.cache_data
def generer_figure(corpus, X_reduced):
    import pandas as pd
    df = pd.DataFrame({
        "PC1": X_reduced[:,0],
        "PC2": X_reduced[:,1],
        "cluster": [item["cluster"] for item in corpus],
        "titre": [item["titre"] for item in corpus]
    })

    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="cluster",
        hover_name="titre",
        width=900,
        height=600,
        title="Clustering des appels d’offres"
    )
    return fig

# Fonction LLM désactivée par défaut
def generer_ao_via_llm(titre, procedure, contraintes):
    """
    # Pour activer la génération LLM, décommenter et insérer la clé API
    import openai
    openai.api_key = "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
"
    prompt = f"Génère un appel d'offre avec le titre '{titre}', procédure '{procedure}' et contraintes : {contraintes}."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role":"user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Erreur génération LLM : {e}"
    """
    return "[Génération LLM désactivée - Insérer votre clé OpenAI]"

def main():
    st.title("Assistant d’analyse et génération d’Appels d’Offres (BOAMP)")

    fichier = "boamp_sample.json"
    try:
        with open(fichier, encoding="utf-8") as f:
            records = json.load(f)
    except Exception as e:
        st.error(f"Erreur chargement fichier : {e}")
        return

    
    st.sidebar.header("Ajouter un nouvel appel d’offre")
    with st.sidebar.form("form_nouvel_ao", clear_on_submit=True):
        titre = st.text_input("Titre")
        procedure = st.text_input("Type de procédure")
        date_pub = st.text_input("Date publication (YYYY-MM-DD)")
        organisme = st.text_input("Organisme")
        lieu = st.text_input("Lieu")
        description = st.text_area("Description")
        submit = st.form_submit_button("Ajouter")

        if submit:
            if not titre:
                st.sidebar.warning("Le titre est obligatoire.")
            else:
                nouvel_id = f"id_{len(records) + 1}"
                nouvel_ao = {
                    "id": nouvel_id,
                    "title": titre,
                    "publication_date": date_pub or None,
                    "procedure_type": procedure,
                    "organism": organisme,
                    "location": lieu,
                    "description": description or ""
                }
                records.append(nouvel_ao)
                with open(fichier, "w", encoding="utf-8") as f:
                    json.dump(records, f, indent=2, ensure_ascii=False)
                st.sidebar.success("Nouvel appel d’offre ajouté avec succès !")

    corpus = construire_corpus_boamp(records)

    st.header("Clustering des appels d’offre")
    n_clusters = st.slider("Nombre de clusters", 2, 10, 3)

    corpus_cluster, X_reduced, labels = clustering_aos(corpus, n_clusters=n_clusters)

    mots_cles = afficher_mots_cles_par_cluster(corpus_cluster, n_clusters)
    st.subheader("Mots-clés par cluster")
    for cluster_id in range(n_clusters):
        mots = mots_cles.get(cluster_id, [])
        st.write(f"Cluster {cluster_id} : " + ", ".join([f"{mot} ({freq})" for mot, freq in mots]))

    fig = generer_figure(corpus_cluster, X_reduced)
    st.plotly_chart(fig, use_container_width=True)

    st.header("Liste des appels d’offre")
    for i, ao in enumerate(corpus_cluster):
        st.markdown(f"**{i+1}. {ao.get('titre', 'N/A')}**")
        st.write(f"- Organisme : {ao.get('organisme', 'N/A')}")
        st.write(f"- Procédure : {ao.get('procedure', 'N/A')}")
        desc = ao.get('description') or ""
        if len(desc) > 200:
            st.write(f"- Description : {desc[:200]}... (clic pour voir plus)")
            if st.checkbox(f"Afficher description complète {i+1}", key=f"desc_{i}"):
                st.write(desc)
        else:
            st.write(f"- Description : {desc}")

        st.write("**Sections extraites :**")
        for sec, contenu in ao.get("sections", {}).items():
            if contenu.strip():
                st.write(f"- {sec.capitalize()} : {contenu[:250]}{'...' if len(contenu) > 250 else ''}")
        st.markdown("---")

    # Section génération AO via LLM
    st.header("Génération d’un appel d’offre (LLM)")
    with st.form("form_generation_llm"):
        titre_gen = st.text_input("Titre pour génération")
        procedure_gen = st.text_input("Procédure")
        contraintes_gen = st.text_area("Contraintes spécifiques")
        submit_gen = st.form_submit_button("Générer")

        if submit_gen:
            resultat = generer_ao_via_llm(titre_gen, procedure_gen, contraintes_gen)
            st.text_area("Résultat génération", resultat, height=300)

if __name__ == "__main__":
    main()
