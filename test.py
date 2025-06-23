import streamlit as st
import json
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

#import openai

def nettoyer_texte(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\s]", "", text)
    stopwords = {"de", "la", "le", "les", "des", "d'", "du", "et", "à", "en", "un", "une", "pour", "sur", "par", "avec"}
    mots = text.split()
    mots_filtrés = [m for m in mots if m not in stopwords]
    return " ".join(mots_filtrés)

def construire_corpus_boamp(records):
    corpus = []
    for rec in records:
        corpus.append({
            "titre": rec.get("title", "N/A"),
            "description": rec.get("description", "") or "",
            "procedure": rec.get("procedure_type", "N/A"),
            "organisme": rec.get("organism", "N/A") or "N/A",
            "lieu": rec.get("location", "N/A") or "N/A"
        })
    return corpus

def clustering_aos(corpus, n_clusters=3):
    documents = [nettoyer_texte(item["titre"] + " " + item["description"]) for item in corpus]
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
        textes_cluster = [nettoyer_texte(item["titre"]) for item in corpus if item.get("cluster") == cluster_id]
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

# Fonction LLM : désactivée pour limiter l’usage de quota
def generer_ao_via_llm(titre, procedure, contraintes):
    """
    # Décommenter et configurer ta clé OpenAI ici
    openai.api_key = "TA_CLE_API_ICI"
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
    st.title("Analyse des Appels d’Offres (BOAMP)")

    fichier = "boamp_sample.json"
    try:
        with open(fichier, encoding="utf-8") as f:
            records = json.load(f)
    except Exception as e:
        st.error(f"Erreur chargement fichier : {e}")
        return

    # Sidebar ajout AO
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
    for i, ao in enumerate(records):
        st.markdown(f"**{i+1}. {ao.get('title', 'N/A')}**")
        st.write(f"- Organisme : {ao.get('organism', 'N/A')}")
        st.write(f"- Date publication : {ao.get('publication_date', 'N/A')}")
        st.write(f"- Procédure : {ao.get('procedure_type', 'N/A')}")
        st.write(f"- Lieu : {ao.get('location', 'N/A')}")
        desc = ao.get('description') or ""
        if len(desc) > 200:
            st.write(f"- Description : {desc[:200]}... (clic pour voir plus)")
            if st.checkbox(f"Afficher description complète {i+1}", key=f"desc_{i}"):
                st.write(desc)
        else:
            st.write(f"- Description : {desc}")
        st.markdown("---")

    # Section optionnelle génération AO avec LLM 
    
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
