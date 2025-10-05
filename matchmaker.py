import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import subprocess
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Drug Repurposing Matchmaker", layout="wide")
# ---------------------------
# 1. Query LLM (local LLaMA-2 via Ollama)
# ---------------------------
def query_ollama(prompt, model="llama2"):
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            capture_output=True,
            check=True
        )
        return result.stdout.decode("utf-8").strip()
    except Exception as e:
        return f"(LLM unavailable) Mock narrative: {str(e)}"

# ---------------------------
# 2. Load Kaggle Drug Repositioning CSVs
# ---------------------------
@st.cache_data
def load_data():
    diseases = pd.read_csv("diseasesInfo.csv")  # DiseaseID, DiseaseName, etc.
    drugs = pd.read_csv("drugsInfo.csv")        # DrugID, DrugName, DrugTarget, etc.
    mapping = pd.read_csv("mapping.csv")        # DrugID, DiseaseID
    return diseases, drugs, mapping

diseases_df, drugs_df, mapping_df = load_data()

# ---------------------------
# 3. Build drug-target mapping
# ---------------------------
# Merge DrugTarget info from drugsInfo
mapping_df = mapping_df.merge(drugs_df[["DrugID", "DrugTarget", "DrugName"]], on="DrugID", how="left")

# drug_id -> set of targets
drug_to_targets = mapping_df.groupby("DrugID")["DrugTarget"].apply(lambda x: set([t for t in x if pd.notna(t)])).to_dict()
drug_id_to_name = drugs_df.set_index("DrugID")["DrugName"].to_dict()
disease_id_to_name = diseases_df.set_index("DiseaseID")["DiseaseName"].to_dict()

# Universe of targets
all_targets = sorted({t for targets in drug_to_targets.values() for t in targets})
drug_list = list(drug_to_targets.keys())

# Binary drug x target matrix
drug_gene_df = pd.DataFrame(0, index=drug_list, columns=all_targets, dtype=int)
for d, targets in drug_to_targets.items():
    for t in targets:
        if t in drug_gene_df.columns:
            drug_gene_df.at[d, t] = 1
drug_vectors = drug_gene_df.values

# ---------------------------
# 4. Similarity / scoring functions
# ---------------------------
def vector_for_targets(targets):
    return np.array([1 if t in targets else 0 for t in all_targets], dtype=int)

def score_similarity(vec1, vec2):
    sim = cosine_similarity(vec1.reshape(1,-1), vec2.reshape(1,-1))[0,0]
    return (sim + 1)/2  # normalize 0-1

def top_matches(disease_targets, top_k=5):
    disease_vec = vector_for_targets(disease_targets)
    scores = [(d, score_similarity(drug_vectors[i], disease_vec)) for i, d in enumerate(drug_list)]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores[:top_k]

# ---------------------------
# 5. Streamlit Dashboard
# ---------------------------

st.title("💊 AI Molecule Matchmaker")

st.markdown("""
This tool finds potential repurposing candidates for a disease from existing drugs using their targets.
""")

# Disease selection
disease_name = st.selectbox("Select a disease:", sorted(diseases_df["DiseaseName"].tolist()))
disease_id = diseases_df[diseases_df["DiseaseName"] == disease_name]["DiseaseID"].values[0]

# Targets for this disease
disease_targets = mapping_df[mapping_df["DiseaseID"] == disease_id]["DrugTarget"].dropna().unique().tolist()

if len(disease_targets) == 0:
    st.warning("No known targets for this disease. Showing top random drugs instead.")
    results_df = drugs_df.sample(5)
    results_df["score"] = np.random.rand(len(results_df))
else:
    # Compute top matches
    matches = top_matches(disease_targets, top_k=5)
    results_df = pd.DataFrame({
        "drug_name": [drug_id_to_name[d[0]] for d in matches],
        "score": [d[1] for d in matches]
    })

# Show top matches
st.subheader(f"Top candidate drugs for **{disease_name}**")
st.table(results_df)

# Simple Plotly scatter for demo
st.subheader("Drug Compatibility Map (mock layout)")
np.random.seed(42)
results_df["x"] = np.random.rand(len(results_df))
results_df["y"] = np.random.rand(len(results_df))
fig = px.scatter(results_df, x="x", y="y", text=results_df.get("drug_name", results_df.get("DrugName","")),
                 color="score", size="score", size_max=20, title="Top Drug Matches")
st.plotly_chart(fig, use_container_width=True)

# Generate LLM narratives
# st.subheader("LLM Narrative for Each Drug")
# for d in results_df.get("drug_name", results_df.get("DrugName","")):
#     prompt = f"Explain in simple terms why {d} could be repurposed for {disease_name}."
#     narrative = query_ollama(prompt)
#     with st.expander(f"{d}"):
#         st.info(narrative)

@st.cache_data
def get_narrative(drug, disease):
    prompt = f"Explain in simple terms why {drug} could be repurposed for {disease}."
    return query_ollama(prompt)

st.subheader("LLM Narrative for Each Drug")
for d in results_df.get("drug_name", results_df.get("DrugName","")):
    narrative = get_narrative(d, disease_name)
    with st.expander(f"{d}"):
        st.info(narrative)

st.caption("Data from Kaggle Drug Repositioning dataset. Narratives generated by local LLaMA-2 (Ollama).")
