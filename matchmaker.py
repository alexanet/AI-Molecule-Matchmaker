import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

st.set_page_config(page_title="Drug Repurposing Matchmaker", layout="wide")

# ---------------------------
# 1. Query LLM using LangChain Ollama
# ---------------------------
def query_ollama(prompt, model="llama2"):
    try:
        # Initialize Ollama via LangChain
        llm = Ollama(
            model=model,
            base_url="http://localhost:11434",  # Local Ollama instance
            temperature=0.3,
            num_predict=500  # Limit response length
        )
        
        # Create a prompt template for better responses
        prompt_template = PromptTemplate(
            input_variables=["query"],
            template="""You are a biomedical expert specializing in drug repurposing. 
            Provide a clear, concise explanation about drug repositioning opportunities.
            
            Question: {query}
            
            Please explain in simple terms why this drug might work for this disease, 
            focusing on biological mechanisms and clinical potential.
            
            Answer:"""
        )
        
        # Create and run the chain
        chain = LLMChain(llm=llm, prompt=prompt_template)
        response = chain.run(query=prompt)
        
        return response.strip()
        
    except Exception as e:
        return f"(LLM unavailable) Mock narrative: Unable to connect to Ollama. Please ensure Ollama is running locally with 'ollama serve'"

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
**Now powered by LangChain + Ollama for intelligent explanations.**
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

# Generate LLM narratives with caching
@st.cache_data(show_spinner="Generating AI explanations...")
def get_narrative(drug, disease):
    prompt = f"Explain in simple terms why {drug} could be repurposed for {disease}."
    return query_ollama(prompt)

st.subheader("AI-Powered Drug Repurposing Explanations")
st.info("💡 Using LangChain with local Ollama for intelligent biomedical explanations")

for idx, row in results_df.iterrows():
    drug_name = row["drug_name"]
    score = row["score"]
    
    with st.expander(f"🧬 {drug_name} (Match Score: {score:.2f})"):
        with st.spinner(f"Generating explanation for {drug_name}..."):
            narrative = get_narrative(drug_name, disease_name)
            st.write(narrative)
            
            # Add some visual separation
            st.markdown("---")
            st.caption(f"*AI explanation generated for {drug_name} → {disease_name}*")

# Sidebar with Ollama status
with st.sidebar:
    st.header("🔧 Configuration")
    st.info("""
    **LangChain + Ollama Setup:**
    - Using local Ollama instance
    - LangChain for prompt management
    - Better error handling
    - Improved response quality
    """)
    
    # Model selection
    model_option = st.selectbox(
        "Ollama Model:",
        ["llama2", "llama2:13b", "medllama2", "codellama"],
        help="Select which Ollama model to use for explanations"
    )
    
    if st.button("🔄 Test Ollama Connection"):
        with st.spinner("Testing connection..."):
            test_response = query_ollama("Say 'Hello' in one word.", model=model_option)
            if "unavailable" in test_response.lower():
                st.error("❌ Ollama not connected")
                st.code("Run: ollama serve", language="bash")
            else:
                st.success("✅ Ollama connected successfully")
                st.code(f"Response: {test_response}")

st.caption("Data from Kaggle Drug Repositioning dataset. Narratives generated by LangChain + Ollama.")