# ── Load libraries ────────────────────────────────────────────
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns





# ── Load your data ────────────────────────────────────────────
df = pd.read_excel("engagement-anova.xlsx", sheet_name="Sheet0")

# Check it loaded
#print(df.columns.tolist())
#print(df.shape)
#print(df[["Condition", "Open_End"]].head())


 
impelling_goal = "to re-establish relationship and trust between parent and child."

hidden_goal = ("You are preparing for a divorce and wants the child to feel "
               "safe opening up to her, partly because you believe that "
               "emotional closeness may make the child more likely to choose "
               "her in the custody process.")

 
model = SentenceTransformer("all-mpnet-base-v2")

print("Model loaded successfully")



# ── Filter to B and C only ────────────────────────────────────
df_BC = df[df["Condition"].isin(["A", "C"])].copy()
df_BC = df_BC.dropna(subset=["Open_End"])  
df_BC = df_BC[df_BC["Open_End"].str.strip() != ""]   


 
impelling_embedding = model.encode(impelling_goal,    convert_to_tensor=True)
hidden_embedding    = model.encode(hidden_goal,       convert_to_tensor=True)

 
response_embeddings = model.encode(
    df_BC["Open_End"].tolist(), 
    convert_to_tensor=True,
    show_progress_bar=True
)

# ── Compute cosine similarity scores ─────────────────────────
# Score ranges from 0 (unrelated) to 1 (identical meaning)

impelling_scores = []
hidden_scores    = []

for i, emb in enumerate(response_embeddings):
    imp_sim = util.cos_sim(emb, impelling_embedding).item()
    hid_sim = util.cos_sim(emb, hidden_embedding).item()
    impelling_scores.append(imp_sim)
    hidden_scores.append(hid_sim)

 
df_BC["impelling_similarity"] = impelling_scores
df_BC["hidden_similarity"]    = hidden_scores

 
print(df_BC[["Condition", "Open_End", 
             "impelling_similarity", 
             "hidden_similarity"]].head(10))


# ── Descriptives by condition ─────────────────────────────────
print("\n=== Impelling Goal Similarity ===")
print(df_BC.groupby("Condition")["impelling_similarity"].agg(
    n="count",
    mean=lambda x: round(x.mean(), 3),
    sd=lambda x: round(x.std(), 3),
    median=lambda x: round(x.median(), 3)
))

print("\n=== Hidden Goal Similarity ===")
print(df_BC.groupby("Condition")["hidden_similarity"].agg(
    n="count",
    mean=lambda x: round(x.mean(), 3),
    sd=lambda x: round(x.std(), 3),
    median=lambda x: round(x.median(), 3)
))



# ── Independent t-tests ───────────────────────────────────────
B = df_BC[df_BC["Condition"] == "B"]
C = df_BC[df_BC["Condition"] == "C"]

print("\n=== T-test: Impelling Goal Similarity (B vs C) ===")
t_imp, p_imp = stats.ttest_ind(
    B["impelling_similarity"], 
    C["impelling_similarity"],
    equal_var=True
)
print(f"t = {t_imp:.3f}, p = {p_imp:.4f}")

print("\n=== T-test: Hidden Goal Similarity (B vs C) ===")
t_hid, p_hid = stats.ttest_ind(
    B["hidden_similarity"], 
    C["hidden_similarity"],
    equal_var=True
)
print(f"t = {t_hid:.3f}, p = {p_hid:.4f}")