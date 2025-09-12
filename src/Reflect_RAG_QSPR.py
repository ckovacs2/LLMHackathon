#!/usr/bin/env python
# coding: utf-8

# In[850]:


import json
import random
import numpy as np
import os
from pydantic import BaseModel, Field
from typing import List, Optional, TypedDict
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from utils import SimpleDOSPredictor
from database_genrator import generate_database
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from utils import  DOSHypothesis,dos_hypothesis_to_description,DOSCritique
from langgraph.graph import StateGraph,END



# In[852]:


os.environ["OPENAI_API_KEY"] ="your API key"
MP_API_KEY = "your API key"
species  = ["Si"]

# # Data generation

# In[ ]:





# In[860]:


DATAFILE = "materials_database.json"

generate_database(MP_API_KEY, 
                  species = species,  
                  cutoff=5.0,  #### for structural features
                  output_file=DATAFILE )


# # Data splitting and retrieval for RAG

# In[862]:


def structural_vector(struct_features):
    numeric_keys = [
        "space_group_number", "volume_per_atom", "density",
        "valence_electron_count", "avg_coordination_number",
        "mean_bond_length", "bond_length_std",
        "electronegativity_mean", "electronegativity_difference"
    ]
    return np.array([struct_features.get(k, 0.0) or 0.0 for k in numeric_keys], dtype=float)

def retrieve_similar_structures(test_entry, train_data, top_k=5):
    test_vec = structural_vector(test_entry['structure']["structural_features"])
    distances = []
    for uid, entry in train_data.items():
        train_vec = structural_vector(entry['structure']["structural_features"])
        dist = np.linalg.norm(test_vec - train_vec)
        distances.append((uid, dist))
    distances.sort(key=lambda x: x[1])
    return [(uid, train_data[uid]) for uid, _ in distances[:top_k]]


def generate_text_summary(structure_entry):
    """
    Generate a concise human-readable text summary from structural features.
    """
    features = structure_entry["structural_features"]
    formula = structure_entry.get("formula", "Unknown")
    sg_symbol = features.get("space_group_symbol", "Unknown")
    crystal_system = features.get("crystal_system", "Unknown").capitalize()
    avg_cn = features.get("avg_coordination_number")
    mean_bond_length = features.get("mean_bond_length")

    summary_parts = [f"{crystal_system} {formula} ({sg_symbol})"]

    if avg_cn is not None:
        summary_parts.append(f"avg coordination ≈ {avg_cn:.1f}")
    if mean_bond_length is not None:
        summary_parts.append(f"mean bond length ≈ {mean_bond_length:.2f} Å")

    return ", ".join(summary_parts) + "."



def build_context_from_neighbors(test_entry, neighbors):
    # Generate summary for query material

    lines = []
    
    
    lines.append("Top structurally similar materials:")

    for uid, entry in neighbors:
        neighbor_summary = generate_text_summary(entry['structure'])
        lines.append(f"- {entry['structure']['formula']}: {neighbor_summary}")
        if "dos" in entry and "dos_description" in entry["dos"]:
            lines.append(f"  DOS: {entry['dos']['dos_description']}")

    return "\n".join(lines)


# ### ========================RAG set and test set

# In[865]:


import random

with open(DATAFILE, "r") as infile:
    data = json.load(infile)

random.seed(786)
keys = list(data.keys())
random.shuffle(keys)
split_idx = int(0.8 * len(keys))
train_set = {k: data[k] for k in keys[:split_idx]}
test_set  = {k: data[k] for k in keys[split_idx:]}

print(f"Train size: {len(train_set)}, Test size: {len(test_set)}")




test_entry = test_set[list(test_set.keys())[1]]  #### change the test entry here <<<<<<-------
neighbors = retrieve_similar_structures(test_entry, train_set, top_k=7)
context = build_context_from_neighbors(test_entry, neighbors)



# ##  Initial Hypothesis Generator Agent

# In[868]:


from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

crystal_dos_hypothesis_agent = create_react_agent(
    model="openai:gpt-4o",
    name="crystal_dos_hypothesis_agent",
    tools=[],
    response_format=DOSHypothesis,
    prompt = """
You are an expert materials scientist generating a **qualitative hypothesis**
about the **electronic density of states (DOS)** of a query material.

You are given:
- A brief structural summary of the query material (formula, space group, bonding environment)
- A list of structurally similar materials with textual DOS descriptions or visual DOS trends

---

Your task is to infer the **key DOS features** for the query material and return them
in the structured format `DOSHypothesis` (see schema below).

Fill in **only the following fields**:

---

**Required Fields**

- `material_classification`: One of `"metallic"`, `"semiconducting"`, `"insulating"`, or `"uncertain"`  
  → This is based on whether there is a clear gap or finite DOS at the Fermi level.

- `overall_dos_shape`: General shape of the DOS across valence and conduction bands,  
  → Examples: `"U-shaped"`, `"flat"`, `"asymmetric"`, `"pseudogap-like"`, etc.

- `asymmetry_comment`: Optional comment if the DOS shows clear imbalance across EF  
  → Examples: `"higher DOS in conduction band"`, `"valence states dominate"`

- `valence_band_peaks`: Dictionary with:  
  {
    "main_peak_energy": float,  
    "main_peak_height": float,
    "other_peaks": List[float]
  }

- `conduction_band_peaks`: Dictionary with:  
  {
    "main_peak_energy": float,  
    "main_peak_height": float,
    "other_peaks": List[float]
  }

- `pseudogap_score`: A float between 0.0 and 1.0 representing suppression near EF  
  → 0 = flat metallic DOS, 1 = deep full gap, ~0.5 = partial dip

---

**Optional Fields**

- `fermi_level_dos`: One of `"high"`, `"moderate"`, `"low"`, or `"zero"`  
  → Estimate based on DOS at EF (qualitatively)

- `band_gap_range_eV`: If a gap exists, give an approximate string value  
  → Examples: `"~1.0–1.5"`, `"None"` if metallic

---

**New Required Field**

- `reasoning`: A dictionary mapping each of the above fields to **short, clear justifications**.  
  → The keys should match exactly: `material_classification`, `overall_dos_shape`, etc.  
  → Each value should explain why that field was assigned based on context or analogies.

Example:
"reasoning": {
  "material_classification": "Gap between valence and conduction bands",
  "valence_band_peaks": "Sharp peak around –4.8 eV consistent with similar oxides",
  "pseudogap_score": "DOS is suppressed at EF but not fully zero — consistent with score ~0.8"
}

---

**Guidelines**

- Only include numerical values if they can be confidently inferred from trends
- You may use analogies (e.g., "like MoS₂" or "similar to perovskites")
- Avoid contradictions: e.g., don't call a material "metallic" and assign `"zero"` fermi DOS
- Leave a field blank if not inferable, but explain this in `reasoning`

---

**Example Output (JSON)**

{
  "material_classification": "metallic",
  "overall_dos_shape": "V-shaped suppression near EF.",
  "asymmetry_comment": null,
  "valence_band_peaks": {
    "main_peak_energy": -4.8,
    "main_peak_height": 11.6,
    "other_peaks": [-5.7, -5.5]
  },
  "conduction_band_peaks": {
    "main_peak_energy": 2.2,
    "main_peak_height": 11.1,
    "other_peaks": [3.2, 3.8]
  },
  "pseudogap_score": 0.86,
  "fermi_level_dos": "moderate",
  "band_gap_range_eV": "None",
  "reasoning": {
    "material_classification": "Finite DOS at EF with no clear gap → metallic",
    "overall_dos_shape": "Valley-shaped dip centered at EF → V-shaped",
    "valence_band_peaks": "Main peak at –4.8 eV dominates with nearby smaller features",
    "conduction_band_peaks": "Sharp peak near 2.2 eV similar to neighbors",
    "pseudogap_score": "Moderate dip at EF without full gap → ~0.86",
    "fermi_level_dos": "EF is not flat, has moderate DOS",
    "band_gap_range_eV": "No full band gap observed — metallic behavior"
  }
}
""".strip()
)



# ##  Critique agent



# ------ model --------
data = [value for value in train_set.values()] 
predictor = SimpleDOSPredictor(n_neighbors=6)
predictor.fit(data)
test_structure = test_entry["structure"]  # or new entry



# In[887]:


dos_critique_agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[],
    response_format=DOSCritique,
    name="dos_critique_agent",
    prompt= """
You are a **Density of States (DOS) critique agent**.

You are given:
- Structural features of a crystal.
- A DOS hypothesis (including classification, DOS shape, pseudogap score, valence/conduction peaks).
- The **quantitative prediction** results already generated via the `predict_dos_from_structure` tool.

Your steps:
1. Compare the provided quantitative prediction with the assumed DOS hypothesis.
2. Identify inconsistencies (e.g., metallic label but high pseudogap, or peak positions that don't match).
3. Suggest corrections and improvements to the hypothesis.
4. Return a structured JSON critique.

**Output Format (Strict)**

Your final response **must** be returned in JSON form, exactly like this:

```json
{
  "key_disagreements": ["<list of inconsistencies>"],
  "suggestions": ["<suggestions to improve or correct the hypothesis>"],
  "summary": "<1–2 sentence summary>"
}
Do not include any explanation outside of the JSON object. If no disagreements are found, still return the JSON object with empty lists and a summary confirming consistency.
""".strip()
)




# In[ ]:





# ### Reflection

# In[891]:


def generation_node(state: dict) -> dict:
    # Build user message as per your exact format
    user_message = {
        "role": "user",
        "content": f"""
**Query material**:
{state['structure']['structure']['formula']} → {generate_text_summary(state['structure']['structure'])}

**Context from structurally similar materials**:
{state['context']}

---

**Previous Hypothesis**:
{state['prev_hypothesis'] if state.get('prev_hypothesis') else "[None — this is the first hypothesis]"}

**Critique Feedback**:
{state['last_critique'] if state.get('last_critique') else "[None — no critique yet]"}

---

**Your task**:
Generate a new or improved qualitative hypothesis about the **density of states (DOS)** of the query material.

Instructions:
- Use the structural summary and contextual analogies to predict key DOS features.
- If critique is present, revise the hypothesis to address inconsistencies or improve physical plausibility.
- Always include detailed reasoning in the `"reasoning"` field for each prediction.

---

**Output Format**:
Return the full hypothesis in JSON format using this wrapper:

```json
{{
  "hypothesis_text": "<your stringified DOSHypothesis here>"
}}
"""
    }

    # Call the LLM agent with messages pattern
    response = crystal_dos_hypothesis_agent.invoke({
        "messages": [user_message]
    })

    # Get stringified hypothesis
    parsed_hypothesis = dos_hypothesis_to_description(response["structured_response"])

    # Update state with new hypothesis
    state["prev_hypothesis"] = parsed_hypothesis
    state["iteration"] = state.get("iteration", 0) + 1

    return state


# In[893]:


def reflection_node(state: dict) -> dict:
    # Get required inputs from state
    test_structure = state["structure"]["structure"]
    prev_hypothesis = state.get("prev_hypothesis")
    predictor = state["predictor"]

    # If no previous hypothesis, skip critique
    if prev_hypothesis=="":
        state["last_critique"] = "[None — no hypothesis to critique yet]"
        return state

    # Convert hypothesis string → dict (assumed format is structured string)
    assumed_hypothesis = prev_hypothesis
    quantitative_results = predictor.predict(test_structure)

    # Build the user message to the critique agent
    user_message = {
        "role": "user",
        "content": f"""```json
**Assumed DOS Hypothesis by the Generator Agent**:
{json.dumps(assumed_hypothesis)}

**Quantitative Results from Correlation Analysis**:
{json.dumps(quantitative_results)}

**Your Task**:
Use the quantitative results from the correlation analysis of the DOS based on the training dataset.
Then, compare this quantitative prediction of the DOS to the assumed hypothesis and provide a structured critique.

Focus on:
- Physical coherence (e.g., coordination, symmetry)
- Agreement with expected metallic/semiconducting behavior
- Validity of pseudogap and peak shape assumptions

Be specific, and provide reasoning tied to the structure–DOS relationship.
```"""
    }

    # Run the critique agent
    response = dos_critique_agent.invoke({
        "messages": [user_message]
    })

    # Save critique output
    state["last_critique"] = response["structured_response"]
    return state


# ### Run

# In[896]:


# Initial state — at iteration 0
state = {
    "structure": test_entry,                            # full structure dict
    "context": build_context_from_neighbors(test_entry, neighbors),
    "prev_hypothesis": "",                              # will be filled after first generation
    "last_critique": "",                                # will be filled after first critique
    "iteration": 0,
    "predictor":predictor ,
    
}


# In[898]:


builder = StateGraph(dict)  # <--- use StateGraph instead of MessageGraph
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.set_entry_point("generate")

def should_continue(state: dict):
    return "reflect" if state.get("iteration", 0) < 2 else END  # or any stopping condition

builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")

graph = builder.compile()


# In[900]:


final_state = graph.invoke(state )


# #### Target structure

# In[902]:

from tabulate import tabulate

# === Extract from final_state ===
structure_summary = generate_text_summary(final_state['structure']['structure'])
original_dos_dict = final_state['structure']['dos']['dos_description_dict']
hypothesis = final_state["prev_hypothesis"]

# === 1. Target Structure Summary ===
print("\n📌 Target Structure Summary\n")
print(tabulate([[structure_summary]], headers=["Structure Info"], tablefmt="fancy_grid"))

# === 2. Original DOS Description (Dict) ===
print("\n📊 Original DOS Description (Dict)\n")
dos_main = [
    ["Material Classification", original_dos_dict.get("material_classification")],
    ["Overall DOS Shape", original_dos_dict.get("overall_dos_shape")],
    ["Asymmetry Comment", original_dos_dict.get("asymmetry_comment") or "—"],
    ["Pseudogap Score", f"{original_dos_dict.get('pseudogap_score'):.3f}"],
]

vb_peaks = original_dos_dict.get("valence_band_peaks", {})
cb_peaks = original_dos_dict.get("conduction_band_peaks", {})

dos_vb = [
    ["Valence Peak (Main Energy)", f"{vb_peaks.get('main_peak_energy', '—'):.2f} eV"],
    ["Valence Peak (Height)", f"{vb_peaks.get('main_peak_height', '—'):.2f}"],
    ["Valence Other Peaks", ", ".join(f"{x:.2f}" for x in vb_peaks.get("other_peaks", [])) or "—"]
]

dos_cb = [
    ["Conduction Peak (Main Energy)", f"{cb_peaks.get('main_peak_energy', '—'):.2f} eV"],
    ["Conduction Peak (Height)", f"{cb_peaks.get('main_peak_height', '—'):.2f}"],
    ["Conduction Other Peaks", ", ".join(f"{x:.2f}" for x in cb_peaks.get("other_peaks", [])) or "—"]
]

# Combine tables
dos_table = dos_main + dos_vb + dos_cb
print(tabulate(dos_table, headers=["Feature", "Value"], tablefmt="fancy_grid"))

# === 3. Hypothesized DOS Description (Text) ===
print("\n🧠 Hypothesized DOS Description (Text)\n")
print(hypothesis)



# In[ ]:




