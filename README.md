# QSPHAgent (Qualitative Structure-to-Property Hypothesis)
<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" />

## Authors

- Ankita Biswas (https://github.com/Anny-tech)
- Collin Kovacs (https://github.com/ckovacs2)
- Huanhuan Zhao(https://github.com/HuanhuanZhao08)

- # QSPHAgent (Qualitative Structure-to-Property Hypothesis)

## 📌 Project Statement
For a given structure, the agent will generate a hypothesis explaining how certain structural features contribute to the specific characteristics of the material’s density of states (DOS).

---

## 💡 Idea
Given a composition and all corresponding materials entries from the Materials Project (with DOS data):

1. Generate textual descriptions from the structural and DOS data.  
2. Convert these descriptions into a vector database for retrieval-augmented generation (RAG).  
3. Use a reflection agent to hypothesize how and why particular structural characteristics are related to observed DOS features.  

---

## 🧪 Example

**User:**  
I have a material composition `<Al, O>` with structural features `<textual description of the structure>`. Hypothesize how these structural features might correlate with its possible DOS.  

**Agent:**  
“Based on the retrieved knowledge, here is a possible hypothesis on how the density of states may behave:  

The tetrahedral coordination around Al atoms and the short Al–O bond lengths are likely to result in a wide band gap and sharp oxygen p-orbital peaks below the Fermi level...”:contentReference[oaicite:0]{index=0}

---

## 🚀 Motivation
The DOS at the Fermi level critically influences key properties such as:
- Electrical conductivity  
- Magnetism  
- Superconductivity  

Rapid screening of DOS characteristics is essential for accelerating materials design and discovery.

---

