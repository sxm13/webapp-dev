# CoRE MOF Database Web APP

<div align="center">
  <img src="https://raw.githubusercontent.com/sxm13/pypi-dev/main/logos/coremof.png" alt="CoRE MOF logo" width="500"/>

  <h3>Computation-Ready, Experimental Metal-Organic Framework Database</h3>

  [![Online Website](https://img.shields.io/badge/Maintained%20by-Pusan%20National%20University-blue)](https://mof-db.pusan.ac.kr)
  [![Database Version](https://img.shields.io/badge/Database-CoRE%20MOF%202024%2F2025-green)](https://mof-db.pusan.ac.kr)
  [![Platform](https://img.shields.io/badge/Platform-Web--based-orange)](https://mof-db.pusan.ac.kr)

  [**🌐 Visit the Online Database**](https://mof-db.pusan.ac.kr)
</div>

---

## 📖 Introduction

The **CoRE MOF (Computation-Ready, Experimental Metal-Organic Framework)** Database is an open-access digital repository developed by **Pusan National University**. It addresses a critical bottleneck in molecular simulation: the transition from raw experimental crystallographic data (CIFs) to "computation-ready" structures. 

As of the **2025 update**, the platform hosts over **43,000+** MOF structures, meticulously curated to remove solvents, balance charges, and fix structural inconsistencies, while providing machine-learning (ML) predicted physicochemical properties.

---

## ✨ Key Features

### 1. Advanced Structural Curation
* **Solvent Removal:** Provides both **ASR** (All Solvent Removed) and **FSR** (Free Solvent Removed) versions.
* **NCR Diagnostics:** Integrated **MOSAEC** and **MOFChecker** tools to identify "Not Computation-Ready" structures (e.g., overlapping atoms or missing coordinates).
* **Charge Assignment:** Pre-calculated **DDEC06** partial atomic charges using ML models for rapid electrostatic potential modeling.

### 2. Multi-dimensional Search & Discovery
* **Text & ID Search:** Query by MOF name, DOI, Publication Year, or **MOFid (v1/v2)**.
* **Numerical Filtering:** Screen frameworks based on calculated geometric properties (e.g., PLD, LCD, surface area).
* **Stability Categorization:** Access ML-predicted thermal stability (decomposition temperature, $T_d$).

### 3. Integrated Analysis Tools
* **Online 3D Visualizer:** High-fidelity browser-based rendering of crystal structures without external software.
* **Property Prediction:** Real-time access to heat capacity ($C_p$), hydrophobicity/hydrophilicity classification, and Pore Size Distribution (PSD).
* **CIF Diagnostic Upload:** A "sandbox" feature allowing users to upload personal CIF files for automated cleaning and diagnostic reporting.

---

## 📂 Database Versions

| Version | Entry Count | Core Updates |
| :--- | :--- | :--- |
| **CoRE MOF 2014** | ~5,000 | Initial release of curated experimental MOFs. |
| **CoRE MOF 2019** | ~14,000 | Improved solvent removal; introduced DDEC charges. |
| **CoRE MOF 2024/25** | **43,439** | Expanded to CSD 2025 data; added ML-based thermal and thermodynamic properties. |

---

## 🛠 Technology Stack

* **Backend:** Python-based architecture utilizing **Pymatgen**, **OpenBabel**, and **Zeo++**.
* **Frontend:** Interactive JS-driven interface for dynamic data plotting and 3D rendering.
* **ML Engines:** Graph Neural Networks (GNNs) for high-speed property prediction.

---

## 🔬 Citation

If you use this database or the provided structures in your research, please cite:

**Current Version (2024/2025):**
> *Zhao, G., et al. "CoRE MOF DB: A curated experimental metal-organic framework database with machine-learned properties for integrated material-process screening." **Matter**, 2025. DOI: 10.1016/j.matt.2025.102140*                        
> *Zhao, G., et al. "MOFClassifier: A Machine Learning Approach for Validating Computation-Ready Metal–Organic Frameworks." **Journal of the American Chemical Society**, 2025. DOI: 10.1021/jacs.5c10126*             

---

## ✉️ Contact & Feedback

* **Laboratory:** Graduate School of Data Science, Pusan National University.
* **Support:** Issues can be reported via the [Contact Page](https://mof-db.pusan.ac.kr/contact) or emailed to `drygchung@gmail.com`.

---
<p align="center"><i>Developed with passion for the porous materials community.</i></p>
