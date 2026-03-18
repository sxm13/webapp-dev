# PACMAN-Charge: ML-Based Partial Charge Assignment for Nanoporous Materials

<div align="center">
  <img src="https://raw.githubusercontent.com/sxm13/pypi-dev/main/logos/pacman-charge.png" alt="PACMAN charge logo" width="500"/>

  <h3>Rapid and Accurate Partial Atomic Charge Prediction using Machine Learning</h3>

  [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pacman-charge-mtap.streamlit.app/)
  [![Method](https://img.shields.io/badge/Support-DDEC6/Bader/CM5/REPEAT-blue)](https://pacman-charge-mtap.streamlit.app/)

  [**🚀 Launch PACMAN-Charge App**](https://pacman-charge-mtap.streamlit.app/)
</div>

---

## 🌟 Overview

**PACMAN-Charge** (Partial Atomic Charge Manifold) is a specialized Web Application developed by the **Guobin Zhao at Pusan National University**. It addresses the computational bottleneck of assigning partial atomic charges to Metal-Organic Frameworks (MOFs). 

While traditional methods like **DDEC6** or **RESP** require expensive Density Functional Theory (DFT) calculations taking hours or days, **PACMAN-Charge** leverages pre-trained Machine Learning models to predict high-quality, DDEC6-equivalent charges in **seconds**.

---

## 🛠 Key Features

### 1. DDEC6-Equivalent Accuracy
* The underlying ML model is trained on a vast library of experimental and hypothetical MOFs with charges calculated via the **DDEC6** (Density Derived Electrostatic and Chemical) method.
* Provides the necessary accuracy for Grand Canonical Monte Carlo (GCMC) simulations of polar molecule adsorption (e.g., $H_2O, CO_2, SO_2$).

### 2. High-Speed Processing
* **Seconds vs. Days:** Skip the DFT electronic structure calculation entirely.
* **Batch Processing:** Designed to handle multiple structures or large unit cells that are traditionally prohibitive for quantum chemical methods.

### 3. Structural Compatibility
* Supports a wide range of MOF chemistries, including diverse metal nodes and organic linkers.
* Automatically handles periodic boundary conditions for crystalline systems.

### 4. Simple Web Interface
* **Upload:** Simply drag and drop your `.cif` file.
* **Process:** The ML engine assigns charges based on the local chemical environment of each atom.
* **Download:** Export the charge-assigned CIF file, ready for integration into simulation packages like **RASPA**, **LAMMPS**, or **GROMACS**.

---

## 📂 How It Works



1.  **Atom Featurization:** The tool analyzes the coordination environment, electronegativity, and local geometry of each atom in the CIF.
2.  **ML Inference:** A trained neural network predicts the partial charge based on learned chemical patterns.
3.  **Charge Neutralization:** The system ensures the total net charge of the unit cell is exactly zero to maintain physical consistency.

---

## 🚀 Usage Guide

1. **Upload:** Upload your "Computation-Ready" CIF file to the Streamlit interface.
2. **Predict:** Click the **Assign Charges** button.
3. **Inspect:** View the distribution of charges across different atom types (C, H, O, Metal) through the interactive dashboard.
4. **Export:** Download the updated CIF with the `_atom_site_charge` data block included.

---

## 🔬 Citation

If **PACMAN-Charge** contributes to your research, please cite the original methodology:

> *Guobin Zhao and Yongchul G. Chung. Journal of Chemical Theory and Computation 2024 20 (12), 5368-5380. DOI: 10.1021/acs.jctc.4c00434*

---

## ✉️ Contact

* **Developer:** Chung Research Group, Pusan National University.
* **Technical Support:** [sxmzhaogb@gmail.com](mailto:sxmzhaogb@gmail.com)

---
<p align="center"><i>Enabling Large-Scale Electrostatic Modeling of Porous Materials.</i></p>
