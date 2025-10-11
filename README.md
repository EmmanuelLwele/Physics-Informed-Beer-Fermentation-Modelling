# 🍺 Physics-Informed Beer Fermentation Modelling

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()  
[![Made with PyTorch](https://img.shields.io/badge/Made%20with-PyTorch-red)](https://pytorch.org/)  

---

## 📖 Overview  

This repository implements **Physics-Informed Neural Networks (PINNs)** and baseline machine learning models for **beer fermentation modelling**. The goal is to develop a **digital twin** of the fermentation process by combining **mechanistic biochemical equations** with **data-driven learning**.  

Fermentation is modelled using **ordinary differential equations (ODEs)** that describe:  
- Substrate (sugar) consumption  
- Biomass (yeast) growth  
- Ethanol & by-product formation  
- Temperature and pH dependencies  

By embedding these dynamics into the loss function, the PINN enforces **physical realism** while learning from **experimental data**, outperforming purely data-driven approaches in sparse or noisy data regimes.  

---

## 🎯 Motivation  

Traditional kinetic models (e.g., Monod-type) offer mechanistic insights but struggle to adapt to **industrial variability**. Pure machine learning models, while flexible, often fail to generalize outside the training set.  

**PhysicsX-style hybrid approaches**—blending physics and AI—overcome these limitations. This work demonstrates how PINNs can:  
- Learn effectively from **limited or incomplete datasets**  
- Enforce **realistic biological and chemical constraints**  
- Act as **surrogate models** for real-time brewery digital twins  
- Support **process optimization** and **quality prediction** in beer production  

---

## 🔬 Methodology  



### PINN Loss Function  

### PINN Loss Function  

$$
\mathcal{L} = \lambda_\text{data}\,\mathcal{L}_\text{data} + \lambda_\text{physics}\,\mathcal{L}_\text{residual}
$$  

- **Data Loss**: Mean Squared Error (MSE) between model predictions and experimental data  
- **Physics Loss**: Residuals of fermentation ODEs enforced at collocation points  



### Governing Equations  

We consider a simplified fermentation system:  

$$
\frac{dS}{dt} = -\frac{1}{Y_{X/S}} \mu(S, T, pH)\, X
$$  

$$
\frac{dX}{dt} = \mu(S, T, pH)\, X - k_d X
$$  

$$
\frac{dP}{dt} = Y_{P/X}\, \mu(S, T, pH)\, X
$$  

where:  

- $S$: Substrate (sugar) concentration  
- $X$: Biomass (yeast) concentration  
- $P$: Product (ethanol) concentration  
- $\mu(S, T, pH)$: Specific growth rate (temperature & pH dependent)  
- $Y_{X/S}, Y_{P/X}$: Yield coefficients  
- $k_d$: Death rate constant  


---


# 🍺 Physics-Informed Neural Network (PINN) for Beer Fermentation Modeling

This repository implements a **Physics-Informed Neural Network (PINN)** to model the dynamics of sugar (Cs), biomass (X), and ethanol (Ce) concentrations during beer fermentation.  
The PINN embeds the full **Haldane kinetics**, **ethanol inhibition**, **lag phase**, and **temperature correction via Q10**, making it interpretable and physically grounded.

---

## 📐 **1. Fermentation Kinetics**

The microbial growth rate is modeled using the **full Haldane equation** with ethanol inhibition, lag phase, and temperature correction via Q10:

\[
\mu(t) = \mu_{\max} \cdot Q_{10}^{\frac{T(t) - T_{ref}}{10}}
\cdot \frac{C_s(t)}{k_s + C_s(t) + \frac{C_s(t)^2}{k_{i,\text{sugar}}}}
\cdot \frac{1}{1 + \frac{C_e(t)}{k_{i,\text{eth}}}}
\cdot \left( 1 - e^{-t/L} \right)
\]

where:

- \( \mu_{\max} \) = maximum specific growth rate [h⁻¹]  
- \( Q_{10} \) = temperature correction factor (trainable)  
- \( T(t) \) = fermentation temperature [°C]  
- \( T_{ref} \) = reference temperature [°C]  
- \( k_s \) = Monod half-saturation constant [g/L]  
- \( k_{i,\text{sugar}} \) = substrate inhibition constant [g/L]  
- \( k_{i,\text{eth}} \) = ethanol inhibition constant [g/L]  
- \( L \) = lag time constant [h]  
- \( C_s(t) \) = substrate (sugar) concentration [g/L]  
- \( C_e(t) \) = ethanol concentration [g/L]  

---

## 🧮 **2. Governing State Equations**

The state evolution is governed by mass-balance ODEs embedded in the PINN loss:

\[
\dot{X} = \mu(t) X
\]

\[
\dot{C}_s = -\frac{1}{Y_{x/s}} \, \mu(t) X
\]

\[
\dot{C}_e = \frac{Y_{e/s}}{Y_{x/s}} \, \mu(t) X
\]

where:

- \( X \) = biomass concentration [g/L]  
- \( Y_{x/s} \) = biomass yield coefficient [g/g]  
- \( Y_{e/s} \) = ethanol yield coefficient [g/g] (fixed in this implementation)

---

## 📏 **3. Observation Mapping**

Experimental measurements are obtained as **Plato gravity readings**, which are mapped to substrate concentration:

\[
C_s(t)\ [\text{g/L}] \approx 10 \times \text{Plato}(t)
\]

This is a quick empirical conversion used in brewing.

- **Fluid temperature** \( T(t) \) enters the growth model through the Q10 factor.

---

## 🧠 **4. PINN Loss Function**

The total loss combines **data fitting**, **physics residuals**, and **initial conditions**:

\[
\mathcal{L} =
w_{\text{data}} \cdot \text{MSE}\big(\hat{C}_s(t_i), C_{s,\text{data}}(t_i)\big)
+ w_{\text{phys}} \cdot \sum \text{MSE}\big(\text{residuals}\big)
+ w_{\text{IC}} \cdot \text{MSE}\big(\text{initial conditions}\big)
\]

where:

- \( w_{\text{data}} \) = weight for data loss (e.g., 1.0)  
- \( w_{\text{phys}} \) = weight for physics residuals (e.g., 100.0)  
- \( w_{\text{IC}} \) = weight for initial condition matching (e.g., 100.0)

**Residual terms** enforce the governing ODEs at collocation points, improving model generalization even with sparse data.

---

## 📊 **5. Training Notes**

- **Stage 1:** Adam optimizer (fast convergence on data and physics).  
- **Stage 2:** Optional L-BFGS optimizer refinement (for smooth convergence).  
- **Outputs:** Biomass \( X(t) \), Substrate \( C_s(t) \), and Ethanol \( C_e(t) \).

---

## 📈 **6. Performance Metrics**

Typical evaluation metrics:

- \( R^2 \) — Coefficient of determination  
- RMSE — Root Mean Squared Error  
- MAE — Mean Absolute Error  

These are computed between **predicted Cs** and **measured Plato-derived Cs**.

---

## 🧪 Example Learned Parameters (Run Output)




## 📂 Repository Structure  

```text
├── data/                 # Raw & processed fermentation datasets
├── src/
│   ├── data_loader.py    # Data preprocessing and batching
│   ├── models/
│   │   ├── pinn.py       # Physics-Informed Neural Network
│   │   ├── rnn.py        # RNN (LSTM/GRU) baseline
│   │   ├── gp.py         # Gaussian Process baseline
│   │   └── rf.py         # Random Forest baseline
│   ├── losses.py         # Data + physics-informed losses
│   ├── train.py          # Training engine
│   ├── evaluate.py       # Evaluation pipeline
│   └── utils/            # Seeds, logging, plotting, checkpointing
├── configs/              # YAML/JSON configs for experiments
├── notebooks/            # Exploratory notebooks for results & plots
├── requirements.txt      # Dependencies
├── run_experiment.py     # Entry point for training
├── evaluate.py           # Evaluate trained models
└── README.md             # Project documentation

---

# 📑 Thesis Information  

## 🎓 Thesis Title  
**Physics-Informed Surrogate Modelling and Digital Twin Development for Bioprocess Optimization**

---

## 📝 Abstract  

The accurate modelling of yeast fermentation remains a critical challenge in both the brewing and biotechnology sectors. Traditional mechanistic models, such as Monod-type kinetic equations, provide valuable insights into substrate utilization and biomass growth but often suffer from limited flexibility and the need for extensive parameter calibration. Conversely, purely data-driven machine learning approaches, including recurrent neural networks, offer predictive accuracy but lack interpretability and generalization beyond the training domain.  

In this study, we introduce **Physics-Informed Neural Networks (PINNs)** as a hybrid modelling framework for single yeast fermentation. By embedding mass-balance equations governing sugar consumption, biomass growth, and ethanol production into the neural network loss function, the PINN framework ensures that predictions remain consistent with known physical laws while fitting experimental data.  

Two laboratory-scale fermentation experiments were performed using *Saccharomyces cerevisiae* (SafAle US-05) and *Saccharomyces pastorianus* (SafLager S-23), where specific gravity, pH, and temperature were measured across a 300 h fermentation period. The trained PINNs successfully captured the dynamic behaviour of both yeast strains, demonstrating strong agreement with experimental profiles and enhanced extrapolation capabilities compared to baseline black-box models.  

These findings highlight the potential of PINNs as a robust and interpretable modelling tool for fermentation processes, paving the way for their integration into advanced control and optimization strategies in food, beverage, and bioprocess industries.  

**Keywords:** Physics-Informed Neural Networks (PINNs); fermentation modelling; yeast dynamics; *Saccharomyces cerevisiae*; *Saccharomyces pastorianus*; bioprocess systems  

---

## 📚 Thesis Contents  

### 1. Introduction  
1.1 Background  
1.2 Research Motivation  
1.3 Research Objectives  
1.4 Aims and Contributions  
1.5 Thesis Structure  

### 2. Literature Review  
2.1 Beer Fermentation: Overview and Bioprocess Challenges  
2.2 Mathematical Modelling of Bioprocess Systems  
2.3 Surrogate Modelling Approaches for Bioprocess Systems  
2.4 Physics-Informed Neural Networks (PINNs)  
- 2.4.1 Mathematical Formulation of PINNs (ODEs/PDEs in PINNs)  
- 2.4.2 PINNs for ODEs, PDEs, and Dynamic Systems  
- 2.4.3 Loss Functions and Physical Constraints  
- 2.4.4 Variants: Bayesian PINNs, Adaptive PINNs  
2.5 PINNs for Complex Systems Modelling  
- 2.5.1 Architecture of PINNs and Custom Loss Functions  
- 2.5.2 Types of PINNs: Classical, Bayesian, and Hybrid Approaches  
2.6 Applications of PINNs in Food & Drink Manufacturing Processes  
- 2.6.1 Bioprocessing, Biomedical, and Environmental Systems  
- 2.6.2 Beer Fermentation & Bioprocessing  
2.7 Integration of PINNs within Digital Twin Technologies  
- 2.7.1 Digital Twins in Bioprocessing and Beer Fermentation  
- 2.7.2 Architecture and Core Components  
- 2.7.3 Real-Time Constraints and Surrogate Modelling  
- 2.7.4 Use-Cases in Industry  
2.8 Uncertainty Quantification in Surrogate Models  
- 2.8.1 Methods, Challenges, and Open Questions  
- 2.8.2 Opportunities in Bioprocess Digital Twins  
2.9 Summary, Research Gaps and Research Questions  

### 3. Mathematical Formulation and Physical Principles  
3.1 Governing Equations in Beer Fermentation Processes  
- Substrate & Product Balances  
- Biomass Growth Kinetics (Monod & Beyond)  
3.2 PDEs in Fermentation Dynamics  
- Mass Transport (Advection-Diffusion-Reaction)  
- Fluid Flow and Mixing (Simplified Navier-Stokes)  
3.3 Dimensional Analysis and Non-Dimensional Parameters  
3.4 Formulation of Physics-Informed Surrogate Models  
3.5 Generalized PINN Framework for Dynamic Systems  
3.6 Problem Constraints, Boundary and Initial Conditions  

### 4. Development of Physics-Informed Surrogate Models  
- Modular Framework Design  
- Data Collection & Preprocessing  
- Active Learning & Sampling Strategies  
- Neural Network Architecture, Training Algorithms  
- Performance Metrics & Uncertainty Evaluation  
- Comparisons: PINNs vs GP, Random Forests, RNNs  

### 5. Experimental Validation and Case Studies  
- Beer Fermentation with Transport PDEs  
- Physical + Mathematical Model Setup  
- Simulation Configuration & Validation  
- Surrogate Model Results & Comparative Evaluation  

### 6. Digital Twin Framework for Beer Brewing Process Optimization  
- System Architecture  
- Real-Time Data Integration  
- Deployment in Industrial Brewing  
- Comparison with Existing Digital Twin Approaches  

### 7. Results and Analysis  
- Model Accuracy & Performance  
- Robustness and Generalisation  
- Uncertainty Estimation  
- Computational Cost & Scalability  
- Interpretability of Physics-Informed Layers  

### 8. Discussion  
- Key Findings  
- Contributions to Surrogate Modelling & Digital Twins  
- Limitations and Challenges  
- Future Research Directions  

### 9. Conclusion  
- Summary of Contributions  
- Industrial Implications  
- Broader Impact  

### Appendices  
- A: Additional Experimental Data  
- B: Mathematical Derivations & Analytical Comparisons  
- C: Source Code and Implementation Details  
- D: Glossary of PINN Terms and Domain-Specific Adaptations  

---

# Physics-Informed Beer Fermentation Modelling (PINNs)

**Purpose:**  
This notebook demonstrates the implementation of **Physics-Informed Neural Networks (PINNs)** and baseline surrogate models for **beer fermentation kinetics**. The goal is to integrate **mechanistic fermentation equations** (sugar consumption, biomass growth, ethanol production) with **experimental data** to build a robust **digital twin** of the fermentation process.

**Key Features:**
- Physics-informed modelling using **ODE-based loss functions**.
- Comparison with baseline models: **Random Forests, Gaussian Processes, RNN/LSTM/GRU**.
- Flexible framework for **training, evaluation, and visualization**.
- Supports **real-time prediction** for digital twin applications.

**Experimental Setup:**
- Fermentation data collected from *Saccharomyces cerevisiae* (US-05) and *Saccharomyces pastorianus* (S-23).
- Measurements include: **specific gravity, pH, and temperature** over a 300 h period.

**Notebook Structure:**
1. **Data Loading & Preprocessing** — Prepare datasets for PINN and baseline models.
2. **Model Definition** — Define PINN architecture, custom loss, and baseline ML models.
3. **Training** — Optimize PINN and baselines with multi-term loss and physics constraints.
4. **Evaluation & Visualization** — Compare predicted fermentation curves with experimental data.
5. **Surrogate Model Integration** — Showcase trained models for potential digital twin deployment.

**References:**
- Emmanuel Lwele, *Physics-Informed Surrogate Modelling and Digital Twin Development for Bioprocess Optimization*, Sheffield Hallam University, 2025.
- PINNs literature: Raissi et al., 2019; Karniadakis et al., 2021.

---

## 📊 Illustrations  

<p align="center">
  <img src="images/fermentation_tanks.png." alt="Fermentation Tanks" width="45%"/>
  <img src="images/IPA-vs-Pale-Ale.png" alt="IPA vs Pale" width="45%"/>
</p>

