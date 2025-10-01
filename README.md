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

## 📂 Repository Structure  

