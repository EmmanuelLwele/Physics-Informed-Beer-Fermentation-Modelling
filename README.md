# Physics-Informed Beer Fermentation Modelling

Author: Emmanuel Lwele  
Description: Physics-informed neural network (PINN) and baseline models for modelling beer fermentation kinetics (substrate, biomass, product, temperature & pH dependency).

## Quick start

1. Create env:
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2. Train (example):
   python run_experiment.py --config configs/default.yaml

3. Evaluate:
   python evaluate.py --checkpoint outputs/checkpoint.pt --data data/test.csv

## Repo layout
- data/                # raw & processed datasets
- src/
  - data_loader.py
  - models/
    - pinn.py
    - baseline_rnn.py
  - losses.py
  - train.py
  - evaluate.py
- notebooks/           # exploratory notebooks
- configs/             # YAML/JSON experiment configs
- requirements.txt

## Reproducibility
- Seeds set in `utils/seed.py`
- Example `Dockerfile` included

## Contact
Emmanuel Lwele — emmanuel@example.com
