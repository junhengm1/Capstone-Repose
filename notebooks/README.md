# Notebooks

This directory stores exploratory data processing and model development notebooks.

## `Capstone_PD_v3.ipynb`

Prepares and validates the raw geospatial/property datasets used by the project. It covers fire facilities, bushfire-prone areas, fire history, vegetation, renewable project data, property geometry handling, and feature engineering.

## `Capstone_ML_v2.ipynb`

Trains and evaluates the LightGBM bushfire risk model using processed property-level features. It produces calibrated risk probability, risk score, and risk level outputs used by downstream SFT and DPO data preparation.

Run notebooks from the project root after installing the environment:

```
pip install -r requirements.txt
jupyter lab
```
