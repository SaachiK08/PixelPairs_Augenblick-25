# Causal Inference in Healthcare: Treatment Effect Analysis

## Project Overview

This Streamlit application demonstrates causal inference techniques using the Infant Health and Development Program (IHDP) dataset to understand the impact of medical interventions on outcomes.

### Problem Statement

Understanding the true causal effects of medical interventions is crucial in healthcare. This project uses advanced causal inference methods to separate correlation from causation in observational data.

## Key Features

- Interactive web interface for causal inference analysis
- Multiple causal inference methods implementation
- Visualization of treatment effects
- Personalized outcome prediction
- Statistical analysis of treatment impacts

## Methodology

### Causal Inference Techniques Implemented

1. **Propensity Score Matching (PSM)**
   - Balances covariates between treatment and control groups
   - Estimates treatment effects by matching similar units

2. **Doubly Robust Estimation**
   - Combines propensity score modeling with outcome regression
   - Provides robust estimates of treatment effects
   - Reduces bias from model misspecification

### Key Metrics Calculated

- Average Treatment Effect (ATE)
- Individual Treatment Effect (ITE)
- Propensity Scores
- Model Performance Metrics (R², MSE)

## Technical Components

### Data Processing
- Synthetic data generation
- Confounder identification
- Data normalization

### Machine Learning Models
- RandomForestRegressor for outcome prediction
- LogisticRegression for propensity score estimation
- LinearRegression for outcome modeling

### Setup
```bash
git clone <repository-url>
cd causal-inference-project
pip install -r requirements.txt
streamlit run app.py
```

## Usage

1. Navigate the sidebar to different sections
2. Use sliders to input patient characteristics
3. Select treatment status
4. Explore causal inference results and visualizations

## Sections

1. **Input & Results**
   - Personalized treatment effect estimation
   - Outcome predictions
   - Causal inference metrics

2. **Dataset Overview**
   - Data distribution
   - Covariate balance
   - Treatment group characteristics

3. **Model Diagnostics**
   - Model performance metrics
   - Feature importance analysis


## Contributors
- PixelPair
