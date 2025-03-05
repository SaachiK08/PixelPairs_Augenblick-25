import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.spatial.distance import cdist

# Set Streamlit page configuration
st.set_page_config(page_title="Causal Inference Analysis", layout="wide")

# Load or create dataset
@st.cache_data
def load_data():
    """
    Attempts to load the dataset from a CSV file.
    If file is not found, creates synthetic data instead.
    
    Returns:
        pandas.DataFrame: The loaded or synthetic dataset
    """
    try:
        data = pd.read_csv("balanced_ihdata.csv")
        # Remove unnamed columns
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        
        # Ensure treatment is numeric
        data["treatment"] = data["treatment"].astype(float)
        return data
    except FileNotFoundError:
        st.warning("balanced_ihdata.csv not found. Using synthetic data instead.")
        return create_synthetic_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return create_synthetic_data()

def create_synthetic_data(n=500, random_seed=42):
    """
    Creates synthetic data for causal inference analysis.
    
    Args:
        n (int): Number of samples to generate
        random_seed (int): Random seed for reproducibility
        
    Returns:
        pandas.DataFrame: Synthetic dataset
    """
    np.random.seed(random_seed)
    n = 500
    data = pd.DataFrame({
        "treatment": np.concatenate([np.ones(n//2), np.zeros(n//2)]),
        "income": np.random.normal(50000, 10000, n),
        "birth_weight": np.random.normal(3000, 500, n),
        "parent_edu": np.random.randint(8, 18, n),
        "health_index": np.random.normal(70, 15, n),
        "housing_quality": np.random.normal(6, 2, n),
        "neighborhood_safety": np.random.normal(7, 2, n)
    })
    beta = np.array([0.5, 0.3, 0.8, 0.4, 0.6, 0.2])
    confounders = ["income", "birth_weight", "parent_edu", "health_index", "housing_quality", "neighborhood_safety"]
    X = data[confounders].values
    data["mu0"] = X.dot(beta) + np.random.normal(0, 1, n)
    data["mu1"] = data["mu0"] + 5 + 0.1 * data["income"]/10000 + 0.2 * data["parent_edu"]
    data["outcome_factual"] = np.where(data["treatment"] == 1, data["mu1"], data["mu0"])
    data["outcome_counterfactual"] = np.where(data["treatment"] == 1, data["mu0"], data["mu1"])
    for i in range(11, 26):
        data[f"x{i}"] = np.random.normal(0, 1, n)
    return data

# Allow resetting the data
if 'data' not in st.session_state or st.sidebar.button("Reset Data"):
    st.session_state.data = load_data()
    st.session_state.models_fitted = False

data = st.session_state.data
confounders = ["income", "birth_weight", "parent_edu", "health_index", "housing_quality", "neighborhood_safety"]

# Initialize scaler for normalizing numeric inputs
if 'scaler' not in st.session_state:
    st.session_state.scaler = StandardScaler().fit(data[confounders])

@st.cache_data(show_spinner=True)
def train_model(data, treatment_group, confounders):
    """
    Trains a RandomForestRegressor model for a specific treatment group.
    
    Args:
        data (pandas.DataFrame): Input dataset
        treatment_group (int): Treatment value (0 or 1)
        confounders (list): List of confounder variable names
        
    Returns:
        tuple: (fitted model, r2 score, mean squared error)
    """
    subset = data[data["treatment"] == treatment_group]
    if len(subset) == 0:
        raise ValueError(f"No data available for treatment group {treatment_group}")
    
    X, y = subset[confounders], subset["outcome_factual"]
    # Normalize inputs
    X_scaled = st.session_state.scaler.transform(X)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Calculate model metrics
    y_pred = model.predict(X_scaled)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    return model, r2, mse

def predict_outcome(model, input_data):
    """
    Predicts outcome using a fitted model.
    
    Args:
        model: Fitted model to use for prediction
        input_data (pandas.DataFrame): Input features
        
    Returns:
        float: Predicted outcome
    """
    try:
        # Scale input data
        input_data_scaled = st.session_state.scaler.transform(input_data)
        return model.predict(input_data_scaled)[0]
    except NotFittedError:
        st.error("Model is not fitted yet. Please check your data.")
        return None
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

@st.cache_data
def calculate_ate(data):
    """
    Calculates the Average Treatment Effect.
    
    Args:
        data (pandas.DataFrame): Input dataset
        
    Returns:
        float: Average Treatment Effect
    """
    return data["mu1"].mean() - data["mu0"].mean()

def check_covariate_balance(data, confounders):
    """
    Checks balance of covariates between treatment and control groups.
    
    Args:
        data (pandas.DataFrame): Input dataset
        confounders (list): List of confounder variable names
        
    Returns:
        pandas.DataFrame: Statistics showing balance
    """
    balance_stats = []
    for var in confounders:
        treated_mean = data[data["treatment"] == 1][var].mean()
        control_mean = data[data["treatment"] == 0][var].mean()
        std_diff = (treated_mean - control_mean) / data[var].std()
        balance_stats.append({
            "Variable": var,
            "Treated Mean": treated_mean,
            "Control Mean": control_mean,
            "Diff": treated_mean - control_mean,
            "Std Diff": std_diff,
            "Balanced": abs(std_diff) < 0.25
        })
    return pd.DataFrame(balance_stats)

def propensity_score_matching(data, confounders):
    """
    Implement Propensity Score Matching for causal inference.
    
    Args:
        data (pd.DataFrame): Input dataset
        confounders (list): List of confounder variables
    
    Returns:
        dict: Matching results and statistics
    """
    # Estimate propensity scores
    X = data[confounders]
    y = data['treatment']
    
    propensity_model = LogisticRegression(max_iter=1000)
    propensity_model.fit(X, y)
    
    # Calculate propensity scores
    propensity_scores = propensity_model.predict_proba(X)[:, 1]
    
    # Matching process
    treated = data[data['treatment'] == 1]
    control = data[data['treatment'] == 0]
    
    matched_outcomes = []
    matched_ites = []
    
    for _, treated_row in treated.iterrows():
        # Find closest control unit
        treated_ps = propensity_model.predict_proba(treated_row[confounders].values.reshape(1, -1))[:, 1]
        control_ps = propensity_model.predict_proba(control[confounders].values)[:, 1]
        
        distances = np.abs(control_ps - treated_ps)
        closest_control_idx = distances.argmin()
        
        # Calculate Individual Treatment Effect for matched pair
        treated_outcome = treated_row['outcome_factual']
        control_outcome = control.iloc[closest_control_idx]['outcome_factual']
        ite = treated_outcome - control_outcome
        
        matched_outcomes.append((treated_outcome, control_outcome))
        matched_ites.append(ite)
    
    return {
        'matched_ate': np.mean(matched_ites),
        'matched_outcomes': matched_outcomes,
        'pscore_balance': propensity_model.coef_[0]
    }

def doubly_robust_estimation(data, confounders):
    """
    Implement Doubly Robust Estimation for causal inference.
    
    Args:
        data (pd.DataFrame): Input dataset
        confounders (list): List of confounder variables
    
    Returns:
        dict: Doubly robust estimation results
    """
    # Propensity score model
    propensity_model = LogisticRegression(max_iter=1000)
    propensity_model.fit(data[confounders], data['treatment'])
    propensity_scores = propensity_model.predict_proba(data[confounders])[:, 1]
    
    # Outcome models for treatment and control
    outcome_model_treated = LinearRegression()
    outcome_model_control = LinearRegression()
    
    # Fit separate outcome models
    outcome_model_treated.fit(
        data[data['treatment'] == 1][confounders], 
        data[data['treatment'] == 1]['outcome_factual']
    )
    outcome_model_control.fit(
        data[data['treatment'] == 0][confounders], 
        data[data['treatment'] == 0]['outcome_factual']
    )
    
    # Doubly robust ATE estimation
    n = len(data)
    dr_estimates = []
    
    for i in range(n):
        x = data.loc[i, confounders].values.reshape(1, -1)
        true_treatment = data.loc[i, 'treatment']
        
        # Predict outcomes
        outcome_treated = outcome_model_treated.predict(x)[0]
        outcome_control = outcome_model_control.predict(x)[0]
        
        # Propensity score components
        ps = propensity_scores[i]
        
        # Doubly robust estimate
        dr_estimate = (
            true_treatment * (data.loc[i, 'outcome_factual'] - outcome_treated) / ps +
            (1 - true_treatment) * (data.loc[i, 'outcome_factual'] - outcome_control) / (1 - ps) +
            outcome_treated - outcome_control
        )
        
        dr_estimates.append(dr_estimate)
    
    return {
        'doubly_robust_ate': np.mean(dr_estimates),
        'dr_estimates': dr_estimates
    }

# Sidebar navigation
page = st.sidebar.radio("Navigate", ["Input & Results", "Dataset Overview", "Model Diagnostics"])
n_treated = (data["treatment"] == 1).sum()
n_control = (data["treatment"] == 0).sum()

st.sidebar.write(f"Dataset: {len(data)} samples")
st.sidebar.write(f"- Treated group: {n_treated} samples")
st.sidebar.write(f"- Control group: {n_control} samples")

# Train models
try:
    if 'models_fitted' not in st.session_state or not st.session_state.models_fitted:
        with st.spinner("Training models..."):
            st.session_state.treated_model, st.session_state.treated_r2, st.session_state.treated_mse = train_model(data, 1, confounders)
            st.session_state.control_model, st.session_state.control_r2, st.session_state.control_mse = train_model(data, 0, confounders)
            st.session_state.models_fitted = True
except Exception as e:
    st.error(f"Error training models: {str(e)}")
    st.session_state.models_fitted = False

if page == "Input & Results":
    st.title("Causal Inference Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        income = st.slider("Income", float(data["income"].min()), float(data["income"].max()), float(data["income"].median()))
        birth_weight = st.slider("Birth Weight", float(data["birth_weight"].min()), float(data["birth_weight"].max()), float(data["birth_weight"].median()))
        parent_edu = st.slider("Parent Education", int(data["parent_edu"].min()), int(data["parent_edu"].max()), int(data["parent_edu"].median()))
    with col2:
        health_index = st.slider("Health Index", float(data["health_index"].min()), float(data["health_index"].max()), float(data["health_index"].median()))
        housing_quality = st.slider("Housing Quality", float(data["housing_quality"].min()), float(data["housing_quality"].max()), float(data["housing_quality"].median()))
        neighborhood_safety = st.slider("Neighborhood Safety", float(data["neighborhood_safety"].min()), float(data["neighborhood_safety"].max()), float(data["neighborhood_safety"].median()))
    
    treatment = st.radio("Treatment Applied?", ["Yes", "No"])
    treatment_value = 1 if treatment == "Yes" else 0

    if st.session_state.models_fitted:
        input_data = pd.DataFrame({
            "income": [income], 
            "birth_weight": [birth_weight], 
            "parent_edu": [parent_edu],
            "health_index": [health_index], 
            "housing_quality": [housing_quality], 
            "neighborhood_safety": [neighborhood_safety]
        })
        
        treated_outcome = predict_outcome(st.session_state.treated_model, input_data)
        control_outcome = predict_outcome(st.session_state.control_model, input_data)
        
        if treated_outcome is not None and control_outcome is not None:
            individual_treatment_effect = treated_outcome - control_outcome
            factual_outcome = treated_outcome if treatment_value else control_outcome
            counterfactual_outcome = control_outcome if treatment_value else treated_outcome
            
            # Perform Propensity Score Matching and Doubly Robust Estimation
            psm_results = propensity_score_matching(data, confounders)
            dre_results = doubly_robust_estimation(data, confounders)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Population ATE", f"{calculate_ate(data):.4f}")
                st.metric("Your Factual Outcome", f"{factual_outcome:.4f}")
                st.metric("Propensity Score Matched ATE", f"{psm_results['matched_ate']:.4f}")
            with col2:
                st.metric("Your Counterfactual Outcome", f"{counterfactual_outcome:.4f}")
                st.metric("Estimated Individual Treatment Effect", f"{individual_treatment_effect:.4f}")
                st.metric("Doubly Robust ATE", f"{dre_results['doubly_robust_ate']:.4f}")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Outcome Comparison
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=["Control", "Treated"], y=[control_outcome, treated_outcome], ax=ax)
                ax.set_ylabel("Predicted Outcome")
                ax.set_title("Predicted Outcomes by Treatment Group")
                st.pyplot(fig)
            
            with col2:
                # Matched Outcomes Distribution
                matched_outcomes = np.array(psm_results['matched_outcomes'])
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(matched_outcomes[:, 0], label='Treated Outcomes', kde=True, ax=ax)
                sns.histplot(matched_outcomes[:, 1], label='Control Outcomes', kde=True, ax=ax)
                ax.set_title("Matched Outcomes Distribution")
                ax.set_xlabel("Outcome")
                plt.legend()
                st.pyplot(fig)
            
            # Detailed Propensity Score and Doubly Robust Estimation Information
            st.subheader("Causal Inference Details")
            
            expander1 = st.expander("Propensity Score Matching Details")
            expander1.write("Propensity Score Balance:")
            expander1.write(pd.DataFrame({
                'Confounder': confounders,
                'Balance Coefficient': psm_results['pscore_balance']
            }))
            
            expander2 = st.expander("Doubly Robust Estimation Details")
            expander2.write(f"Mean Doubly Robust Estimates: {np.mean(dre_results['dr_estimates']):.4f}")
            expander2.write("Individual Doubly Robust Estimates Distribution:")
            expander2.line_chart(dre_results['dr_estimates'])
    
    # if st.session_state.models_fitted:
    #     input_data = pd.DataFrame({
    #         "income": [income], 
    #         "birth_weight": [birth_weight], 
    #         "parent_edu": [parent_edu],
    #         "health_index": [health_index], 
    #         "housing_quality": [housing_quality], 
    #         "neighborhood_safety": [neighborhood_safety]
    #     })
        
    #     treated_outcome = predict_outcome(st.session_state.treated_model, input_data)
    #     control_outcome = predict_outcome(st.session_state.control_model, input_data)
        
    #     if treated_outcome is not None and control_outcome is not None:
    #         individual_treatment_effect = treated_outcome - control_outcome
    #         factual_outcome = treated_outcome if treatment_value else control_outcome
    #         counterfactual_outcome = control_outcome if treatment_value else treated_outcome
            
    #         col1, col2 = st.columns(2)
    #         with col1:
    #             st.metric("Population ATE", f"{calculate_ate(data):.4f}")
    #             st.metric("Your Factual Outcome", f"{factual_outcome:.4f}")
    #         with col2:
    #             st.metric("Your Counterfactual Outcome", f"{counterfactual_outcome:.4f}")
    #             st.metric("Estimated Individual Treatment Effect", f"{individual_treatment_effect:.4f}")
            
    #         fig, ax = plt.subplots(figsize=(10, 6))
    #         sns.barplot(x=["Control", "Treated"], y=[control_outcome, treated_outcome], ax=ax)
    #         ax.set_ylabel("Predicted Outcome")
    #         ax.set_title("Predicted Outcomes by Treatment Group")
    #         st.pyplot(fig)

elif page == "Dataset Overview":
    st.title("Dataset Overview")
    st.dataframe(data.head())
    
    st.subheader("Treatment Group Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=["Control", "Treated"], y=[n_control, n_treated], ax=ax)
    ax.set_ylabel("Count")
    ax.set_title("Number of Samples by Treatment Group")
    st.pyplot(fig)
    
    st.subheader("Covariate Balance")
    balance_df = check_covariate_balance(data, confounders)
    st.dataframe(balance_df)
    
    st.subheader("Confounder Distributions by Treatment Group")
    confounder_col1, confounder_col2 = st.columns(2)
    
    for i, confounder in enumerate(confounders):
        col = confounder_col1 if i % 2 == 0 else confounder_col2
        with col:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=data, x=confounder, hue="treatment", common_norm=False, 
                      element="step", palette=["blue", "red"], ax=ax)
            ax.set_title(f"{confounder} Distribution by Treatment Group")
            st.pyplot(fig)

elif page == "Model Diagnostics":
    st.title("Model Diagnostics")
    
    if st.session_state.models_fitted:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Treated Group Model (Treatment = 1)")
            st.metric("R² Score", f"{st.session_state.treated_r2:.4f}")
            st.metric("Mean Squared Error", f"{st.session_state.treated_mse:.4f}")
            
            if hasattr(st.session_state.treated_model, "feature_importances_"):
                st.subheader("Feature Importance (Treated Model)")
                importance_df = pd.DataFrame({
                    'Feature': confounders,
                    'Importance': st.session_state.treated_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
                st.pyplot(fig)
        
        with col2:
            st.subheader("Control Group Model (Treatment = 0)")
            st.metric("R² Score", f"{st.session_state.control_r2:.4f}")
            st.metric("Mean Squared Error", f"{st.session_state.control_mse:.4f}")
            
            if hasattr(st.session_state.control_model, "feature_importances_"):
                st.subheader("Feature Importance (Control Model)")
                importance_df = pd.DataFrame({
                    'Feature': confounders,
                    'Importance': st.session_state.control_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
                st.pyplot(fig)
    else:
        st.warning("Models not fitted yet. Please check your data or refresh the page.")