import streamlit as st
import traceback
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer


st.set_page_config(page_title="Fire Weather Index", layout='wide')

# --- Configuration ---
Pipeline_path = "linear_regression_pipeline.joblib"

# -- Function to load the complete pipeline --
@st.cache_resource
def load_pipeline():

    try:
        pipeline = joblib.load(Pipeline_path)
        return pipeline
    
    except FileNotFoundError:
        st.error(f"Pipeline file {Pipeline_path} not found. Please check your file.")
        return None
    
    except Exception as e:
        st.error(f"An error occured while loading the pipeline: {e}")
        return None
    
# -- Streamlit App Interface --

def main():
    st.title("🔥 Algerian Forest FWI Predictor")
    st.markdown("Predict the **Fire Weather Index (FWI)** value based on daily weather conditions")

    pipeline = load_pipeline()

    if pipeline is None:
        return
    
    st.head("Enter Daily Weather Details:")

    