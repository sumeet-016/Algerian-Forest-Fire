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


def try_load_model(path='linear_regression_pipeline.joblib'):
    """Attempt to load a joblib model and print full traceback on failure."""
    try:
        model = joblib.load(path)
        print("MODEL LOADED OK")
        print("model type:", type(model))
        if hasattr(model, "feature_names_in_"):
            print("model.feature_names_in_ (preview up to 50):", list(getattr(model, "feature_names_in_")[:50]))
        return model
    except Exception:
        tb = traceback.format_exc()
        print("=== MODEL LOAD FAILED ===")
        print(tb)