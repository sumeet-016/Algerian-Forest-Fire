import streamlit as st
import joblib
import pandas as pd
from PIL import Image

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
    
with st.sidebar:
    st.title("🔥 Algerian Forest FWI Predictor")
    st.markdown(
        "Predict the **Fire Weather Index (FWI)** using daily weather and regional conditions."
    )
    st.markdown("---")

    st.markdown("📊 **Project Domain:** Environmental Risk & Climate Analytics")
    st.markdown("🌍 **Dataset:** Algerian Forest Fires Dataset")
    st.markdown("🧠 **Model:** Machine Learning Regression Pipeline")

    st.markdown("---")
    st.markdown("👨‍💻 Developed by: **Sumeet Kumar Pal**")
    st.markdown("🔗 GitHub: [sumeet-016](https://github.com/sumeet-016)")
    st.markdown("🔗 LinkedIn: [Profile](https://www.linkedin.com/in/palsumeet/)")

    st.markdown("---")
    st.markdown("⚠️ *This tool is for educational and research purposes only.*")


# -- Streamlit App Interface --

def main():
    st.title("🔥 Algerian Forest FWI Predictor")
    st.markdown("Predict the **Fire Weather Index (FWI)** value based on daily weather conditions")

    pipeline = load_pipeline()

    if pipeline is None:
        return
    
    st.header("Enter Daily Weather & Location Details:")

    col1, col2, col3 = st.columns(3)

    with col1:
        region_name = st.selectbox("Select Region", ["Bejaia", "Sidi-Bel Abbes"])
        region_vale = 0 if region_name == "Bejaia" else 1

        temp = st.number_input("Temperature (°C)", min_value=10.0, max_value=50.0)
        rh = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0)

    with col2:
        ws = st.number_input("wind Speed (km/hr)", min_value=10.0, max_value=100.0)
        rain = st.number_input("Rainfall (mm)", min_value=0.0, max_value=100.0)
        ffmc = st.number_input("FFMC Index", min_value=22.0, max_value=100.0)

    with col3:
        dmc = st.number_input("DMC Index", min_value=1.0, max_value=300.0)
        dc = st.number_input("DC Index", min_value=5.0, max_value=300.0)
        isi = st.number_input("ISI Index", min_value=0.0, max_value=50.0)

    bui = st.number_input("BUI Index", min_value=1.0, max_value=100.0)
    
# -- Value input in the columns as per the pipeline
    feature_columns = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'Region']
    input_values = [[temp, rh, ws, rain, ffmc, dmc, dc, isi, bui, region_vale]]
    input_data = pd.DataFrame(input_values, columns=feature_columns)
    st.divider()

    if st.button("Predict FWI Index", type="primary"):
        try:
            prediction = pipeline.predict(input_data)
            predicted_fwi = round(float(prediction[0]), 2) 
            
            st.metric(label=f"Predicted FWI for {region_name}", value=predicted_fwi)
            
            # Feedback based on FWI scale
            if predicted_fwi >= 15:
                st.error("### 🚨 High to Extreme Fire Danger")
            elif predicted_fwi >= 5:
                st.warning("### ⚠️ Moderate Fire Danger")
            else:
                st.success("### ✅ Low Fire Danger")
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info("Tip: Double-check if the pipeline expects 'Region' to be the first or last column.")
    image = Image.open("dataset-cover.jpg")
    st.image(image, width=900)



if __name__ == "__main__":
    main()