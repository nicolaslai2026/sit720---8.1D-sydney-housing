from pathlib import Path
import numpy as np, pandas as pd, streamlit as st, joblib

st.set_page_config(page_title="Sydney Housing Price Estimator", page_icon="🏠", layout="centered")

@st.cache_resource
def load_artifact():
    p = Path("model/housing_model.joblib")
    return joblib.load(p) if p.exists() else None

art = load_artifact()
st.title("🏠 Sydney Housing Price Estimator")
st.caption("Decision-support tool — an estimate to inform an appraisal, not replace it.")

if art is None:
    st.error("model/housing_model.joblib not found. Run the notebook (Part 6 cell) first.")
    st.stop()
st.info(f"Model in use: **{art['model_name']}**  ·  Suburbs: {', '.join(art['suburbs'])}")

c1, c2 = st.columns(2)
with c1:
    suburb = st.selectbox("Suburb", art["suburbs"])
    property_type = st.selectbox("Property type", art["property_types"])
    bedrooms = st.number_input("Bedrooms", 0, 10, 3)
with c2:
    bathrooms = st.number_input("Bathrooms", 0, 10, 2)
    car_spaces = st.number_input("Car spaces", 0, 10, 1)

if st.button("Estimate sale price", type="primary"):
    row = pd.DataFrame([{
        "bedrooms": bedrooms, "bathrooms": bathrooms, "car_spaces": car_spaces,
        "total_rooms": bedrooms + bathrooms,          # engineered feature (matches notebook)
        "suburb": suburb, "property_type": property_type,
    }])
    pred = float(art["pipeline"].predict(row)[0])
    st.metric("Estimated sale price", f"${pred:,.0f}")
    st.caption(f"Indicative range (±15%): ${pred*0.85:,.0f} – ${pred*1.15:,.0f}")
    st.warning("Estimate only. Built on rooms, car spaces, suburb and property type — "
               "it can't see condition, land size, aspect, views or renovations.")
