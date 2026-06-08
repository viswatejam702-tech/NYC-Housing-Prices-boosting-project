import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="NYC Housing Intelligence",
    page_icon="🏠",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        r"C:\ds and AI\ALL DATASETS\nyc_housing_base.csv"
    )

df = load_data()

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    return joblib.load(
        "nyc_housing_ai.pkl"
    )

model = load_model()

# =====================================================
# SIDEBAR
# =====================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Prediction",
        "Analytics"
    ]
)

# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":

    st.title("🏠 NYC Housing Intelligence Platform")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Properties",
        len(df)
    )

    c2.metric(
        "Average Price",
        f"${df['sale_price'].mean():,.0f}"
    )

    c3.metric(
        "Maximum Price",
        f"${df['sale_price'].max():,.0f}"
    )

    c4.metric(
        "Minimum Price",
        f"${df['sale_price'].min():,.0f}"
    )

    fig = px.histogram(
        df,
        x="sale_price",
        nbins=50,
        title="Sale Price Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# PREDICTION PAGE
# =====================================================

elif page == "Prediction":

    st.title("💰 Property Price Prediction")

    col1, col2 = st.columns(2)

    with col1:

        zip_code = st.number_input(
            "Zip Code",
            value=10001
        )

        yearbuilt = st.number_input(
            "Year Built",
            value=2000
        )

        lotarea = st.number_input(
            "Lot Area",
            value=1000.0
        )

        bldgarea = st.number_input(
            "Building Area",
            value=2000.0
        )

        resarea = st.number_input(
            "Residential Area",
            value=1500.0
        )

        comarea = st.number_input(
            "Commercial Area",
            value=0.0
        )

    with col2:

        unitsres = st.number_input(
            "Residential Units",
            value=1
        )

        unitstotal = st.number_input(
            "Total Units",
            value=1
        )

        numfloors = st.number_input(
            "Floors",
            value=2
        )

        latitude = st.number_input(
            "Latitude",
            value=40.7128
        )

        longitude = st.number_input(
            "Longitude",
            value=-74.0060
        )

        building_age = st.number_input(
            "Building Age",
            value=20
        )

    if st.button("Predict Price"):

        input_df = pd.DataFrame({

            "zip_code":[zip_code],
            "yearbuilt":[yearbuilt],

            "lotarea":[lotarea],
            "bldgarea":[bldgarea],
            "resarea":[resarea],
            "comarea":[comarea],

            "unitsres":[unitsres],
            "unitstotal":[unitstotal],

            "numfloors":[numfloors],

            "latitude":[latitude],
            "longitude":[longitude],

            "building_age":[building_age]

        })

        prediction = model.predict(
            input_df
        )[0]

        st.success(
            f"Predicted Price: ${prediction:,.2f}"
        )

        st.metric(
            "Predicted Property Value",
            f"${prediction:,.0f}"
        )

# =====================================================
# ANALYTICS
# =====================================================

elif page == "Analytics":

    st.title("📊 Analytics Dashboard")

    numeric_df = df.select_dtypes(
        include=np.number
    )

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig2 = px.scatter(
        df,
        x="bldgarea",
        y="sale_price",
        title="Building Area vs Sale Price"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.dataframe(
        df.head()
    )