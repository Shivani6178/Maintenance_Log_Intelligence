import streamlit as st
import requests
import pandas as pd
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Aircraft Maintenance Intelligence",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Aircraft Maintenance Intelligence System")
st.caption("RUL prediction from sensor data + RAG-based historical incident retrieval")

tab_search, tab_rul = st.tabs(["🔍 Maintenance Log Search", "📉 RUL Prediction"])

with tab_search:
    st.subheader("Search Historical Maintenance Incidents")
    st.write("Ask a question about engine issues, and get a grounded answer with citations from real ASRS incident reports.")

    query = st.text_input(
        "Your question",
        placeholder="e.g. What causes compressor stalls during climb?"
    )
    top_k = st.slider("Number of sources to retrieve", min_value=1, max_value=10, value=5)

    if st.button("Search", type="primary", key="search_btn"):
        if not query or len(query.strip()) < 3:
            st.warning("Please enter a question with at least 3 characters.")
        else:
            with st.spinner("Searching maintenance reports and generating answer..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/query",
                        json={"query": query, "top_k": top_k},
                        timeout=180
                    )
                    response.raise_for_status()
                    result = response.json()

                    st.markdown("### Answer")
                    st.write(result["answer"])

                    st.markdown("### Sources")
                    for source in result["sources"]:
                        with st.expander(
                            f"ACN {source['acn']} — {source['aircraft_model']} ({source['flight_phase']}) "
                            f"— relevance: {source['relevance_score']:.2f}"
                        ):
                            st.write(source["excerpt"] + "...")

                except requests.exceptions.ConnectionError:
                    st.error(f"Could not connect to the API at {API_BASE_URL}. Is the FastAPI server running?")
                except requests.exceptions.Timeout:
                    st.error("The request took too long. Please try again.")
                except requests.exceptions.HTTPError as e:
                    st.error(f"API returned an error: {e}")

with tab_rul:
    st.subheader("Predict Remaining Useful Life")
    st.write("Upload a CSV of the last 30 engine cycles (14 sensor columns) to predict remaining useful life.")

    uploaded_file = st.file_uploader("Upload sensor data CSV", type=["csv"])

    st.caption(
        "Expected format: 30 rows (most recent cycle last), 14 sensor columns. "
        "No headers required, or a header row will be auto-detected and skipped."
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if df.shape[1] > 14:
                numeric_cols = df.select_dtypes(include="number").columns[-14:]
                df = df[numeric_cols]

            st.write(f"Detected shape: {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(df.tail(5), use_container_width=True)

            if st.button("Predict RUL", type="primary", key="predict_btn"):
                if df.shape != (30, 14):
                    st.error(
                        f"Expected exactly 30 rows and 14 columns, got {df.shape}. "
                        "Please check your CSV format."
                    )
                else:
                    with st.spinner("Running prediction..."):
                        try:
                            sensor_readings = df.values.tolist()
                            response = requests.post(
                                f"{API_BASE_URL}/predict-rul",
                                json={"sensor_readings": sensor_readings},
                                timeout=30
                            )
                            response.raise_for_status()
                            result = response.json()

                            col1, col2, col3 = st.columns(3)
                            col1.metric("Predicted RUL", f"{result['predicted_rul']} cycles")
                            col2.metric("Cycles Remaining", result["cycles_remaining"])

                            warning = result["warning_level"]
                            if warning == "critical":
                                col3.error(f"⚠️ {warning.upper()}")
                            elif warning == "warning":
                                col3.warning(f"⚠️ {warning.upper()}")
                            else:
                                col3.success(f"✅ {warning.upper()}")

                        except requests.exceptions.ConnectionError:
                            st.error(f"Could not connect to the API at {API_BASE_URL}.")
                        except requests.exceptions.HTTPError as e:
                            st.error(f"API returned an error: {e}")

        except Exception as e:
            st.error(f"Could not read the uploaded file: {e}")

st.divider()
st.caption("Aircraft Maintenance Log Intelligence System — CMAPSS RUL prediction + ASRS-based RAG retrieval")