import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Forecasting Dashboard", layout="wide")

st.title("📈 Student Enrollment Forecasting")

# File uploader
uploaded_file = st.file_uploader("Upload Excel file with 'Admit Semester' column", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name="All Enrolled (2)")
    
    # Aggregate enrollments per Admit Semester
    df_counts = df.groupby("Admit Semester").size().reset_index(name="Students")
    
    # Convert Admit Semester to date
    def semester_to_date(x):
        term, years = x.split()
        start, end = years.split("-")
        start = int("20" + start)
        end = int("20" + end)
        if term == "Fall":
            return f"{start}-09-01"
        elif term == "Spring":
            return f"{end}-02-01"
        elif term == "Summer":
            return f"{end}-06-01"
        else:
            return None
    
    df_counts["ds"] = pd.to_datetime(df_counts["Admit Semester"].apply(semester_to_date))
    df_counts = df_counts.rename(columns={"Students": "y"}).sort_values("ds")

    # Forecast horizon
    horizon = st.slider("Forecast horizon (months)", min_value=3, max_value=24, value=12)

    # Prophet model
    m = Prophet(yearly_seasonality=True)
    m.fit(df_counts)

    future = m.make_future_dataframe(periods=horizon, freq="M")
    forecast = m.predict(future)

    # Plot
    st.subheader("Enrollment Forecast")
    fig1 = m.plot(forecast)
    st.pyplot(fig1)

    # Show forecast data
    st.subheader("Forecasted Values")
    st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon))
else:
    st.info("Upload a dataset to begin forecasting.")
