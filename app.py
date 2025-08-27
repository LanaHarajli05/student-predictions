import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Student Forecasting", layout="wide")

st.title("Student Enrollment Forecasting (by Semester)")

# Upload dataset
uploaded_file = st.file_uploader("Upload Excel file with 'Admit Semester' column", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name="All Enrolled (2)")
    
    # Aggregate
    df_counts = df.groupby("Admit Semester").size().reset_index(name="Students")

    # Convert semester → date
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
        return None

    df_counts["ds"] = pd.to_datetime(df_counts["Admit Semester"].apply(semester_to_date))
    df_counts = df_counts.rename(columns={"Students": "y"}).sort_values("ds")

    # Forecast horizon (semesters)
    horizon = st.slider("Forecast horizon (semesters)", min_value=2, max_value=8, value=6)

    # Prophet
    m = Prophet(yearly_seasonality=True)
    m.fit(df_counts)

    future = m.make_future_dataframe(periods=horizon, freq="6M")
    forecast = m.predict(future)

    # Map forecast dates back → Admit Semester labels
    def date_to_semester(date):
        year = date.year
        month = date.month
        if month == 9:
            return f"Fall {str(year)[2:]}-{str(year+1)[2:]}"
        elif month == 2:
            return f"Spring {str(year-1)[2:]}-{str(year)[2:]}"
        elif month == 6:
            return f"Summer {str(year-1)[2:]}-{str(year)[2:]}"
        else:
            return str(year)

    forecast["Admit Semester"] = forecast["ds"].apply(date_to_semester)

    # Plot with Plotly (clean labels)
    st.subheader("Enrollment Forecast")
    fig = px.line(
        forecast.tail(horizon+len(df_counts)), 
        x="Admit Semester", y="yhat",
        error_y=forecast["yhat_upper"] - forecast["yhat"],
        error_y_minus=forecast["yhat"] - forecast["yhat_lower"],
        title="Forecast of Student Enrollments per Semester"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show forecasted values
    st.subheader("Forecasted Table (per Semester)")
    st.dataframe(forecast[["Admit Semester", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon))
else:
    st.info("Please upload your dataset to generate a forecast.")
