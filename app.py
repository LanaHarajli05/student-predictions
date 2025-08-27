# Install if not available
!pip install pandas matplotlib prophet openpyxl

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Load your dataset
file_path = "/content/AI & DS Enrolled Students Course .xlsx"
df = pd.read_excel(file_path, sheet_name="All Enrolled (2)")

# Make sure Admit Semester column exists
print(df.columns)

# Count students per Admit Semester
df_counts = df.groupby("Admit Semester").size().reset_index(name="Students")

# Clean Admit Semester -> approximate date
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
df_counts = df_counts.rename(columns={"Students": "y"})
df_counts = df_counts.sort_values("ds")

# Prophet model
m = Prophet(yearly_seasonality=True, seasonality_mode="additive")
m.fit(df_counts)

future = m.make_future_dataframe(periods=6, freq="M")  # forecast 6 months (≈ 2 semesters)
forecast = m.predict(future)

# Plot
fig1 = m.plot(forecast)
plt.title("Forecast of Student Enrollments")
plt.show()
