import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.ticker as mticker

# Title
st.title("Sales Forecasting Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file (must have 'Date' and 'Sales' or 'sales')", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.write(df.head())

    # Check for required columns (case insensitive)
    if 'Date' not in df.columns or not any(col.lower() == 'sales' for col in df.columns):
        st.error("The file must contain 'Date' and 'Sales' (or 'sales') columns.")
    else:
        # Use the correct sales column
        sales_col = [col for col in df.columns if col.lower() == 'sales'][0]

        # Clean and convert sales to numeric
        df[sales_col] = df[sales_col].astype(str).str.replace(',', '', regex=False)
        df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce')

        # Convert Date and group by day
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', sales_col])
        df = df.groupby('Date', as_index=False)[sales_col].sum()
        df = df.rename(columns={'Date': 'ds', sales_col: 'y'})
        df = df.sort_values('ds')

        st.subheader("Aggregated Daily Sales")
        st.write(df.tail())

        # Plot sales trend
        st.subheader("Sales Trend Over Time")
        fig, ax = plt.subplots()
        ax.plot(df['ds'], df['y'], label='Sales', color='blue')
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.set_title("Daily Sales Trend")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        st.pyplot(fig)

        # Train-test split
        split_index = int(len(df) * 0.8)
        train = df.iloc[:split_index]
        test = df.iloc[split_index:]

        st.write(f"Training rows: {len(train)} | Testing rows: {len(test)}")

        # Prophet model
        model = Prophet()
        model.fit(train)
        st.success("Model trained successfully.")

        # Forecast
        future = model.make_future_dataframe(periods=len(test), freq='D')
        forecast = model.predict(future)

        st.subheader("Forecast (last few days)")
        st.write(forecast[['ds', 'yhat']].tail())

        # Forecast plot
        st.subheader("Forecast Plot")
        fig2 = model.plot(forecast)
        fig2.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        st.pyplot(fig2)

        # Evaluation
        forecast_df = forecast[['ds', 'yhat']].set_index('ds')
        test_df = test.set_index('ds')
        merged = forecast_df.join(test_df, how='inner')

        if not merged.empty:
            mae = mean_absolute_error(merged['y'], merged['yhat'])
            rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))

            # Filter out invalid values for MAPE
            filtered = merged[(merged['y'] != 0) & (~merged['y'].isna())]

            if not filtered.empty:
                mape = np.mean(np.abs((filtered['y'] - filtered['yhat']) / filtered['y'])) * 100
            else:
                mape = float('nan')

            st.subheader("Evaluation Metrics")
            st.write(f"**MAE**: {mae:,.2f}")
            st.write(f"**RMSE**: {rmse:,.2f}")
            st.write(f"**MAPE**: {mape if not np.isnan(mape) else 'N/A'}%")
        else:
            st.warning("⚠️ Not enough overlap between forecast and test data to calculate metrics.")
