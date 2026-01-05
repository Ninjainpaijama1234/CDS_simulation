import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from io import StringIO
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------------------------
# CONSTANTS & CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="CESIM Dashboard", layout="wide")

TEAMS_DEFAULT = ["Green", "Red", "Blue", "Orange", "Grey", "Ochre"]

# Mapping keywords to Model Types
MODEL_MAPPING = {
    "Sales": "linear_trend",
    "Revenue": "linear_trend",
    "Cost": "exp_smoothing",
    "Expense": "exp_smoothing",
    "R&D": "exp_smoothing",
    "Promotion": "exp_smoothing",
    "Profit": "linear_trend",
    "EBIT": "linear_trend",
    "Assets": "drift",
    "Equity": "drift",
    "Debt": "drift",
    "Cash": "drift",
    "Inventory": "drift",
    "Ratio": "moving_average",
    "Margin": "moving_average"
}

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS: PARSING & CLEANING
# -----------------------------------------------------------------------------

def clean_currency(val):
    """
    Cleans string values like "4,13,52,625" -> 41352625.0
    Handles Indian/standard separators by stripping all non-numeric chars except '.' and '-'
    """
    if pd.isna(val) or val == "":
        return 0.0
    val_str = str(val)
    # Remove everything that is not a digit, a minus sign, or a decimal point
    clean_str = re.sub(r'[^\d.-]', '', val_str)
    try:
        return float(clean_str)
    except ValueError:
        return 0.0

def extract_round_number(filename):
    """Extracts round number from filename (e.g., 'results-ir01.csv' -> 1)"""
    digits = re.findall(r'\d+', filename)
    if digits:
        return int(digits[-1])
    return 0

def parse_cesim_file(uploaded_file):
    """
    Parses a single CESIM CSV/Excel file into a tidy DataFrame.
    """
    filename = uploaded_file.name
    round_num = extract_round_number(filename)
    
    # Read file content
    if filename.endswith('.csv'):
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        # Read without header initially to parse blocks manually
        raw_df = pd.read_csv(stringio, header=None)
    else:
        # Excel fallback
        raw_df = pd.read_excel(uploaded_file, header=None)

    parsed_data = []
    
    current_scope = "Unknown"
    current_statement = "Unknown"
    current_teams = []
    in_block = False
    
    # Iterate row by row to detect blocks
    for idx, row in raw_df.iterrows():
        first_col = str(row[0]).strip()
        
        # Detect Header Block (e.g., "Income statement, k USD, Global")
        if "Income statement" in first_col or "Balance sheet" in first_col:
            parts = [p.strip() for p in first_col.split(',')]
            # Usually: [Type, Unit, Scope]
            current_statement = parts[0] if len(parts) > 0 else "Unknown"
            current_scope = parts[-1] if len(parts) > 1 else "Global"
            in_block = True
            # The NEXT row usually contains team names, handled by logic below
            continue
            
        if in_block:
            # Check if this row is the Team Header row
            # Team header rows typically start with an empty cell or match known team names
            row_values = [str(x).strip() for x in row.values]
            
            # Heuristic: verify if intersection with known teams exists or if col 1 is empty/null 
            # and col 2 is a potential team name
            potential_teams = row_values[1:]
            
            # If we find standard team names in this row, treat it as header
            if any(t in TEAMS_DEFAULT for t in potential_teams):
                current_teams = potential_teams
                continue
            
            # If first column is empty, it might be a spacer or header, skip
            if not first_col or first_col == "nan":
                continue
                
            # Otherwise, it's a Data Row (Metric + Values)
            metric_name = first_col
            
            # Safety: Ensure we have teams to map to
            if not current_teams:
                continue
                
            for col_idx, team_name in enumerate(current_teams):
                # Data values start at index 1 (index 0 is metric name)
                if col_idx + 1 < len(row):
                    raw_val = row[col_idx + 1]
                    val = clean_currency(raw_val)
                    
                    # Store tidy record
                    if team_name and team_name != "nan": # filter empty cols
                        parsed_data.append({
                            "round": round_num,
                            "scope": current_scope,
                            "statement_type": current_statement,
                            "metric": metric_name,
                            "team": team_name,
                            "value_k_usd": val
                        })

    return pd.DataFrame(parsed_data)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS: ANALYTICS & FORECAST
# -----------------------------------------------------------------------------

def compute_derived_metrics(df):
    """
    Adds calculated rows (Gross Profit, Ratios) to the DataFrame.
    Note: Ideally handled by pivoting, calculating, and melting back, 
    but for simplicity, we compute on the fly in the UI or here.
    """
    # For this dashboard, we will compute derived metrics dynamically 
    # in the Overview tab to avoid complex DataFrame manipulation here.
    return df

def get_default_model(metric_name):
    """Determine default forecast model based on metric keywords."""
    for key, model in MODEL_MAPPING.items():
        if key.lower() in metric_name.lower():
            return model
    return "naive"

def forecast_series(series, model_type="auto", horizon=3):
    """
    Generates a forecast for a pandas Series (indexed by round).
    series: pd.Series (index=round, value=float)
    returns: (pd.Series[historical], pd.Series[forecast])
    """
    if series.empty:
        return series, pd.Series()

    # Sort index to be sure
    series = series.sort_index()
    last_round = series.index.max()
    future_rounds = np.arange(last_round + 1, last_round + 1 + horizon)
    
    # Auto-selection logic handled before call, but fallback here
    if model_type == "auto":
        model_type = "naive"

    predictions = []

    if model_type == "naive":
        # Last value carried forward
        last_val = series.iloc[-1]
        predictions = [last_val] * horizon

    elif model_type == "moving_average":
        # Average of last 3 rounds (or length of series)
        window = min(len(series), 3)
        avg = series.tail(window).mean()
        predictions = [avg] * horizon

    elif model_type == "linear_trend":
        if len(series) > 1:
            X = np.array(series.index).reshape(-1, 1)
            y = series.values
            model = LinearRegression()
            model.fit(X, y)
            X_future = future_rounds.reshape(-1, 1)
            pred_vals = model.predict(X_future)
            predictions = pred_vals
        else:
            # Fallback to naive if not enough points
            predictions = [series.iloc[-1]] * horizon

    elif model_type == "exp_smoothing":
        # Simple implementation: Level_t = alpha * x_t + (1-alpha) * Level_{t-1}
        # With trend component simplified
        values = series.values
        if len(values) < 2:
            predictions = [values[-1]] * horizon
        else:
            # Calculate simple geometric growth rate or just smooth
            # Use simple weighted average of recent vs old
            alpha = 0.5
            level = values[0]
            for v in values[1:]:
                level = alpha * v + (1 - alpha) * level
            # Project level flatly (Conservative for costs)
            predictions = [level] * horizon

    elif model_type == "drift":
        # Random Walk with Drift: Last + Average Change
        if len(series) > 1:
            diffs = series.diff().dropna()
            avg_drift = diffs.mean()
            last_val = series.iloc[-1]
            predictions = [last_val + avg_drift * (i + 1) for i in range(horizon)]
        else:
            predictions = [series.iloc[-1]] * horizon

    else:
        predictions = [series.iloc[-1]] * horizon

    forecast_series = pd.Series(predictions, index=future_rounds)
    return series, forecast_series

# -----------------------------------------------------------------------------
# MAIN APP UI
# -----------------------------------------------------------------------------

def main():
    st.sidebar.title("CESIM Dashboard")
    
    # 1. FILE UPLOAD
    uploaded_files = st.sidebar.file_uploader(
        "Upload Round Results", 
        type=['csv', 'xlsx'], 
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Please upload 'results-irXX.csv' files to begin.")
        st.markdown("""
        **Expected Format:**
        * CSV files with block headers (e.g., "Income statement, k USD, Global").
        * Team names in columns.
        * Filenames should contain round number (e.g., `results-ir01.csv`).
        """)
        return

    # 2. PARSE & COMBINE
    all_data = []
    with st.spinner("Parsing files..."):
        for f in uploaded_files:
            try:
                df_round = parse_cesim_file(f)
                all_data.append(df_round)
            except Exception as e:
                st.error(f"Error parsing {f.name}: {e}")
    
    if not all_data:
        st.error("No valid data found.")
        return

    df_full = pd.concat(all_data, ignore_index=True)
    df_full = df_full.sort_values(by="round")
    
    # Global Filters
    st.sidebar.markdown("---")
    st.sidebar.header("Filters")
    
    available_teams = sorted(df_full['team'].unique())
    available_scopes = sorted(df_full['scope'].unique())
    available_rounds = sorted(df_full['round'].unique())
    
    selected_team = st.sidebar.selectbox("Select Team", available_teams, index=0)
    selected_scope = st.sidebar.selectbox("Select Scope", available_scopes, index=0)
    
    # Get latest round data for Summary
    max_round = max(available_rounds)
    
    # TABS
    tab_overview, tab_explorer, tab_prediction = st.tabs(["Overview", "Variable Explorer", "Prediction"])

    # --- TAB 1: OVERVIEW ---
    with tab_overview:
        st.header(f"Overview: {selected_team} ({selected_scope}) - Round {max_round}")
        
        # Filter Data
        df_curr = df_full[
            (df_full['round'] == max_round) & 
            (df_full['team'] == selected_team) & 
            (df_full['scope'] == selected_scope)
        ].set_index('metric')['value_k_usd']

        # Helper to safely get value
        def get_val(metric_part, default=0):
            # Fuzzy match metric name
            matches = [k for k in df_curr.index if metric_part.lower() in k.lower()]
            if matches:
                # Prioritize exact start match or shortest match
                return df_curr[matches[0]]
            return default

        # KPIs
        sales = get_val("Sales revenue")
        profit = get_val("Profit for the round")
        assets = get_val("Total assets")
        equity = get_val("Shareholder's equity") # Common name in CESIM, usually "Shareholder's equity" or just "Total equity"
        if equity == 0: equity = get_val("Total equity")
        
        # Calculations
        ebit = get_val("Operating profit")
        if ebit == 0: ebit = get_val("EBIT")
        
        net_margin = (profit / sales * 100) if sales else 0
        equity_ratio = (equity / assets * 100) if assets else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sales Revenue", f"${sales:,.0f}k")
        col2.metric("Net Profit", f"${profit:,.0f}k", f"{net_margin:.1f}% Margin")
        col3.metric("Total Assets", f"${assets:,.0f}k")
        col4.metric("Equity Ratio", f"{equity_ratio:.1f}%")

        st.subheader("Cost Breakdown")
        # Identify cost metrics
        cost_metrics = [
            "In-house manufacturing", "Feature costs", "Contract manufacturing", 
            "Transportation", "R&D", "Promotion", "Warranty", "Administration"
        ]
        
        cost_data = {}
        for cm in cost_metrics:
            val = get_val(cm)
            if val > 0:
                cost_data[cm] = val
        
        if cost_data:
            df_costs = pd.DataFrame(list(cost_data.items()), columns=['Cost Item', 'Amount'])
            df_costs['% of Sales'] = (df_costs['Amount'] / sales * 100) if sales else 0
            
            c1, c2 = st.columns([2, 1])
            with c1:
                st.bar_chart(df_costs.set_index('Cost Item')['Amount'])
            with c2:
                st.dataframe(df_costs.style.format({'Amount': '{:,.0f}', '% of Sales': '{:.1f}%'}))
        else:
            st.info("No detailed cost data found for this scope.")

    # --- TAB 2: VARIABLE EXPLORER ---
    with tab_explorer:
        st.header("Variable Explorer")
        
        # Dropdown for metric
        all_metrics = sorted(df_full['metric'].unique())
        selected_metric = st.selectbox("Select Metric", all_metrics)
        
        # Prepare Time Series Data
        df_ts = df_full[
            (df_full['team'] == selected_team) & 
            (df_full['scope'] == selected_scope) & 
            (df_full['metric'] == selected_metric)
        ].sort_values(by='round')
        
        if not df_ts.empty:
            col_a, col_b = st.columns([3, 1])
            
            with col_a:
                st.subheader(f"History: {selected_metric}")
                st.line_chart(df_ts.set_index('round')['value_k_usd'])
            
            with col_b:
                st.subheader("Values")
                st.dataframe(df_ts[['round', 'value_k_usd']].style.format("{:,.0f}"))
                
                # Compare to Total Sales (if it's not sales itself)
                if "Sales" not in selected_metric:
                    # Get sales for the same rounds
                    df_sales = df_full[
                        (df_full['team'] == selected_team) & 
                        (df_full['scope'] == selected_scope) & 
                        (df_full['metric'].str.contains("Sales revenue"))
                    ]
                    if not df_sales.empty:
                        # Merge to calc %
                        merged = pd.merge(df_ts, df_sales, on='round', suffixes=('_item', '_sales'))
                        merged['% of Sales'] = merged['value_k_usd_item'] / merged['value_k_usd_sales'] * 100
                        st.write("vs Sales Revenue:")
                        st.dataframe(merged[['round', '% of Sales']].style.format("{:.2f}%"))

    # --- TAB 3: PREDICTION ---
    with tab_prediction:
        st.header("Forecast Engine")
        
        col_p1, col_p2, col_p3 = st.columns(3)
        pred_metric = col_p1.selectbox("Metric to Forecast", all_metrics, index=0)
        
        default_model = get_default_model(pred_metric)
        model_options = ["naive", "moving_average", "linear_trend", "exp_smoothing", "drift"]
        
        pred_model = col_p2.selectbox(
            "Model Type", 
            model_options, 
            index=model_options.index(default_model) if default_model in model_options else 0
        )
        
        horizon = col_p3.slider("Forecast Horizon (Rounds)", 1, 5, 3)
        
        # Get Data
        df_hist = df_full[
            (df_full['team'] == selected_team) & 
            (df_full['scope'] == selected_scope) & 
            (df_full['metric'] == pred_metric)
        ].set_index('round')['value_k_usd']
        
        if not df_hist.empty:
            hist_series, forecast_series = forecast_series(df_hist, pred_model, horizon)
            
            # Combine for plotting
            combined_df = pd.DataFrame({
                "Historical": hist_series,
                "Forecast": forecast_series
            })
            
            st.subheader(f"Projection: {pred_metric}")
            st.line_chart(combined_df)
            
            st.write(f"**Forecast Values ({pred_model}):**")
            st.dataframe(forecast_series.to_frame(name="Forecast Value").style.format("{:,.0f}"))
            
            st.markdown("""
            *Note: Simple statistical models used. Linear Trend assumes constant growth. Exponential Smoothing weighs recent data more heavily. Drift assumes the trend continues at the average historical rate.*
            """)
        else:
            st.warning("Insufficient data for forecasting.")

if __name__ == "__main__":
    main()
