import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CESIM Analytics Dashboard",
    page_icon="xC",
    layout="wide"
)

# Default Team Names fallback (Parsing logic will try to detect them dynamically first)
DEFAULT_TEAMS = ["Green", "Red", "Blue", "Orange", "Grey", "Ochre"]

# -----------------------------------------------------------------------------
# PARSING LOGIC
# -----------------------------------------------------------------------------

def clean_cesim_number(val):
    """
    Cleans CESIM numeric strings with Indian-style formatting.
    Example: "4,13,52,625" -> 41352625.0
    Example: "-2,74,308" -> -274308.0
    """
    if pd.isna(val) or val == "":
        return 0.0
    
    val_str = str(val)
    # Remove standard thousands separators, indian separators, currency symbols, spaces
    # Keep only digits, '.' and '-'
    clean_str = re.sub(r'[^\d.-]', '', val_str)
    
    try:
        # Handle cases where result is just "-" or empty
        if not clean_str or clean_str == "-":
            return 0.0
        return float(clean_str)
    except ValueError:
        return 0.0

def extract_round_from_filename(filename):
    """
    Extracts round number from filename using regex.
    Example: "results-ir01.csv" -> 1
    """
    # Find all sequences of digits
    digits = re.findall(r'\d+', filename)
    if digits:
        # Usually the last number is the round number in standard CESIM exports
        return int(digits[-1])
    return 0

def parse_single_file(uploaded_file):
    """
    Parses a single CESIM CSV file into a tidy list of dictionaries.
    """
    filename = uploaded_file.name
    round_num = extract_round_from_filename(filename)
    
    # Read raw content. CESIM exports are usually UTF-8.
    # We read as string first to handle the complex block structure manually.
    try:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    except UnicodeDecodeError:
        # Fallback for excel-generated CSVs which might be latin-1
        stringio = StringIO(uploaded_file.getvalue().decode("latin-1"))

    lines = stringio.readlines()
    
    parsed_records = []
    
    current_statement_type = "Unknown"
    current_scope = "Global"
    current_team_map = {} # Map column index to team name
    
    # State tracking
    in_block = False
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Split by comma, respecting quotes
        # We use a simple csv reader for the line to handle "1,23,000" correctly
        row_values = next(pd.read_csv(StringIO(line), header=None).iterrows())[1].values
        row_values = [str(x).strip() if pd.notna(x) else "" for x in row_values]
        
        first_col = row_values[0]
        
        # 1. DETECT BLOCK HEADER
        # e.g., "Income statement, k USD, Global" or "Balance sheet, k USD, USA"
        if "Income statement" in first_col or "Balance sheet" in first_col:
            parts = [p.strip() for p in first_col.split(',')]
            
            current_statement_type = parts[0]
            # Try to extract scope (last part usually)
            if len(parts) >= 3:
                current_scope = parts[-1]
            elif len(parts) == 2:
                # Fallback if unit is missing or format differs
                current_scope = parts[-1]
            else:
                current_scope = "Global"
                
            in_block = True
            current_team_map = {} # Reset teams for this new block
            continue
            
        if in_block:
            # 2. DETECT TEAM HEADER ROW
            # A row is a team header if it contains known team names
            # Heuristic: Check if intersection of row values and DEFAULT_TEAMS is non-empty
            potential_teams = [x for x in row_values if x in DEFAULT_TEAMS]
            
            if len(potential_teams) > 0:
                # Map column indices to team names
                for col_idx, val in enumerate(row_values):
                    if val in DEFAULT_TEAMS:
                        current_team_map[col_idx] = val
                continue
            
            # 3. DETECT DATA ROW
            # If we have a team map, and the first column is not empty/numeric-only
            if current_team_map and first_col:
                metric_name = first_col
                
                # Iterate through the columns we identified as teams
                for col_idx, team_name in current_team_map.items():
                    if col_idx < len(row_values):
                        raw_val = row_values[col_idx]
                        clean_val = clean_cesim_number(raw_val)
                        
                        parsed_records.append({
                            "round": round_num,
                            "scope": current_scope,
                            "statement_type": current_statement_type,
                            "metric": metric_name,
                            "team": team_name,
                            "value_k_usd": clean_val
                        })
    
    return parsed_records

def load_all_files(uploaded_files):
    """
    Orchestrates parsing of multiple files and concatenates them.
    """
    all_records = []
    
    for f in uploaded_files:
        try:
            records = parse_single_file(f)
            all_records.extend(records)
        except Exception as e:
            st.error(f"Error parsing file {f.name}: {e}")
            
    if not all_records:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_records)
    
    # Sort for consistent time series
    df = df.sort_values(by=["team", "scope", "metric", "round"])
    return df

# -----------------------------------------------------------------------------
# ANALYTICS & FORECASTING ENGINE
# -----------------------------------------------------------------------------

def get_model_recommendation(metric_name, statement_type):
    """
    Returns the recommended model type based on variable semantics.
    """
    m = metric_name.lower()
    
    # Revenue items -> Linear Trend
    if "sales" in m or "revenue" in m or "turnover" in m:
        return "linear_trend"
        
    # Cost items -> Exponential Smoothing (often stable growth or level shifts)
    if any(x in m for x in ["cost", "expense", "r&d", "promotion", "warranty", "admin", "manufacturing"]):
        return "exp_smoothing"
        
    # Balance Sheet / Stocks -> Drift (Random walk)
    if any(x in m for x in ["asset", "inventory", "cash", "receivable", "debt", "equity", "liability"]):
        return "drift"
        
    # Ratios / Margins -> Moving Average (smooth out volatility)
    if "margin" in m or "ratio" in m or "%" in m:
        return "moving_average"
        
    # Profitability -> Linear Trend (assume goal is growth)
    if "profit" in m or "ebit" in m or "result" in m:
        return "linear_trend"

    return "naive"

def forecast_series(series, model_type="auto", horizon=3):
    """
    Generates forecast for a time series.
    series: pd.Series indexed by round (int).
    model_type: specific model to use.
    horizon: number of rounds to forecast.
    
    Returns: Tuple (historical_series, forecast_series)
    """
    if series.empty:
        return series, pd.Series(dtype=float)

    last_round = series.index.max()
    future_rounds = np.arange(last_round + 1, last_round + 1 + horizon)
    
    values = series.values
    predictions = []
    
    # Model Logic
    if model_type == "naive":
        # Forecast = Last observed value
        predictions = [values[-1]] * horizon
        
    elif model_type == "moving_average":
        # Forecast = Average of last N periods (e.g., 3)
        window = 3
        if len(values) < window:
            avg = np.mean(values)
        else:
            avg = np.mean(values[-window:])
        predictions = [avg] * horizon
        
    elif model_type == "linear_trend":
        if len(values) > 1:
            X = np.array(series.index).reshape(-1, 1)
            y = values
            reg = LinearRegression().fit(X, y)
            X_future = future_rounds.reshape(-1, 1)
            predictions = reg.predict(X_future).flatten()
        else:
            # Fallback to naive if not enough points
            predictions = [values[-1]] * horizon
            
    elif model_type == "exp_smoothing":
        # Simple Exponential Smoothing
        # Level_t = alpha * Value_t + (1-alpha) * Level_t-1
        alpha = 0.5 # Smoothing factor
        level = values[0]
        for v in values[1:]:
            level = alpha * v + (1 - alpha) * level
        # Forecast is flat line at last level
        predictions = [level] * horizon
        
    elif model_type == "drift":
        # Random Walk with Drift: Forecast = Last + (Average Change * n)
        if len(values) > 1:
            diffs = np.diff(values)
            avg_drift = np.mean(diffs)
            last_val = values[-1]
            predictions = [last_val + avg_drift * (i+1) for i in range(horizon)]
        else:
            predictions = [values[-1]] * horizon
            
    else:
        # Default fallback
        predictions = [values[-1]] * horizon

    forecast_ser = pd.Series(predictions, index=future_rounds)
    return series, forecast_ser

def calculate_extra_kpis(df):
    """
    Calculates derived metrics (margins, ratios) and appends to dataframe.
    Note: Doing this properly requires pivoting the dataframe to have metrics as columns.
    For this robust app, we will calculate these on the fly in the UI to avoid 
    complicating the tidy data structure in the main storage.
    """
    return df

# -----------------------------------------------------------------------------
# MAIN UI
# -----------------------------------------------------------------------------

def main():
    st.sidebar.header("Data Upload")
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload CESIM Results (CSV/Excel)", 
        type=["csv", "xlsx"], 
        accept_multiple_files=True
    )
    
    if not uploaded_files:
        st.info("👋 Welcome! Please upload your 'results-irXX.csv' files in the sidebar to begin.")
        st.markdown("""
        ### How to use this dashboard:
        1.  **Export** your Round Results from CESIM (CSV format).
        2.  **Upload** them in the sidebar (you can upload multiple rounds at once).
        3.  **Analyze** your team's performance in the tabs.
        
        **Note:** The app supports the Indian-style number formatting (e.g., `4,13,52,625`) natively.
        """)
        return

    # 1. LOAD DATA
    with st.spinner("Parsing and consolidating round data..."):
        df = load_all_files(uploaded_files)
    
    if df.empty:
        st.error("Could not parse any data. Please check your file format.")
        return
        
    # 2. GLOBAL FILTERS
    st.sidebar.markdown("---")
    st.sidebar.header("Analysis Filters")
    
    available_teams = sorted(list(df['team'].unique()))
    default_team_idx = 0
    if "Green" in available_teams:
        default_team_idx = available_teams.index("Green")
        
    selected_team = st.sidebar.selectbox("Select Team", available_teams, index=default_team_idx)
    
    available_scopes = sorted(list(df['scope'].unique()))
    selected_scope = st.sidebar.selectbox("Select Scope (Region)", available_scopes)
    
    # 3. MAIN TABS
    tab_overview, tab_explorer, tab_prediction = st.tabs(["📊 Overview", "📈 Variable Explorer", "🔮 Prediction"])
    
    # Current round info (max round)
    max_round = df['round'].max()
    
    # --- TAB: OVERVIEW ---
    with tab_overview:
        st.title(f"Overview: {selected_team} - {selected_scope}")
        st.caption(f"Data based on Round {max_round}")
        
        # Filter for current snapshot
        df_curr = df[
            (df['team'] == selected_team) & 
            (df['scope'] == selected_scope) & 
            (df['round'] == max_round)
        ]
        
        if df_curr.empty:
            st.warning("No data found for this combination of Team/Scope/Round.")
        else:
            # Create a dictionary for easy O(1) access to metrics
            metrics_map = dict(zip(df_curr['metric'], df_curr['value_k_usd']))
            
            # Helper to safely get metric (fuzzy match)
            def get_val(key_fragment):
                for k, v in metrics_map.items():
                    if key_fragment.lower() in k.lower():
                        return v
                return 0.0

            # KPI CARDS
            col1, col2, col3, col4 = st.columns(4)
            
            sales = get_val("Sales revenue")
            profit = get_val("Profit for the round")
            assets = get_val("Total assets")
            equity = get_val("Shareholder's equity")
            if equity == 0: equity = get_val("Total equity") # Fallback
            
            # Derived ratios
            net_margin = (profit / sales * 100) if sales != 0 else 0
            equity_ratio = (equity / assets * 100) if assets != 0 else 0
            
            col1.metric("Sales Revenue", f"${sales:,.0f}k")
            col2.metric("Net Profit", f"${profit:,.0f}k", f"{net_margin:.1f}% Margin")
            col3.metric("Total Assets", f"${assets:,.0f}k")
            col4.metric("Equity Ratio", f"{equity_ratio:.1f}%")
            
            st.markdown("---")
            
            # COST BREAKDOWN CHART
            st.subheader("Cost Structure Analysis")
            
            # Standard CESIM Cost Metrics
            cost_items = [
                "In-house manufacturing",
                "Feature costs",
                "Contract manufacturing",
                "Transportation",
                "R&D",
                "Promotion",
                "Warranty",
                "Administration"
            ]
            
            cost_data = []
            for item in cost_items:
                val = get_val(item)
                if val > 0:
                    cost_data.append({"Cost Item": item, "Value": val})
            
            if cost_data:
                df_costs = pd.DataFrame(cost_data)
                df_costs["% of Sales"] = (df_costs["Value"] / sales * 100) if sales else 0
                
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.bar_chart(df_costs.set_index("Cost Item")["Value"])
                with c2:
                    st.write("Detailed Breakdown:")
                    st.dataframe(df_costs.style.format({"Value": "{:,.0f}", "% of Sales": "{:.1f}%"}))
            else:
                st.info("No detailed cost data available for this scope.")

    # --- TAB: VARIABLE EXPLORER ---
    with tab_explorer:
        st.header("Variable Analysis")
        
        # Filter metrics relevant to the current selection to prevent empty data
        df_scope_specific = df[
            (df['team'] == selected_team) & 
            (df['scope'] == selected_scope)
        ]
        
        available_metrics = sorted(df_scope_specific['metric'].unique())
        
        if not available_metrics:
            st.warning(f"No metrics found for {selected_team} in {selected_scope}. Please check your filters.")
        else:
            selected_metric = st.selectbox("Select Metric to Analyze", available_metrics)
            
            # Filter history
            df_hist = df_scope_specific[
                df_scope_specific['metric'] == selected_metric
            ].sort_values(by="round")
            
            if df_hist.empty:
                st.warning("No historical data for this metric.")
            else:
                col_left, col_right = st.columns([3, 1])
                
                with col_left:
                    st.subheader(f"Trend: {selected_metric}")
                    st.line_chart(df_hist.set_index("round")["value_k_usd"])
                
                with col_right:
                    st.subheader("Data Points")
                    st.dataframe(df_hist[["round", "value_k_usd"]].style.format("{:,.0f}"))
                    
                    # Compare to Sales (if appropriate)
                    if "Sales" not in selected_metric:
                        # Fetch sales for same rounds
                        df_sales = df_scope_specific[
                            df_scope_specific['metric'].str.contains("Sales revenue")
                        ]
                        
                        if not df_sales.empty:
                            st.markdown("**% of Sales**")
                            # Merge on round
                            merged = pd.merge(df_hist, df_sales, on="round", suffixes=("_item", "_sales"))
                            merged["pct"] = merged["value_k_usd_item"] / merged["value_k_usd_sales"] * 100
                            st.dataframe(merged[["round", "pct"]].set_index("round").style.format("{:.2f}%"))

    # --- TAB: PREDICTION ---
    with tab_prediction:
        st.header("Forecast Engine")
        
        p_col1, p_col2, p_col3 = st.columns(3)
        
        # Filter metrics specifically for this tab as well
        df_pred_scope = df[
            (df['team'] == selected_team) & 
            (df['scope'] == selected_scope)
        ]
        available_pred_metrics = sorted(df_pred_scope['metric'].unique())

        if not available_pred_metrics:
            st.warning(f"No metrics available to forecast for {selected_team} in {selected_scope}.")
        else:
            pred_metric = p_col1.selectbox("Metric to Forecast", available_pred_metrics, key="pred_metric")
            
            # Auto-detect model recommendation
            rec_model = "naive"
            # Find statement type for better recommendation
            stmt_series = df[df['metric'] == pred_metric]['statement_type']
            stmt_type = stmt_series.iloc[0] if not stmt_series.empty else "Unknown"
            rec_model = get_model_recommendation(pred_metric, stmt_type)
            
            model_options = ["naive", "moving_average", "linear_trend", "exp_smoothing", "drift"]
            
            # Set index of selectbox to the recommended one
            default_idx = model_options.index(rec_model) if rec_model in model_options else 0
            
            selected_model = p_col2.selectbox(
                f"Model (Recommended: {rec_model})", 
                model_options, 
                index=default_idx
            )
            
            horizon = p_col3.slider("Forecast Horizon (Rounds)", 1, 5, 3)
            
            # Prepare Data
            df_pred_hist = df_pred_scope[
                df_pred_scope['metric'] == pred_metric
            ].set_index("round")["value_k_usd"].sort_index()
            
            # We now allow len >= 1 (even if it's just 1 point, we can do naive forecast)
            if len(df_pred_hist) < 1:
                st.warning("No data found to forecast (0 rows).")
            else:
                hist_series, forecast_vals = forecast_series(df_pred_hist, selected_model, horizon)
                
                # Combine for Charting
                # We construct a dataframe with index covering both history and future
                all_indices = sorted(list(set(hist_series.index) | set(forecast_vals.index)))
                chart_df = pd.DataFrame(index=all_indices)
                chart_df["Historical"] = hist_series
                chart_df["Forecast"] = forecast_vals
                
                st.subheader(f"Projection: {pred_metric}")
                st.line_chart(chart_df)
                
                st.markdown("### Forecast Values")
                
                # Formatting forecast for display
                disp_df = forecast_vals.to_frame(name="Forecast Value")
                disp_df.index.name = "Round"
                st.dataframe(disp_df.style.format("{:,.0f}"))
                
                st.info(f"""
                **Model Logic Used:**
                * **{selected_model}**: {
                    "Assumes last value persists." if selected_model == 'naive' else
                    "Averages recent history." if selected_model == 'moving_average' else
                    "Fits a straight line trend." if selected_model == 'linear_trend' else
                    "Weighs recent data more heavily." if selected_model == 'exp_smoothing' else
                    "Projects the average historical change forward."
                }
                """)

if __name__ == "__main__":
    main()
