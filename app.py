import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import warnings
import re

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
st.set_page_config(page_title="Team Green AI Dashboard", layout="wide")
st.title("🌱 Team Green: AI Decision Engine")

# --- PARSER FOR CESIM CSV/EXCEL ---
def parse_cesim_file(uploaded_file):
    """
    Parses the specific Cesim Results CSV/Excel format.
    Handles UTF-8, Latin-1, and Excel binary formats automatically.
    Extracts metrics for 'Green' (Column index 1).
    """
    df = None
    
    # 1. Attempt to read file with different methods
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin-1')
        except Exception:
            pass
    except Exception:
        pass

    # 2. If CSV failed, try reading as Excel (in case it's .xls/.xlsx)
    if df is None:
        try:
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error parsing {uploaded_file.name}: {e}")
            return None

    try:
        # Helper to find values safely
        def get_value(row_label, col_index=1): # col_index 1 is Green
            # Find row index where column 0 contains the label (case insensitive)
            mask = df.iloc[:, 0].astype(str).str.contains(row_label, case=False, na=False)
            rows = df[mask]
            
            if not rows.empty:
                val = rows.iloc[0, col_index]
                
                # Robust cleaning
                if pd.isna(val) or str(val).lower() == 'nan':
                    return 0.0
                
                try:
                    # Remove all non-numeric characters except dots and minus signs
                    val_str = str(val)
                    # Use regex to keep only digits, dots, and negative signs
                    clean_val = re.sub(r'[^\d.-]', '', val_str)
                    if clean_val == '' or clean_val == '.' or clean_val == '-':
                        return 0.0
                    return float(clean_val)
                except:
                    return 0.0
            return 0.0

        # Extract Round Number (heuristic based on filename)
        filename = uploaded_file.name.lower()
        round_num = 0
        if "ir00" in filename: round_num = 0
        elif "r01" in filename: round_num = 1
        elif "r02" in filename: round_num = 2
        elif "r03" in filename: round_num = 3
        elif "r04" in filename: round_num = 4
        elif "r05" in filename: round_num = 5

        # Extract Metrics for Green
        # Note: We assume 'Green' is at index 1 based on your previous files.
        data = {
            "Round": round_num,
            "Sales Revenue": get_value("Sales revenue"),
            "Operating Profit": get_value("Operating profit"), 
            "Cash": get_value("Cash and cash equivalents"),
            "Market Share": get_value("Global market shares", col_index=1),
            
            # Technology Specifics
            "Price_Combustion": get_value("Selling price", col_index=1), 
            "Features": get_value("Number of offered features", col_index=1),
            "Promo": get_value("Promotion", col_index=1),
            "R&D Spend": get_value("R&D", col_index=1),
            "Capacity Usage": get_value("Capacity usage", col_index=1),
            "Inventory": get_value("Inventory", col_index=1),
            
            # Costs
            "COGS": get_value("Variable production costs", col_index=1),
            "Logistics Cost": get_value("Transportation and tariffs", col_index=1)
        }
        return data
    except Exception as e:
        st.error(f"Error processing data in {uploaded_file.name}: {e}")
        return None

# --- SIDEBAR: DATA UPLOAD ---
st.sidebar.header("📁 Data Management")
uploaded_files = st.sidebar.file_uploader(
    "Upload Result CSVs/Excel", 
    accept_multiple_files=True,
    type=["csv", "xls", "xlsx"]
)

history_df = pd.DataFrame()

if uploaded_files:
    data_list = []
    for f in uploaded_files:
        parsed_data = parse_cesim_file(f)
        if parsed_data:
            data_list.append(parsed_data)
    
    if data_list:
        history_df = pd.DataFrame(data_list).sort_values("Round")
        st.sidebar.success(f"Loaded {len(history_df)} rounds of data.")
    else:
        st.sidebar.warning("Could not parse uploaded files.")
else:
    st.sidebar.info("Upload files to enable analytics. Using default initialization.")

# --- ML ENGINE ---
@st.cache_resource
def train_model(history):
    """
    Trains a model using real history + synthetic data for stability.
    """
    # 1. Base params
    if not history.empty:
        # Fill any missing values in history with sensible defaults to prevent crashes
        safe_history = history.fillna(0)
        
        base_price = safe_history["Price_Combustion"].mean()
        if base_price == 0: base_price = 20000
        
        base_features = safe_history["Features"].mean()
        if base_features == 0: base_features = 3
        
        base_share = safe_history["Market Share"].mean()
        if base_share == 0: base_share = 16.0
    else:
        base_price = 20000
        base_features = 3
        base_share = 16.0

    # 2. Generate Synthetic Training Data
    np.random.seed(42)
    N = 500
    
    prices = np.random.normal(base_price, 3000, N)
    features = np.random.randint(1, 10, N)
    promo = np.random.normal(1500000, 500000, N) # kUSD
    
    # Normalized factors for synthetic demand curve
    p_norm = 1 - (prices / 30000) 
    f_norm = features / 10
    pr_norm = promo / 3000000
    
    est_share = (p_norm * 0.5 + f_norm * 0.3 + pr_norm * 0.2) * 40 
    est_share += np.random.normal(0, 2, N)
    est_share = np.clip(est_share, 5, 50)

    # 3. Combine with Real History
    X_syn = pd.DataFrame({'Price': prices, 'Features': features, 'Promo': promo})
    y_syn = est_share
    
    if not history.empty:
        # Select relevant columns and Drop NaNs to ensure clean training data
        cols = ['Price_Combustion', 'Features', 'Promo', 'Market Share']
        
        # Check if columns exist
        if set(cols).issubset(history.columns):
            # Create a clean dataframe for training
            training_data = history[cols].copy()
            training_data.columns = ['Price', 'Features', 'Promo', 'Target']
            
            # Drop rows where any value is NaN or 0 (assuming Price=0 is invalid)
            training_data = training_data.dropna()
            training_data = training_data[training_data['Price'] > 100] # Basic sanity check
            
            if not training_data.empty:
                X_real = training_data[['Price', 'Features', 'Promo']]
                y_real = training_data['Target']
                
                # Weight real data heavily (50x duplication)
                X_train = pd.concat([X_syn] + [X_real]*50, ignore_index=True)
                y_train = np.concatenate([y_syn, np.tile(y_real, 50)])
            else:
                X_train, y_train = X_syn, y_syn
        else:
             X_train, y_train = X_syn, y_syn
    else:
        X_train, y_train = X_syn, y_syn

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model

# Train Model
model = train_model(history_df)

# --- DASHBOARD TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Historical Performance", "🤖 Decision Simulation", "💡 Strategic Insights"])

# --- TAB 1: HISTORY ---
with tab1:
    if not history_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        last_round = history_df.iloc[-1]
        
        col1.metric("Last Sales Rev", f"${last_round['Sales Revenue']:,.0f}")
        col2.metric("Cash on Hand", f"${last_round['Cash']:,.0f}")
        col3.metric("Market Share", f"{last_round['Market Share']:.1f}%")
        col4.metric("Op. Profit", f"${last_round['Operating Profit']:,.0f}")
        
        # Charts
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Revenue & Profit Trend")
            if len(history_df) > 1:
                fig = px.line(history_df, x="Round", y=["Sales Revenue", "Operating Profit"], markers=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Upload more round data to see trends over time.")
        with c2:
            st.subheader("Market Share Trend")
            if len(history_df) > 0:
                fig = px.bar(history_df, x="Round", y="Market Share", color_discrete_sequence=['#00CC00'])
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload data to see historical trends.")

# --- TAB 2: SIMULATION ---
with tab2:
    st.subheader("Predictive Decision Optimizer")
    
    # Set defaults safely
    if not history_df.empty:
        # Use last round's data as defaults, but check if they are valid (>0)
        last = history_df.iloc[-1]
        def_price = float(last['Price_Combustion']) if last['Price_Combustion'] > 0 else 20000.0
        def_feat = int(last['Features']) if last['Features'] > 0 else 3
        def_promo = float(last['Promo']) if last['Promo'] > 0 else 1000000.0
        
        # Estimate capacity based on usage
        def_cap = 2000.0 # Fallback default
        if last['Capacity Usage'] > 0:
             # Heuristic: If we used 80% capacity to make 1600 units, total cap is ~2000
             # We don't have exact units produced, so we use a safe default or user input
             pass
    else:
        def_price = 20000.0
        def_feat = 3
        def_promo = 1500000.0
        def_cap = 2000.0

    # Input Form
    with st.form("sim_inputs"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            p_price = st.number_input("Selling Price ($)", value=def_price, step=500.0)
            p_prod = st.number_input("Planned Production (k units)", value=1000.0)
        with col_b:
            p_feat = st.slider("Features (1-10)", 1, 10, def_feat)
            p_cap = st.number_input("Total Capacity (k units)", value=def_cap)
        with col_c:
            p_promo = st.number_input("Promotion Budget ($)", value=def_promo, step=100000.0)
            p_rnd = st.number_input("R&D Investment ($)", value=500000.0)
            
        submitted = st.form_submit_button("🚀 Run Monte Carlo Simulation")

    if submitted:
        # 1. Predict Market Share
        pred_share = model.predict([[p_price, p_feat, p_promo]])[0]
        
        # 2. Monte Carlo
        scenarios = []
        for _ in range(500):
            # Uncertainties
            u_share = np.random.normal(pred_share, 2.0) # +/- 2% share volatility
            u_market_growth = np.random.normal(1.10, 0.05) # 10% growth +/- 5%
            
            # Calculations
            # Assume base market size ~10,000k units (simplified for demo)
            market_size = 10000 * u_market_growth 
            demand = market_size * (u_share / 100.0) * 0.16 # Assuming 6 teams
            
            sales = min(demand, p_prod)
            revenue = sales * p_price
            
            # Costs
            # U-shaped curve logic: optimal utilization is 90%
            util = p_prod / p_cap if p_cap > 0 else 0
            cost_mult = 1.0 + (abs(0.9 - util) * 0.5) # Penalty for deviation
            base_var_cost = 15000 # Placeholder base cost
            cogs = sales * base_var_cost * cost_mult
            
            # Fixed
            opex = p_promo + p_rnd + 2000000 # Base admin
            
            profit = revenue - cogs - opex
            scenarios.append(profit)
            
        # 3. Results
        profits = np.array(scenarios)
        mean_profit = profits.mean()
        var_95 = np.percentile(profits, 5)
        
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Market Share", f"{pred_share:.2f}%")
        r2.metric("Expected Profit", f"${mean_profit:,.0f}")
        r3.metric("Risk (VaR 95%)", f"${var_95:,.0f}", delta_color="inverse")
        
        # Histogram
        fig_hist = px.histogram(x=profits, nbins=30, title="Profit Distribution (500 Scenarios)",
                               labels={'x': 'Profit ($)'}, color_discrete_sequence=['green'])
        fig_hist.add_vline(x=mean_profit, line_dash="dash", line_color="black", annotation_text="Mean")
        fig_hist.add_vline(x=var_95, line_dash="dash", line_color="red", annotation_text="Risk Floor")
        st.plotly_chart(fig_hist, use_container_width=True)

# --- TAB 3: INSIGHTS ---
with tab3:
    st.subheader("Nuanced Strategy Guide")
    st.markdown("""
    **Optimization Checklist:**
    1.  **Capacity:** Check 'Capacity Usage' in History. If <80% or >100%, adjust 'Planned Production'.
    2.  **Cash:** Ensure `Cash > $2,000,000` to avoid emergency debt.
    3.  **Features vs. Cost:** Increasing features boosts demand but increases R&D and Variable Costs. Use the simulator to find if the extra demand pays for the cost.
    """)
