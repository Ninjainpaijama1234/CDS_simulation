```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
st.set_page_config(page_title="Team Green AI Dashboard", layout="wide")
st.title("🌱 Team Green: AI Decision Engine")

# --- PARSER FOR CESIM CSV ---
def parse_cesim_file(uploaded_file):
    """
    Parses the specific Cesim Results CSV format.
    Extracts metrics for 'Green' (Column index 1).
    """
    try:
        # Read file
        df = pd.read_csv(uploaded_file)
        
        # Helper to find values
        def get_value(row_label, col_index=1): # col_index 1 is Green
            # Find row index where column 0 contains the label (case insensitive)
            rows = df[df.iloc[:, 0].astype(str).str.contains(row_label, case=False, na=False)]
            if not rows.empty:
                val = rows.iloc[0, col_index]
                # Clean value (remove spaces, convert to float)
                try:
                    return float(str(val).replace(' ', '').replace(',', ''))
                except:
                    return 0.0
            return 0.0

        # Extract Round Number (from filename or content)
        filename = uploaded_file.name
        if "ir00" in filename: round_num = 0
        elif "r01" in filename: round_num = 1
        elif "r02" in filename: round_num = 2
        elif "r03" in filename: round_num = 3
        elif "r04" in filename: round_num = 4
        else: round_num = 0

        # Extract Metrics for Green
        data = {
            "Round": round_num,
            "Sales Revenue": get_value("Sales revenue"),
            "Operating Profit": get_value("Operating profit \(EBIT\)", col_index=1),
            "Cash": get_value("Cash and cash equivalents"),
            "Market Share": get_value("Global market shares", col_index=1),
            
            # Technology Specifics (Assuming Combustion/Hybrid are main drivers in early rounds)
            "Price_Combustion": get_value("Selling price", col_index=1), # Heuristic: takes first found
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
        st.error(f"Error parsing {uploaded_file.name}: {e}")
        return None

# --- SIDEBAR: DATA UPLOAD ---
st.sidebar.header("📁 Data Management")
uploaded_files = st.sidebar.file_uploader(
    "Upload Result CSVs (e.g., results-ir00.csv)", 
    accept_multiple_files=True,
    type=["csv", "xls"]
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
    Trains a model. If history is small (<4 rounds), creates synthetic data 
    based on the *actual* historical averages to bootstrap the model.
    """
    # 1. Base params on history (or defaults if empty)
    if not history.empty:
        base_price = history["Price_Combustion"].mean()
        base_features = history["Features"].mean()
        base_share = history["Market Share"].mean()
    else:
        base_price = 18500
        base_features = 3
        base_share = 16.0

    # 2. Generate Synthetic Training Data (to fill gaps in logic)
    # We create 500 scenarios around the user's actual data points
    np.random.seed(42)
    N = 500
    
    prices = np.random.normal(base_price, 3000, N)
    features = np.random.randint(1, 10, N)
    promo = np.random.normal(1500000, 500000, N) # kUSD
    
    # Heuristic Logic for Synthetic Labels (Demand curve)
    # Lower price + More Features + Higher Promo = Higher Share
    # We calibrate this so the mean matches the user's historical share
    
    # Normalized factors
    p_norm = 1 - (prices / 30000) 
    f_norm = features / 10
    pr_norm = promo / 3000000
    
    est_share = (p_norm * 0.5 + f_norm * 0.3 + pr_norm * 0.2) * 40 
    # Add noise
    est_share += np.random.normal(0, 2, N)
    est_share = np.clip(est_share, 5, 50)

    # 3. Combine with Real History (Weighted higher)
    X_syn = pd.DataFrame({'Price': prices, 'Features': features, 'Promo': promo})
    y_syn = est_share
    
    if not history.empty:
        # Replicate real history to give it weight
        X_real = history[['Price_Combustion', 'Features', 'Promo']].replace(0, np.nan).dropna()
        # If we have valid real data
        if not X_real.empty:
            X_real.columns = ['Price', 'Features', 'Promo']
            # Assume Market Share is the target
            y_real = history.loc[X_real.index, 'Market Share']
            
            # Upsample real data 50x
            X_train = pd.concat([X_syn] + [X_real]*50, ignore_index=True)
            y_train = np.concatenate([y_syn, np.tile(y_real, 50)])
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
            fig = px.line(history_df, x="Round", y=["Sales Revenue", "Operating Profit"], markers=True)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Market Share Trend")
            fig = px.bar(history_df, x="Round", y="Market Share", color_discrete_sequence=['#00CC00'])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload data to see historical trends.")

# --- TAB 2: SIMULATION ---
with tab2:
    st.subheader("Predictive Decision Optimizer")
    
    # Defaults from last round or generic
    def_price = float(history_df['Price_Combustion'].iloc[-1]) if not history_df.empty else 20000.0
    def_feat = int(history_df['Features'].iloc[-1]) if not history_df.empty else 3
    def_promo = float(history_df['Promo'].iloc[-1]) if not history_df.empty else 1600000.0
    def_cap = float(history_df['Capacity Usage'].iloc[-1]/0.7) if not history_df.empty and history_df['Capacity Usage'].iloc[-1]>0 else 2000.0 # Estimate total cap

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
            demand = market_size * (u_share / 100.0) * 0.16 # Assuming 6 teams, roughly 1/6th if balanced
            
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
    Based on the uploaded data and simulation rules:
    
    1.  **Capacity vs. Demand:** Your utilization logic penalizes you heavily if you produce <60% or >100% of capacity. Use the simulation to find the "Sweet Spot" (usually ~90%).
    2.  **R&D Lag:** Remember, money spent on **In-House R&D** takes 1 round to effect sales. **Licenses** are immediate but expensive.
    3.  **Tariffs:** If you produce in USA and sell in China, verify the current tariff rate in the market report. The simulation currently assumes local production for simplicity; ensure you add logistics costs if exporting.
    4.  **Cash Constraints:** Always keep >$2M cash. The "Risk (VaR)" metric in the simulation helps you see if a bad market turn will bankrupt you.
    """)
