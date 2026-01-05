# Cesim Global Challenge - Team Green AI Dashboard

This dashboard is customized for **Team Green**. It ingests your simulation result files (CSVs), rebuilds your historical performance, and uses Machine Learning to predict future demand and profitability.

## Features
- **Auto-Parsing:** Automatically detects key metrics (Cash, Sales, Capacity, Market Share) from the specific Cesim CSV format.
- **Multi-Round Support:** Upload multiple result files (e.g., Round 0, Round 1...) to improve prediction accuracy.
- **Predictive Engine:** Uses a Random Forest model (trained on your history + synthetic augmentation) to forecast demand based on your inputs.
- **Monte Carlo Simulation:** Stress-tests your decisions against 500+ market scenarios to calculate Risk (VaR) and Expected Profit.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
