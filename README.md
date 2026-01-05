# CESIM Simulation Dashboard

A comprehensive Streamlit dashboard for analyzing and forecasting CESIM simulation results. This tool allows teams to upload round results (CSV/Excel), visualize financial performance, analyze trends across rounds, and forecast future metrics using various statistical models.

## Features

* **Multi-Round Support**: Upload multiple result files (e.g., `results-ir00.csv`, `results-ir01.csv`). The app automatically organizes them by round.
* **Robust Parsing**: Handles CESIM's specific CSV structure, including block-based reporting (Income Statement/Balance Sheet per region) and currency formatting (e.g., Indian-style digit grouping).
* **Interactive Overview**: View KPIs, cost structures, and financial ratios for any specific Team, Region (Scope), and Round.
* **Variable Explorer**: Deep dive into specific metrics (e.g., R&D, Sales) to see historical trends.
* **Forecasting Engine**:
    * Automatic model selection based on variable type (e.g., Linear Trend for Sales, Exponential Smoothing for Costs, Random Walk for Balance Sheet items).
    * User-overridable model selection.
    * Visualizes historical data alongside future projections.

## Installation

1.  **Clone or download** this repository.
2.  **Install dependencies** using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the application**:

    ```bash
    streamlit run app.py
    ```

2.  **Upload Data**:
    * Open the sidebar.
    * Upload your CESIM `results-irXX.csv` files.
    * *Note*: The app expects filenames to contain digits indicating the round (e.g., `00`, `01`).

3.  **Navigate**:
    * **Overview**: Summary statistics and snapshots of the latest round.
    * **Variable Explorer**: Trend lines for specific metrics across all available rounds.
    * **Prediction**: Forecast future performance for the next N rounds.

## File Format Requirements

The app is designed for CESIM exports with the following characteristics:
* CSV format.
* Section headers like `"Income statement, k USD, Global"`.
* Team names in columns (Green, Red, Blue, etc.).
* Numeric values in strings (e.g., `"4,13,52,625"`).
