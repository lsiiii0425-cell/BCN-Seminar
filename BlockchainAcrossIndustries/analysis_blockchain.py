# Blockchain dataset analysis script (Python)
#
# This script:
#  - reads a blockchain-enabled CSV file (default: 'bcn.csv' in same folder)
#  - cleans and enriches the data (timestamps, GPS, numeric columns, blockchain IDs)
#  - produces a set of illustrative PNG plots into the SAME folder as this script
#  - writes a cleaned copy of the dataset as 'cleaned_bcn.csv' (in the same folder)
#
# No command line arguments are required; just run the file in PyCharm or from terminal:
#   python analysis_blockchain.py
#
# Requires: pandas, numpy, matplotlib, seaborn, networkx, scikit-learn

import os
from pathlib import Path

from datetime import datetime  # not strictly needed anymore but kept if you re-add reports

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


# -------------------------
# Utility functions
# -------------------------
def parse_gps(coord):
    """Parse a GPS coordinate string into (lat, lon) floats."""
    if pd.isna(coord):
        return (np.nan, np.nan)
    try:
        s = str(coord).strip()
        s = s.replace("(", "").replace(")", "")
        parts = [p.strip() for p in s.split(",")]
        if len(parts) >= 2:
            lat = float(parts[0])
            lon = float(parts[1])
            return lat, lon
    except Exception:
        pass
    return (np.nan, np.nan)


def safe_to_numeric(series):
    """Convert a pandas Series to numeric, coercing errors to NaN."""
    return pd.to_numeric(series, errors="coerce")


def get_script_dir():
    """Return the directory where this script is located."""
    return Path(__file__).resolve().parent


# -------------------------
# Data loading & cleaning
# -------------------------
def load_and_clean(input_path):
    print(f"Loading {input_path} ...")
    df = pd.read_csv(input_path, low_memory=False)

    # standardize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]

    n0 = len(df)
    print(f"Rows loaded: {n0}, Columns: {len(df.columns)}")

    # Parse timestamp if present
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    else:
        for c in df.columns:
            if "time" in c.lower():
                df["Timestamp"] = pd.to_datetime(df[c], errors="coerce")
                break

    # Parse GPS coordinates to lat / lon
    gps_col = None
    for candidate in ["GPS Coordinates", "GPS", "Coordinates", "Location Coordinates"]:
        if candidate in df.columns:
            gps_col = candidate
            break
    if gps_col is None:
        for c in df.columns:
            if "gps" in c.lower() or "coord" in c.lower():
                gps_col = c
                break

    if gps_col:
        print(f"Parsing GPS column: {gps_col}")
        latitudes = []
        longitudes = []
        for v in df[gps_col].fillna(""):
            lat, lon = parse_gps(v)
            latitudes.append(lat)
            longitudes.append(lon)
        df["latitude"] = latitudes
        df["longitude"] = longitudes
    else:
        df["latitude"] = np.nan
        df["longitude"] = np.nan

    # Numeric conversions
    for col in ["Temperature", "Humidity", "Order Amount", "Quantity Shipped", "Time to Delivery", "Quantity Mismatch"]:
        if col in df.columns:
            df[col] = safe_to_numeric(df[col])

    # Clean common categorical fields
    if "Order Status" in df.columns:
        df["Order Status"] = df["Order Status"].astype(str).str.strip().replace({"nan": None})
    if "Payment Status" in df.columns:
        df["Payment Status"] = df["Payment Status"].astype(str).str.strip().replace({"nan": None})

    # Extract on-chain id / hex if present (example 0xc11a)
    hex_col = None
    for c in df.columns:
        if df[c].astype(str).str.contains(r"0x[0-9a-fA-F]+", na=False).any():
            hex_col = c
            break
    if hex_col:
        df["onchain_hex"] = df[hex_col].astype(str).str.extract(r"(0x[0-9a-fA-F]+)")[0]
    else:
        df["onchain_hex"] = np.nan

    # Generic fraud/flag detection from columns that look like 'fraud' or using quantity mismatch as fallback
    fraud_cols = [c for c in df.columns if "fraud" in c.lower() or "frau" in c.lower()]
    if fraud_cols:
        df["fraud_flag_any"] = df[fraud_cols].apply(
            lambda row: int(any(v in [1, "1", "True", "true", True] for v in row)), axis=1
        )
    else:
        df["fraud_flag_any"] = df.get("Quantity Mismatch", 0).fillna(0).astype(bool).astype(int)

    # Drop obviously invalid rows (no Transaction ID & no Timestamp)
    if "Transaction ID" in df.columns:
        valid_mask = df["Transaction ID"].notna()
    else:
        valid_mask = df["Timestamp"].notna()
    df = df[valid_mask].copy()

    # Save cleaned snapshot (only extra CSV file, in same folder)
    script_dir = get_script_dir()
    cleaned_path = script_dir / "cleaned_bcn.csv"
    df.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned CSV -> {cleaned_path}")
    return df


# -------------------------
# Exploratory analyses
# -------------------------
def basic_profile(df):
    rows = []
    rows.append(("Total transactions", len(df)))
    if "Order Amount" in df.columns:
        rows.append(("Total order amount", df["Order Amount"].sum(skipna=True)))
        rows.append(("Mean order amount", df["Order Amount"].mean(skipna=True)))
    if "Quantity Shipped" in df.columns:
        rows.append(("Mean quantity shipped", df["Quantity Shipped"].mean(skipna=True)))
    if "Time to Delivery" in df.columns:
        rows.append(("Median time-to-delivery", df["Time to Delivery"].median(skipna=True)))

    summary = pd.DataFrame(rows, columns=["metric", "value"])
    print("Basic profile:")
    print(summary)
    return summary


def timeseries_plots(df):
    if "Timestamp" not in df.columns or df["Timestamp"].isna().all():
        print("No timestamps available for timeseries plots.")
        return

    script_dir = get_script_dir()
    ts = df.set_index("Timestamp").sort_index()

    # 1) Transactions per day + 30-day rolling mean
    daily = ts.resample("D").size()
    if len(daily) > 0:
        plt.figure()
        daily.plot(alpha=0.4, label="Daily count")
        daily.rolling(window=30, min_periods=7).mean().plot(label="30-day rolling mean")
        plt.title("Transactions per day (with 30-day rolling mean)")
        plt.xlabel("Date")
        plt.ylabel("Number of transactions")
        plt.legend()
        plt.tight_layout()
        path = script_dir / "timeseries_transactions_per_day_smooth.png"
        plt.savefig(path)
        plt.close()
        print("Saved", path)

    # 2) Monthly order amount (less noisy than daily)
    if "Order Amount" in ts.columns:
        monthly_amt = ts["Order Amount"].resample("ME").sum()
        if len(monthly_amt) > 0:
            plt.figure()
            monthly_amt.plot()
            plt.title("Monthly order amount")
            plt.xlabel("Date")
            plt.ylabel("Total order amount (per month)")
            plt.tight_layout()
            path = script_dir / "timeseries_order_amount_monthly.png"
            plt.savefig(path)
            plt.close()
            print("Saved", path)

        # Distribution of individual order amounts
        plt.figure()
        sns.histplot(ts["Order Amount"].dropna(), bins=40, kde=True)
        plt.title("Distribution of individual order amounts")
        plt.xlabel("Order amount")
        plt.tight_layout()
        path = script_dir / "hist_order_amount.png"
        plt.savefig(path)
        plt.close()
        print("Saved", path)

    # 3) Transactions per location by month (top 8 locations)
    if "Location" in df.columns:
        df["month"] = df["Timestamp"].dt.to_period("M")
        top_locations = df["Location"].value_counts().nlargest(8).index.tolist()
        pivot = (
            df[df["Location"].isin(top_locations)]
            .groupby(["month", "Location"]).size().unstack(fill_value=0)
        )
        if not pivot.empty:
            plt.figure(figsize=(12, 6))
            pivot.plot(kind="bar", stacked=False)
            plt.title("Transactions per location (monthly, top 8)")
            plt.tight_layout()
            path = script_dir / "timeseries_location_monthly.png"
            plt.savefig(path)
            plt.close()
            print("Saved", path)


def delivery_performance(df):
    script_dir = get_script_dir()

    if "Time to Delivery" not in df.columns:
        print("No 'Time to Delivery' column found.")
        return
    plt.figure()
    sns.histplot(df["Time to Delivery"].dropna(), bins=30, kde=False)
    plt.title("Distribution of Time to Delivery")
    plt.xlabel("Days")
    path = script_dir / "hist_time_to_delivery.png"
    plt.savefig(path)
    plt.close()
    print("Saved", path)

    if "Order Status" in df.columns:
        grp = df.groupby("Order Status")["Time to Delivery"].median().sort_values()
        plt.figure()
        grp.plot(kind="barh")
        plt.title("Median Time to Delivery by Order Status")
        plt.xlabel("Median days")
        path = script_dir / "median_time_by_order_status.png"
        plt.savefig(path)
        plt.close()
        print("Saved", path)


def payment_vs_delivery(df):
    if "Payment Status" not in df.columns or "Order Status" not in df.columns:
        return

    script_dir = get_script_dir()
    ct = pd.crosstab(df["Payment Status"].fillna("Unknown"), df["Order Status"].fillna("Unknown"))
    plt.figure()
    sns.heatmap(ct, annot=True, fmt="d")
    plt.title("Payment Status vs Order Status")
    path = script_dir / "payment_vs_order_status.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print("Saved", path)


def fraud_indicator_by_month_status(df, top_n_status=4):
    """Plot average fraud_flag_any over month by Order Status (top categories)."""
    if "fraud_flag_any" not in df.columns:
        print("No fraud_flag_any column - skipping fraud-over-month plot.")
        return
    if "Order Status" not in df.columns:
        print("No 'Order Status' column - skipping fraud-over-month plot.")
        return

    script_dir = get_script_dir()

    # Ensure we have a month number column
    if "Month_Num" in df.columns:
        month_col = "Month_Num"
    else:
        if "Timestamp" not in df.columns or df["Timestamp"].isna().all():
            print("No Month_Num or usable Timestamp; skipping fraud-over-month plot.")
            return
        df["Month_Num"] = df["Timestamp"].dt.month
        month_col = "Month_Num"

    # Top N order statuses by frequency
    top_statuses = df["Order Status"].value_counts().nlargest(top_n_status).index.tolist()
    df_sub = df[df["Order Status"].isin(top_statuses)].copy()
    if df_sub.empty:
        print("No data for fraud-over-month plot after filtering by top order statuses.")
        return

    grp = (
        df_sub.groupby([month_col, "Order Status"])["fraud_flag_any"]
        .mean()
        .reset_index(name="avg_fraud")
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=grp,
        x=month_col,
        y="avg_fraud",
        hue="Order Status",
        marker="o",
    )
    plt.title(f"Fraud Indicator over {month_col} by Order Status (top {len(top_statuses)} categories)")
    plt.xlabel("Month")
    plt.ylabel("Average Fraud Indicator")
    plt.tight_layout()
    path = script_dir / "fraud_indicator_over_month_by_order_status.png"
    plt.savefig(path)
    plt.close()
    print("Saved", path)




def fraud_indicator_by_month_payment_status(df, top_n_status=4):
    """Plot average fraud_flag_any over month by Payment Status (top categories)."""
    if "fraud_flag_any" not in df.columns:
        print("No fraud_flag_any column - skipping fraud-over-month (payment) plot.")
        return
    if "Payment Status" not in df.columns:
        print("No 'Payment Status' column - skipping fraud-over-month (payment) plot.")
        return

    script_dir = get_script_dir()

    # Ensure we have a month number column
    if "Month_Num" in df.columns:
        month_col = "Month_Num"
    else:
        if "Timestamp" not in df.columns or df["Timestamp"].isna().all():
            print("No Month_Num or usable Timestamp; skipping fraud-over-month (payment) plot.")
            return
        df["Month_Num"] = df["Timestamp"].dt.month
        month_col = "Month_Num"

    # Top N payment statuses by frequency
    top_statuses = df["Payment Status"].value_counts().nlargest(top_n_status).index.tolist()
    df_sub = df[df["Payment Status"].isin(top_statuses)].copy()
    if df_sub.empty:
        print("No data for fraud-over-month (payment) plot after filtering by top statuses.")
        return

    grp = (
        df_sub.groupby([month_col, "Payment Status"])["fraud_flag_any"]
        .mean()
        .reset_index(name="avg_fraud")
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=grp,
        x=month_col,
        y="avg_fraud",
        hue="Payment Status",
        marker="o",
    )
    plt.title(f"Fraud Indicator over {month_col} by Payment Status (top {len(top_statuses)} categories)")
    plt.xlabel("Month")
    plt.ylabel("Average Fraud Indicator")
    plt.tight_layout()
    path = script_dir / "fraud_indicator_over_month_by_payment_status.png"
    plt.savefig(path)
    plt.close()
    print("Saved", path)


def fraud_indicator_by_month_smart_contract(df):
    """Plot average fraud_flag_any over month by Smart Contract Status (blockchain state)."""
    if "fraud_flag_any" not in df.columns:
        print("No fraud_flag_any column - skipping fraud-over-month (smart contract) plot.")
        return

    script_dir = get_script_dir()

    # Find blockchain state column similar to blockchain_state_analysis
    state_col = None
    for c in df.columns:
        unique_vals = df[c].dropna().astype(str).unique()[:10]
        if any(v in ["Active", "Triggered", "Completed"] for v in unique_vals):
            state_col = c
            break

    if not state_col:
        print("No smart contract state column found - skipping fraud-over-month (smart contract) plot.")
        return

    # Ensure we have a month number column
    if "Month_Num" in df.columns:
        month_col = "Month_Num"
    else:
        if "Timestamp" not in df.columns or df["Timestamp"].isna().all():
            print("No Month_Num or usable Timestamp; skipping fraud-over-month (smart contract) plot.")
            return
        df["Month_Num"] = df["Timestamp"].dt.month
        month_col = "Month_Num"

    df_sub = df.dropna(subset=[state_col]).copy()
    if df_sub.empty:
        print("No data for fraud-over-month (smart contract) plot after dropping missing states.")
        return

    grp = (
        df_sub.groupby([month_col, state_col])["fraud_flag_any"]
        .mean()
        .reset_index(name="avg_fraud")
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=grp,
        x=month_col,
        y="avg_fraud",
        hue=state_col,
        marker="o",
    )
    plt.title(f"Fraud Indicator over {month_col} by Smart Contract Status")
    plt.xlabel("Month")
    plt.ylabel("Average Fraud Indicator")
    plt.tight_layout()
    path = script_dir / "fraud_indicator_over_month_by_smart_contract_status.png"
    plt.savefig(path)
    plt.close()
    print("Saved", path)


def blockchain_state_analysis(df):
    script_dir = get_script_dir()

    # Identify a column that looks like a smart-contract state
    state_col = None
    for c in df.columns:
        unique_vals = df[c].dropna().astype(str).unique()[:10]
        if any(v in ["Active", "Triggered", "Completed"] for v in unique_vals):
            state_col = c
            break

    if not state_col:
        print("No explicit blockchain state column found.")
        return

    print("Found blockchain state column:", state_col)

    # 1) Raw counts per state
    states = df[state_col].fillna("Unknown").value_counts()

    plt.figure()
    states.plot(kind="bar")
    plt.title("On-chain state counts")
    plt.xlabel("Smart contract state")
    plt.ylabel("Number of transactions")
    plt.tight_layout()
    path = script_dir / "blockchain_state_counts.png"
    plt.savefig(path)
    plt.close()
    print("Saved", path)

    # 2) Anomaly rate per state, if we have a fraud / anomaly flag
    if "fraud_flag_any" in df.columns:
        rate = df.groupby(state_col)["fraud_flag_any"].mean().sort_values(ascending=False)
        plt.figure()
        rate.plot(kind="bar")
        plt.title("Anomaly rate by smart contract state")
        plt.xlabel("Smart contract state")
        plt.ylabel("Share of transactions flagged as suspicious")
        plt.tight_layout()
        path = script_dir / "anomaly_rate_by_state.png"
        plt.savefig(path)
        plt.close()
        print("Saved", path)

    # 3) Quantity mismatch per state, if available
    if "Quantity Mismatch" in df.columns:
        qm = df.groupby(state_col)["Quantity Mismatch"].mean().sort_values(ascending=False)
        plt.figure()
        qm.plot(kind="bar")
        plt.title("Average quantity mismatch by smart contract state")
        plt.xlabel("Smart contract state")
        plt.ylabel("Mean quantity mismatch")
        plt.tight_layout()
        path = script_dir / "q_mismatch_by_onchain_state.png"
        plt.savefig(path)
        plt.close()
        print("Saved", path)


def network_graph(df, top_n_edges=15):
    """Build a supplier -> customer network and plot a simplified, readable graph."""
    if "Supplier ID" not in df.columns or "Customer ID" not in df.columns:
        print("Supplier ID or Customer ID missing - skipping network graph.")
        return

    script_dir = get_script_dir()

    edges = (
        df.groupby(["Supplier ID", "Customer ID"])
        .size()
        .reset_index(name="weight")
        .sort_values("weight", ascending=False)
    )

    edges = edges.head(top_n_edges)
    if edges.empty:
        print("No edges for network graph.")
        return

    G = nx.DiGraph()
    for _, row in edges.iterrows():
        s = row["Supplier ID"]
        c = row["Customer ID"]
        w = int(row["weight"])
        G.add_edge(s, c, weight=w)

    suppliers = sorted(edges["Supplier ID"].unique().tolist())
    customers = sorted(edges["Customer ID"].unique().tolist())

    pos = {}
    for i, node in enumerate(suppliers):
        pos[node] = (0, i)
    for j, node in enumerate(customers):
        pos[node] = (1, j)

    weights = [G[u][v]["weight"] for u, v in G.edges()]

    import math
    widths = [1.0 + math.log1p(w) for w in weights]

    plt.figure(figsize=(10, 6))
    nx.draw_networkx_nodes(G, pos, nodelist=suppliers, node_color="lightblue", node_size=800, label="Suppliers")
    nx.draw_networkx_nodes(G, pos, nodelist=customers, node_color="lightgreen", node_size=800, label="Customers")
    nx.draw_networkx_labels(G, pos, font_size=8)

    nx.draw_networkx_edges(
        G,
        pos,
        width=widths,
        arrowsize=15,
        alpha=0.6,
        connectionstyle="arc3,rad=0.1",
    )

    plt.axis("off")
    plt.legend(scatterpoints=1, fontsize=8, loc="upper left")
    plt.title("Top supplier–customer relationships (blockchain-logged)")
    path = script_dir / "supplier_customer_network.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print("Saved", path)


def geospatial_snapshot(df):
    script_dir = get_script_dir()

    if df["latitude"].notna().any() and df["longitude"].notna().any():
        base = df.dropna(subset=["latitude", "longitude"])
        sample = base.sample(min(2000, len(base)), random_state=1)
        plt.figure()
        plt.scatter(sample["longitude"], sample["latitude"], s=10, alpha=0.6)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Geographic distribution of transactions (sample)")
        path = script_dir / "geo_scatter_sample.png"
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print("Saved geospatial snapshot")
    else:
        print("No valid lat/lon to create geospatial snapshot.")


def fraud_anomaly_detection(df):
    script_dir = get_script_dir()

    numeric_cols = [
        c for c in ["Order Amount", "Quantity Mismatch", "Quantity Shipped",
                    "Time to Delivery", "Temperature", "Humidity"]
        if c in df.columns
    ]
    if not numeric_cols:
        print("No numeric columns for anomaly detection.")
        return

    X = df[numeric_cols].copy().fillna(df[numeric_cols].median())

    if len(X) < 50:
        print("Too few rows for anomaly detection, skipping.")
        return

    iso = IsolationForest(random_state=1, contamination=0.02)
    preds = iso.fit_predict(X)
    scores = iso.decision_function(X)

    df["anomaly_flag"] = (preds == -1).astype(int)
    df["anomaly_score"] = -scores

    anomalies = df[df["anomaly_flag"] == 1]
    print(f"Detected {len(anomalies)} anomalies.")

    # 1) Distribution of anomaly scores
    plt.figure()
    sns.histplot(df["anomaly_score"], bins=40, kde=True)
    plt.title("Distribution of anomaly scores")
    plt.xlabel("Anomaly score (higher = more unusual)")
    plt.tight_layout()
    path = script_dir / "anomaly_score_distribution.png"
    plt.savefig(path)
    plt.close()
    print("Saved", path)

    # 2) Order amount for normal vs anomalous transactions
    if "Order Amount" in df.columns:
        df_plot = df[["Order Amount", "anomaly_flag"]].copy()
        df_plot["Anomaly label"] = df_plot["anomaly_flag"].map({0: "Normal", 1: "Anomalous"})

        plt.figure(figsize=(8, 6))
        sns.boxplot(
            x="Anomaly label",
            y="Order Amount",
            data=df_plot,
            width=0.5,
            showfliers=False,
        )
        plt.xlabel("")
        plt.ylabel("Order amount")
        plt.title("Order amount distribution: normal vs anomalous transactions")
        plt.tight_layout()
        path = script_dir / "order_amount_by_anomaly_flag.png"
        plt.savefig(path)
        plt.close()
        print("Saved", path)

    # 3) Anomalies by location (top 10)
    if "Location" in df.columns and not anomalies.empty:
        ct = anomalies["Location"].value_counts().head(10)
        if not ct.empty:
            plt.figure()
            ct.plot(kind="bar")
            plt.title("Top locations in anomaly set")
            plt.ylabel("Number of anomalous transactions")
            plt.tight_layout()
            path = script_dir / "anomalies_by_location.png"
            plt.savefig(path)
            plt.close()
            print("Saved", path)


# -------------------------
# Driver
# -------------------------
def analyze(input_path):
    df = load_and_clean(input_path)
    _ = basic_profile(df)
    timeseries_plots(df)
    delivery_performance(df)
    payment_vs_delivery(df)
    fraud_indicator_by_month_status(df)
    fraud_indicator_by_month_payment_status(df)
    fraud_indicator_by_month_smart_contract(df)
    blockchain_state_analysis(df)
    network_graph(df)
    geospatial_snapshot(df)
    fraud_anomaly_detection(df)


# -------------------------
# Simple runner (no CLI arguments, no output folder)
# -------------------------
if __name__ == "__main__":
    script_dir = get_script_dir()
    input_path = script_dir / "bcn.csv"   # CSV file in same folder
    print(f"Using input file: {input_path}")
    analyze(input_path)
