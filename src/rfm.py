"""
RFM Analysis Module for Customer Segmentation Project.

Computes Recency, Frequency, and Monetary (RFM) metrics per customer:
1. Recency: Number of days between a customer's latest purchase and the snapshot reference date.
   - Reference Date Standard: Set to 1 day after the maximum InvoiceDate in the dataset.
   - Rationale: Ensures the most recent buyer has a Recency score of 1 day (rather than 0),
     avoiding zero-division issues in log transforms while keeping all values strictly non-negative.
2. Frequency: Number of unique valid invoices (InvoiceNo) placed by the customer.
3. Monetary: Total spending aggregated across all valid transactions (Quantity * UnitPrice).
"""

from typing import Union, Optional, Tuple, Dict
from datetime import datetime
import numpy as np
import pandas as pd


REQUIRED_RFM_COLUMNS = ["CustomerID", "InvoiceNo", "InvoiceDate"]


def calculate_rfm(
    df: pd.DataFrame, 
    reference_date: Optional[Union[datetime, pd.Timestamp, str]] = None
) -> pd.DataFrame:
    """
    Computes Recency, Frequency, and Monetary metrics per CustomerID.
    
    Args:
        df: Input cleaned transaction DataFrame.
            Must contain columns: CustomerID, InvoiceNo, InvoiceDate, and either TotalSpend or (Quantity, UnitPrice).
        reference_date: Optional custom snapshot reference date.
                        If None, defaults to max(InvoiceDate) + 1 day.
                        
    Returns:
        pd.DataFrame: DataFrame with unique CustomerID as rows and columns [CustomerID, Recency, Frequency, Monetary].
        
    Raises:
        ValueError: If essential columns are missing or input is not a DataFrame.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input dataset must be a valid pandas DataFrame.")

    if df.empty:
        return pd.DataFrame(columns=["CustomerID", "Recency", "Frequency", "Monetary"])

    # Verify column requirements
    missing_cols = [col for col in REQUIRED_RFM_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required columns for RFM calculation: {missing_cols}")

    # Ensure TotalSpend column exists
    df_work = df.copy()
    if "TotalSpend" not in df_work.columns:
        if "Quantity" in df_work.columns and "UnitPrice" in df_work.columns:
            df_work["TotalSpend"] = (
                pd.to_numeric(df_work["Quantity"], errors="coerce") * 
                pd.to_numeric(df_work["UnitPrice"], errors="coerce")
            )
        else:
            raise ValueError("Input DataFrame must contain 'TotalSpend' or both 'Quantity' and 'UnitPrice'.")

    # Safely convert InvoiceDate to pandas Timestamp
    df_work["InvoiceDate"] = pd.to_datetime(df_work["InvoiceDate"], errors="coerce")
    
    # Drop rows missing critical values for grouping
    df_work = df_work.dropna(subset=["CustomerID", "InvoiceDate", "InvoiceNo", "TotalSpend"])
    if df_work.empty:
        return pd.DataFrame(columns=["CustomerID", "Recency", "Frequency", "Monetary"])

    # Determine reference date (snapshot date)
    max_date = df_work["InvoiceDate"].max()
    if reference_date is None:
        # Reference date set to 1 day after the latest transaction date in the dataset
        ref_timestamp = max_date + pd.Timedelta(days=1)
    else:
        ref_timestamp = pd.to_datetime(reference_date)
        if ref_timestamp < max_date:
            # Adjust if reference date is earlier than max transaction date
            ref_timestamp = max_date + pd.Timedelta(days=1)

    # Aggregate by CustomerID
    rfm = (
        df_work.groupby("CustomerID", as_index=False)
        .agg(
            LatestPurchase=("InvoiceDate", "max"),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalSpend", "sum")
        )
    )

    # Compute Recency in days
    rfm["Recency"] = (ref_timestamp - rfm["LatestPurchase"]).dt.total_seconds() / (24 * 3600)
    rfm["Recency"] = rfm["Recency"].astype(int)
    
    # Ensure non-negative Recency
    rfm["Recency"] = np.maximum(rfm["Recency"], 0)
    
    # Round Monetary and ensure non-negative
    rfm["Monetary"] = rfm["Monetary"].round(2)
    rfm["Monetary"] = np.maximum(rfm["Monetary"], 0.0)

    # Reorder columns
    rfm_result = rfm[["CustomerID", "Recency", "Frequency", "Monetary"]].copy()

    return rfm_result


def get_rfm_summary_statistics(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes summary descriptive statistics (mean, std, min, median, max) for RFM features.
    
    Args:
        rfm_df: DataFrame containing [Recency, Frequency, Monetary].
        
    Returns:
        pd.DataFrame: Formatted summary statistics table.
    """
    if rfm_df is None or rfm_df.empty:
        return pd.DataFrame()

    stats = rfm_df[["Recency", "Frequency", "Monetary"]].describe().T
    stats = stats[["mean", "std", "min", "50%", "max"]].copy()
    stats.columns = ["Mean", "Std Dev", "Min", "Median (50%)", "Max"]
    stats["Mean"] = stats["Mean"].round(2)
    stats["Std Dev"] = stats["Std Dev"].round(2)
    stats["Min"] = stats["Min"].round(2)
    stats["Median (50%)"] = stats["Median (50%)"].round(2)
    stats["Max"] = stats["Max"].round(2)
    
    return stats
