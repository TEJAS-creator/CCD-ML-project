"""
Data Preprocessing Module for Customer Segmentation Project.

Provides data cleaning, data type normalization, quality auditing, and stats reporting.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd


def clean_transaction_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Cleans e-commerce transaction data based on standard quality filters:
    1. Identifies and removes exact duplicate transaction rows.
    2. Identifies and removes cancelled invoices (InvoiceNo starting with 'C' or 'c').
    3. Identifies and removes invalid CustomerIDs (NaN, empty strings, 'nan', 'null', 'none').
    4. Formats float CustomerIDs (e.g. 12346.0 -> '12346').
    5. Converts Quantity and UnitPrice to numeric and removes non-positive values (Quantity <= 0 or UnitPrice <= 0).
    6. Safely parses InvoiceDate to datetime and drops unparseable dates.
    7. Computes TotalSpend = Quantity * UnitPrice.
    
    Does NOT mutate the input dataframe.
    
    Args:
        df: Input raw pandas DataFrame.
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, int]]:
            - Cleaned DataFrame ready for RFM analysis.
            - Audit statistics dict containing counts for:
              total_loaded, duplicates_removed, cancellations_removed,
              invalid_values_removed, and total_retained.
    """
    if df is None or df.empty:
        empty_stats = {
            "total_loaded": 0,
            "duplicates_removed": 0,
            "cancellations_removed": 0,
            "invalid_values_removed": 0,
            "total_retained": 0
        }
        return pd.DataFrame(), empty_stats

    # Work on an isolated copy to prevent mutating source data
    df_work = df.copy()
    total_loaded = len(df_work)

    # 1. Deduplication
    duplicate_mask = df_work.duplicated(keep="first")
    duplicates_removed = int(duplicate_mask.sum())
    df_work = df_work[~duplicate_mask].copy()

    # 2. Exclude Cancellations (InvoiceNo starting with 'C' or 'c')
    is_cancellation = (
        df_work["InvoiceNo"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.startswith("C")
    )
    cancellations_removed = int(is_cancellation.sum())
    df_work = df_work[~is_cancellation].copy()

    # 3. Clean and normalize CustomerID
    def format_customer_id(val: Any) -> Any:
        if pd.isna(val) or val is None:
            return None
        val_str = str(val).strip()
        if not val_str or val_str.lower() in ["nan", "null", "none", "n/a", "unknown", "nat"]:
            return None
        if val_str.endswith(".0"):
            val_str = val_str[:-2]
        return val_str if val_str else None

    cleaned_cid = df_work["CustomerID"].apply(format_customer_id)
    cid_valid = cleaned_cid.notna()

    # 4. Clean and parse Quantity
    parsed_qty = pd.to_numeric(df_work["Quantity"], errors="coerce")
    qty_valid = parsed_qty.notna() & (parsed_qty > 0)

    # 5. Clean and parse UnitPrice
    parsed_price = pd.to_numeric(df_work["UnitPrice"], errors="coerce")
    price_valid = parsed_price.notna() & (parsed_price > 0)

    # 6. Clean and parse InvoiceDate
    parsed_date = pd.to_datetime(df_work["InvoiceDate"], errors="coerce")
    date_valid = parsed_date.notna()

    # Combine data quality mask
    valid_mask = cid_valid & qty_valid & price_valid & date_valid
    invalid_values_removed = int((~valid_mask).sum())
    
    # Filter retained dataset
    cleaned_df = df_work[valid_mask].copy()

    # Assign converted clean types
    cleaned_df["CustomerID"] = cleaned_cid[valid_mask].values
    cleaned_df["Quantity"] = parsed_qty[valid_mask].values
    cleaned_df["UnitPrice"] = parsed_price[valid_mask].values
    cleaned_df["InvoiceDate"] = parsed_date[valid_mask].values

    # Compute line item spend
    cleaned_df["TotalSpend"] = (cleaned_df["Quantity"] * cleaned_df["UnitPrice"]).round(2)

    total_retained = len(cleaned_df)

    stats = {
        "total_loaded": total_loaded,
        "duplicates_removed": duplicates_removed,
        "cancellations_removed": cancellations_removed,
        "invalid_values_removed": invalid_values_removed,
        "total_retained": total_retained
    }

    return cleaned_df, stats
