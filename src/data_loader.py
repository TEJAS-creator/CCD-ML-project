"""
Data Loader Module for Customer Segmentation Project.

Provides utilities for:
1. Generating realistic synthetic e-commerce transaction data matching UCI Online Retail schema.
2. Loading CSV datasets with character encoding fallback and malformed CSV handling.
3. Validating required column schema and data quality.
"""

from datetime import datetime, timedelta
from typing import Any, Tuple, List, Union
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["InvoiceNo", "CustomerID", "Quantity", "UnitPrice", "InvoiceDate"]
UCI_SCHEMA_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity", 
    "InvoiceDate", "UnitPrice", "CustomerID", "Country"
]

SYNTHETIC_PRODUCTS = [
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.55),
    ("71053", "WHITE METAL LANTERN", 3.39),
    ("84406B", "CREAM CUPID HEARTS COAT HANGER", 2.75),
    ("84029G", "KNITTED UNION FLAG HOT WATER BOTTLE", 3.75),
    ("22752", "SET 7 BABUSHKA NESTING BOXES", 7.65),
    ("21730", "GLASS STAR FROSTED T-LIGHT HOLDER", 4.25),
    ("22633", "HAND WARMER UNION JACK", 1.85),
    ("22632", "HAND WARMER RED RETROSPOT", 1.85)
]


def generate_synthetic_data(num_customers: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic transaction-level synthetic e-commerce dataset matching UCI Online Retail schema.
    
    Includes distinct purchasing profiles:
    - VIP / Champions: High frequency, high monetary, recent purchases
    - Loyal Customers: Moderate-high frequency, steady spend, recent
    - At-Risk Big Spenders: High spend in the past, no recent purchases
    - New Buyers: Recent 1-2 transactions, moderate spend
    - Casual / Low-Value Shoppers: Infrequent, low monetary spend
    
    Returns:
        pd.DataFrame: Transactional dataset matching standard UCI Online Retail schema.
    """
    np.random.seed(seed)
    ref_date = datetime.now()
    
    records = []
    countries = ["United Kingdom", "Germany", "France", "EIRE", "Spain", "Netherlands"]
    country_weights = [0.75, 0.08, 0.06, 0.05, 0.03, 0.03]
    
    num_vips = int(num_customers * 0.15)
    num_loyals = int(num_customers * 0.25)
    num_at_risk = int(num_customers * 0.15)
    num_new = int(num_customers * 0.20)
    num_casual = num_customers - (num_vips + num_loyals + num_at_risk + num_new)
    
    customer_pool = []
    
    for _ in range(num_vips):
        customer_pool.append({
            "id": f"DEMO-{10000 + len(customer_pool)}",
            "recency_days_range": (1, 30),
            "num_invoices_range": (8, 25),
            "items_per_inv_range": (5, 15),
            "unit_price_multiplier": (1.5, 4.0),
            "qty_range": (3, 20)
        })
        
    for _ in range(num_loyals):
        customer_pool.append({
            "id": f"DEMO-{10000 + len(customer_pool)}",
            "recency_days_range": (5, 60),
            "num_invoices_range": (5, 12),
            "items_per_inv_range": (2, 8),
            "unit_price_multiplier": (1.0, 2.5),
            "qty_range": (2, 10)
        })
        
    for _ in range(num_at_risk):
        customer_pool.append({
            "id": f"DEMO-{10000 + len(customer_pool)}",
            "recency_days_range": (120, 365),
            "num_invoices_range": (4, 10),
            "items_per_inv_range": (3, 10),
            "unit_price_multiplier": (1.2, 3.0),
            "qty_range": (2, 12)
        })
        
    for _ in range(num_new):
        customer_pool.append({
            "id": f"DEMO-{10000 + len(customer_pool)}",
            "recency_days_range": (1, 20),
            "num_invoices_range": (1, 2),
            "items_per_inv_range": (1, 4),
            "unit_price_multiplier": (1.0, 2.0),
            "qty_range": (1, 5)
        })
        
    for _ in range(num_casual):
        customer_pool.append({
            "id": f"DEMO-{10000 + len(customer_pool)}",
            "recency_days_range": (60, 250),
            "num_invoices_range": (1, 3),
            "items_per_inv_range": (1, 3),
            "unit_price_multiplier": (0.8, 1.5),
            "qty_range": (1, 4)
        })
        
    invoice_counter = 500000
    
    for cust in customer_pool:
        cust_id = cust["id"]
        country = np.random.choice(countries, p=country_weights)
        num_invoices = np.random.randint(cust["num_invoices_range"][0], cust["num_invoices_range"][1] + 1)
        
        last_purchase_days_ago = np.random.randint(cust["recency_days_range"][0], cust["recency_days_range"][1] + 1)
        most_recent_date = ref_date - timedelta(days=last_purchase_days_ago)
        
        for inv_idx in range(num_invoices):
            invoice_counter += 1
            inv_no = f"INV{invoice_counter}"
            
            days_offset = (num_invoices - 1 - inv_idx) * np.random.randint(5, 30)
            inv_date = most_recent_date - timedelta(days=days_offset)
            
            num_items = np.random.randint(cust["items_per_inv_range"][0], cust["items_per_inv_range"][1] + 1)
            for _ in range(num_items):
                prod_idx = np.random.randint(0, len(SYNTHETIC_PRODUCTS))
                stock_code, desc, base_price = SYNTHETIC_PRODUCTS[prod_idx]
                
                mult = np.random.uniform(cust["unit_price_multiplier"][0], cust["unit_price_multiplier"][1])
                unit_price = round(float(base_price * mult), 2)
                quantity = int(np.random.randint(cust["qty_range"][0], cust["qty_range"][1] + 1))
                
                records.append({
                    "InvoiceNo": inv_no,
                    "StockCode": stock_code,
                    "Description": desc,
                    "Quantity": quantity,
                    "InvoiceDate": inv_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "UnitPrice": unit_price,
                    "CustomerID": cust_id,
                    "Country": country
                })

    df = pd.DataFrame(records)
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates whether the input dataframe meets schema and quality requirements.
    
    Args:
        df: Input pandas DataFrame.
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list of error or warning messages)
    """
    errors = []
    
    if df is None or df.empty:
        return False, ["Dataset is empty or could not be loaded."]
        
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required column(s): {', '.join(missing_cols)}. Expected schema: {', '.join(REQUIRED_COLUMNS)}")
        return False, errors

    if len(df) < 5:
        errors.append("Dataset has fewer than 5 rows. Insufficient transactions for segmentation analysis.")

    is_valid = len(errors) == 0
    return is_valid, errors


def load_csv_data(file_or_path: Any) -> pd.DataFrame:
    """
    Reads a CSV file from a file path or file-like object with encoding fallback and column normalization.
    Does not mutate the input source.
    
    Args:
        file_or_path: Path string or uploaded file object.
        
    Returns:
        pd.DataFrame: Loaded raw DataFrame with standardized column names.
        
    Raises:
        ValueError: If file is empty or cannot be parsed.
    """
    df = None
    encodings_to_try = ["utf-8", "ISO-8859-1", "latin1", "cp1252"]
    
    for enc in encodings_to_try:
        try:
            if hasattr(file_or_path, "seek"):
                file_or_path.seek(0)
            df = pd.read_csv(file_or_path, encoding=enc)
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
        except pd.errors.EmptyDataError:
            raise ValueError("The provided CSV file is completely empty.")
        except Exception:
            continue

    if df is None:
        raise ValueError("Could not parse the CSV file. Please check file format and encoding.")
        
    if df.empty:
        raise ValueError("The loaded dataset contains 0 rows.")

    # Operate on a copy to prevent side effects
    df = df.copy()
    
    # Strip whitespace from column names
    df.columns = [str(col).strip() for col in df.columns]
    
    # Standardize column mappings if variants are found
    rename_mapping = {}
    for col in df.columns:
        col_lower = col.lower().replace(" ", "").replace("_", "")
        if col_lower in ["invoiceno", "invoice", "transactionid", "orderid"]:
            rename_mapping[col] = "InvoiceNo"
        elif col_lower in ["stockcode", "stock_code", "productcode", "sku"]:
            rename_mapping[col] = "StockCode"
        elif col_lower in ["description", "productname", "itemdescription"]:
            rename_mapping[col] = "Description"
        elif col_lower in ["customerid", "user_id", "userid", "clientid", "customer"]:
            rename_mapping[col] = "CustomerID"
        elif col_lower in ["quantity", "qty", "amount_items"]:
            rename_mapping[col] = "Quantity"
        elif col_lower in ["unitprice", "price", "itemprice", "rate"]:
            rename_mapping[col] = "UnitPrice"
        elif col_lower in ["invoicedate", "date", "ordertimestamp", "timestamp"]:
            rename_mapping[col] = "InvoiceDate"
        elif col_lower in ["country", "location", "region"]:
            rename_mapping[col] = "Country"

    if rename_mapping:
        df = df.rename(columns=rename_mapping)
        
    return df
