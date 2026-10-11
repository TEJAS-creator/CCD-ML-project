"""
Unit Tests for Customer Segmentation Project.

Includes:
- Step 2 Data Ingestion & Cleaning tests
- Step 3 Manual RFM Feature Engineering tests
- Step 4 K-Means Customer Clustering tests
- Step 5 Business Interpretation & Personas Mapping tests
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import generate_synthetic_data, validate_dataset, load_csv_data
from src.preprocessing import clean_transaction_data
from src.rfm import calculate_rfm, get_rfm_summary_statistics
from src.clustering import RFMClusterPipeline, evaluate_cluster_range
from src.segmentation import interpret_rfm_clusters, RECOMMENDED_ACTIONS


# ---------------------------------------------------------------------------
# STEP 2: DATA CLEANING TESTS
# ---------------------------------------------------------------------------

def test_synthetic_data_uci_schema():
    """Verify synthetic data adheres to standard UCI Online Retail schema."""
    df = generate_synthetic_data(num_customers=30, seed=42)
    assert not df.empty
    expected_cols = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected UCI column: {col}"


def test_cleaning_cancelled_invoices():
    """Verify cancelled invoices (InvoiceNo starting with 'C' or 'c') are filtered out."""
    df = pd.DataFrame({
        "InvoiceNo": ["INV1001", "C536379", " c536380 ", "INV1002"],
        "CustomerID": ["12345", "12345", "12346", "12346"],
        "Quantity": [5, -1, -2, 10],
        "UnitPrice": [10.0, 5.0, 5.0, 15.0],
        "InvoiceDate": ["2026-01-01 10:00:00"] * 4
    })
    
    cleaned_df, stats = clean_transaction_data(df)
    
    assert stats["cancellations_removed"] == 2
    assert len(cleaned_df) == 2
    assert "C536379" not in cleaned_df["InvoiceNo"].values
    assert " c536380 " not in cleaned_df["InvoiceNo"].values


def test_cleaning_missing_and_float_customer_ids():
    """Verify missing CustomerIDs are dropped and float CustomerIDs are normalized."""
    df = pd.DataFrame({
        "InvoiceNo": ["INV1", "INV2", "INV3", "INV4", "INV5"],
        "CustomerID": [12345.0, np.nan, "  ", "nan", "DEMO-999"],
        "Quantity": [1, 2, 3, 4, 5],
        "UnitPrice": [10.0, 10.0, 10.0, 10.0, 10.0],
        "InvoiceDate": ["2026-01-01 10:00:00"] * 5
    })
    
    cleaned_df, stats = clean_transaction_data(df)
    
    assert stats["invalid_values_removed"] == 3
    assert len(cleaned_df) == 2
    assert set(cleaned_df["CustomerID"].values) == {"12345", "DEMO-999"}


def test_cleaning_duplicates():
    """Verify exact duplicate rows are removed."""
    df = pd.DataFrame({
        "InvoiceNo": ["INV1", "INV1", "INV2"],
        "CustomerID": ["1001", "1001", "1002"],
        "Quantity": [5, 5, 10],
        "UnitPrice": [2.0, 2.0, 4.0],
        "InvoiceDate": ["2026-01-01 10:00:00"] * 3
    })
    
    cleaned_df, stats = clean_transaction_data(df)
    
    assert stats["duplicates_removed"] == 1
    assert stats["total_retained"] == 2


def test_cleaning_invalid_quantity_and_price():
    """Verify non-positive Quantity and UnitPrice are removed."""
    df = pd.DataFrame({
        "InvoiceNo": ["INV1", "INV2", "INV3", "INV4"],
        "CustomerID": ["C1", "C2", "C3", "C4"],
        "Quantity": [10, 0, -5, 2],
        "UnitPrice": [5.0, 10.0, 2.0, 0.0],
        "InvoiceDate": ["2026-01-01 10:00:00"] * 4
    })
    
    cleaned_df, stats = clean_transaction_data(df)
    
    assert stats["invalid_values_removed"] == 3
    assert len(cleaned_df) == 1
    assert cleaned_df.iloc[0]["CustomerID"] == "C1"


def test_cleaning_malformed_dates():
    """Verify unparseable InvoiceDate entries are dropped."""
    df = pd.DataFrame({
        "InvoiceNo": ["INV1", "INV2"],
        "CustomerID": ["101", "102"],
        "Quantity": [1, 2],
        "UnitPrice": [10.0, 20.0],
        "InvoiceDate": ["2026-01-01 10:00:00", "INVALID_DATE_STRING"]
    })
    
    cleaned_df, stats = clean_transaction_data(df)
    
    assert stats["invalid_values_removed"] == 1
    assert len(cleaned_df) == 1
    assert cleaned_df.iloc[0]["CustomerID"] == "101"


def test_source_dataframe_non_mutation():
    """Verify original input dataframe is not modified during cleaning."""
    original_df = pd.DataFrame({
        "InvoiceNo": ["INV1", "INV2"],
        "CustomerID": [101.0, 102.0],
        "Quantity": [1, 2],
        "UnitPrice": [10.0, 20.0],
        "InvoiceDate": ["2026-01-01", "2026-01-02"]
    })
    df_copy = original_df.copy()
    
    cleaned_df, stats = clean_transaction_data(original_df)
    
    pd.testing.assert_frame_equal(original_df, df_copy)


# ---------------------------------------------------------------------------
# STEP 3: MANUAL RFM FEATURE ENGINEERING TESTS
# ---------------------------------------------------------------------------

def test_manual_rfm_calculation_known_values():
    """Manually verifiable RFM test case."""
    tiny_dataset = pd.DataFrame({
        "CustomerID": ["CUST_A", "CUST_A", "CUST_A", "CUST_B"],
        "InvoiceNo": ["INV-001", "INV-001", "INV-002", "INV-003"],
        "InvoiceDate": [
            "2026-01-01 10:00:00",
            "2026-01-01 10:05:00",
            "2026-01-05 14:00:00",
            "2026-01-02 12:00:00"
        ],
        "Quantity": [2, 1, 1, 4],
        "UnitPrice": [10.0, 5.0, 50.0, 10.0]
    })

    rfm_df = calculate_rfm(tiny_dataset)

    assert len(rfm_df) == 2
    assert set(rfm_df.columns) == {"CustomerID", "Recency", "Frequency", "Monetary"}

    cust_a = rfm_df[rfm_df["CustomerID"] == "CUST_A"].iloc[0]
    cust_b = rfm_df[rfm_df["CustomerID"] == "CUST_B"].iloc[0]

    assert cust_a["Recency"] == 1
    assert cust_a["Frequency"] == 2
    assert cust_a["Monetary"] == 75.0

    assert cust_b["Recency"] == 4
    assert cust_b["Frequency"] == 1
    assert cust_b["Monetary"] == 40.0


def test_rfm_empty_and_invalid_inputs():
    """Verify error handling and empty dataset return types."""
    empty_rfm = calculate_rfm(pd.DataFrame(columns=["CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"]))
    assert empty_rfm.empty
    assert list(empty_rfm.columns) == ["CustomerID", "Recency", "Frequency", "Monetary"]

    try:
        calculate_rfm("invalid_string_input")
        assert False, "Should have raised ValueError for non-DataFrame input"
    except ValueError:
        pass


def test_rfm_summary_statistics():
    """Verify summary statistics calculation function."""
    rfm_data = pd.DataFrame({
        "CustomerID": ["C1", "C2", "C3"],
        "Recency": [10, 20, 30],
        "Frequency": [1, 5, 10],
        "Monetary": [100.0, 500.0, 1000.0]
    })
    
    stats = get_rfm_summary_statistics(rfm_data)
    assert not stats.empty
    assert list(stats.columns) == ["Mean", "Std Dev", "Min", "Median (50%)", "Max"]


# ---------------------------------------------------------------------------
# STEP 4: K-MEANS CUSTOMER CLUSTERING TESTS
# ---------------------------------------------------------------------------

def test_clustering_repeatability_and_shape():
    """Verify K-Means produces deterministic output given random seed and valid shape."""
    rfm_data = pd.DataFrame({
        "CustomerID": [f"C{i}" for i in range(1, 21)],
        "Recency": np.random.randint(1, 100, 20),
        "Frequency": np.random.randint(1, 20, 20),
        "Monetary": np.random.uniform(10.0, 1000.0, 20)
    })

    pipeline1 = RFMClusterPipeline(n_clusters=4, random_state=42)
    res_df1, matrix1 = pipeline1.fit_predict(rfm_data)

    pipeline2 = RFMClusterPipeline(n_clusters=4, random_state=42)
    res_df2, matrix2 = pipeline2.fit_predict(rfm_data)

    assert len(res_df1) == 20
    assert "Cluster" in res_df1.columns
    assert matrix1.shape == (20, 3)

    np.testing.assert_array_equal(res_df1["Cluster"].values, res_df2["Cluster"].values)
    assert set(res_df1["Cluster"].unique()).issubset({0, 1, 2, 3})


def test_clustering_invalid_k():
    """Verify invalid cluster count raises ValueError."""
    try:
        RFMClusterPipeline(n_clusters=1)
        assert False, "Should raise ValueError for n_clusters=1"
    except ValueError:
        pass

    try:
        RFMClusterPipeline(n_clusters=9)
        assert False, "Should raise ValueError for n_clusters=9"
    except ValueError:
        pass


def test_clustering_small_dataset():
    """Verify ValueError is raised if samples < k."""
    tiny_rfm = pd.DataFrame({
        "CustomerID": ["C1", "C2"],
        "Recency": [10, 20],
        "Frequency": [1, 2],
        "Monetary": [100.0, 200.0]
    })

    pipeline = RFMClusterPipeline(n_clusters=4, random_state=42)
    try:
        pipeline.fit_predict(tiny_rfm)
        assert False, "Should raise ValueError when n_samples < k"
    except ValueError as e:
        assert "fewer than the requested" in str(e)


def test_cluster_evaluation_metrics():
    """Verify Elbow and Silhouette evaluation metrics function."""
    rfm_data = pd.DataFrame({
        "CustomerID": [f"C{i}" for i in range(1, 30)],
        "Recency": np.random.randint(1, 100, 29),
        "Frequency": np.random.randint(1, 20, 29),
        "Monetary": np.random.uniform(10.0, 1000.0, 29)
    })

    eval_results = evaluate_cluster_range(rfm_data, k_min=2, k_max=6, random_state=42)

    assert "k" in eval_results
    assert "inertia" in eval_results
    assert "silhouette" in eval_results
    assert eval_results["k"] == [2, 3, 4, 5, 6]
    assert len(eval_results["inertia"]) == 5
    assert len(eval_results["silhouette"]) == 5


# ---------------------------------------------------------------------------
# STEP 5: BUSINESS INTERPRETATION & PERSONA MAPPING TESTS
# ---------------------------------------------------------------------------

def test_dynamic_persona_mapping_not_hardcoded():
    """
    Verify that persona labels are mapped dynamically based on RFM centroids, NOT hardcoded cluster IDs.
    
    Test setup:
    - Cluster 99: High spend ($10,000), high frequency (50 orders), low recency (2 days) -> Should be "Champions"
    - Cluster 55: Low spend ($50), low frequency (1 order), high recency (200 days) -> Should be "Hibernating / Lost Customers"
    """
    rfm_clustered = pd.DataFrame({
        "CustomerID": ["C1", "C2", "C3", "C4"],
        "Recency": [2, 3, 200, 210],
        "Frequency": [50, 45, 1, 1],
        "Monetary": [10000.0, 9500.0, 50.0, 45.0],
        "Cluster": [99, 99, 55, 55]  # Non-standard cluster IDs
    })

    seg_df, summary = interpret_rfm_clusters(rfm_clustered, cluster_col="Cluster")

    assert "Segment" in seg_df.columns
    assert "RecommendedAction" in seg_df.columns

    # Verify cluster 99 mapped to Champions
    c99_label = seg_df[seg_df["Cluster"] == 99].iloc[0]["Segment"]
    c55_label = seg_df[seg_df["Cluster"] == 55].iloc[0]["Segment"]

    assert c99_label == "Champions", f"Expected Cluster 99 to be 'Champions', got '{c99_label}'"
    assert c55_label == "Hibernating / Lost Customers", f"Expected Cluster 55 to be 'Hibernating / Lost Customers', got '{c55_label}'"


def test_persona_summary_table_structure():
    """Verify summary table contains required columns and metrics."""
    rfm_clustered = pd.DataFrame({
        "CustomerID": [f"C{i}" for i in range(1, 11)],
        "Recency": [5, 10, 15, 20, 100, 110, 120, 200, 210, 220],
        "Frequency": [20, 18, 15, 12, 5, 4, 3, 1, 1, 1],
        "Monetary": [2000.0, 1800.0, 1500.0, 1200.0, 500.0, 400.0, 300.0, 100.0, 80.0, 50.0],
        "Cluster": [0, 0, 0, 1, 1, 1, 2, 2, 3, 3]
    })

    seg_df, summary = interpret_rfm_clusters(rfm_clustered, cluster_col="Cluster")

    assert not summary.empty
    expected_summary_cols = [
        "Cluster", "Segment", "CustomerCount", "PercentCustomers",
        "AvgRecency", "AvgFrequency", "AvgMonetary", "TotalMonetary", "RecommendedAction"
    ]
    for col in expected_summary_cols:
        assert col in summary.columns, f"Missing column in summary table: {col}"

    assert summary["CustomerCount"].sum() == 10


if __name__ == "__main__":
    # Step 2
    test_synthetic_data_uci_schema()
    test_cleaning_cancelled_invoices()
    test_cleaning_missing_and_float_customer_ids()
    test_cleaning_duplicates()
    test_cleaning_invalid_quantity_and_price()
    test_cleaning_malformed_dates()
    test_source_dataframe_non_mutation()
    
    # Step 3
    test_manual_rfm_calculation_known_values()
    test_rfm_empty_and_invalid_inputs()
    test_rfm_summary_statistics()
    
    # Step 4
    test_clustering_repeatability_and_shape()
    test_clustering_invalid_k()
    test_clustering_small_dataset()
    test_cluster_evaluation_metrics()

    # Step 5
    test_dynamic_persona_mapping_not_hardcoded()
    test_persona_summary_table_structure()
    
    print("[SUCCESS] All Step 2, Step 3, Step 4, and Step 5 unit tests passed successfully!")
