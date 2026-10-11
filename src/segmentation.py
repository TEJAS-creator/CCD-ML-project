"""
Business Interpretation & Personas Mapping Engine for Customer Segmentation.

Translates unsupervised K-Means numeric cluster IDs (0, 1, 2, ...) into actionable business personas
based on relative RFM centroid characteristics (Recency, Frequency, Monetary).

PERSONAS DEFINED:
1. Champions: Recent buyers with highest frequency and highest monetary spend.
2. Loyal Customers: High purchase frequency and steady monetary contribution.
3. Potential Loyalists / New Buyers: Recent buyers with lower purchase frequency.
4. At-Risk / Needs Attention: High recency (dormant) despite strong historical spend/frequency.
5. Hibernating / Lost Customers: High recency, low frequency, and low monetary value.
6. Generic Segment (Fallback): Used when cluster metrics are ambiguous.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd


RECOMMENDED_ACTIONS: Dict[str, str] = {
    "Champions": "Provide VIP perks, early product access, and request reviews/testimonials.",
    "Loyal Customers": "Cross-sell related products, offer volume discounts, and loyalty rewards.",
    "Potential Loyalists / New Buyers": "Send welcome onboarding incentives and product guides to encourage 2nd purchase.",
    "At-Risk / Needs Attention": "Trigger automated win-back emails, special discount offers, and satisfaction surveys.",
    "Hibernating / Lost Customers": "Send low-cost win-back campaigns or exclude from paid ad targeting to save spend.",
    "Generic Segment": "Monitor purchasing metrics for emerging behavioral patterns."
}


def interpret_rfm_clusters(
    rfm_df: pd.DataFrame, 
    cluster_col: str = "Cluster"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyzes cluster centroids relative to global dataset statistics and maps dynamic business personas.
    
    Does NOT use hardcoded cluster IDs. Instead, ranks and evaluates cluster centroids dynamically.
    
    Args:
        rfm_df: DataFrame containing [CustomerID, Recency, Frequency, Monetary, Cluster].
        cluster_col: Column name containing integer cluster assignments.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - rfm_df enriched with 'Segment' and 'RecommendedAction' columns.
            - Segment summary DataFrame with centroid stats and actionable recommendations.
            
    Raises:
        ValueError: If required columns are missing from rfm_df.
    """
    required = ["CustomerID", "Recency", "Frequency", "Monetary", cluster_col]
    missing = [col for col in required if col not in rfm_df.columns]
    if missing:
        raise ValueError(f"Input DataFrame missing required columns for segmentation: {missing}")

    if rfm_df.empty:
        empty_df = rfm_df.copy()
        empty_df["Segment"] = []
        empty_df["RecommendedAction"] = []
        return empty_df, pd.DataFrame()

    df_work = rfm_df.copy()
    
    # Calculate cluster centroids (mean RFM per cluster)
    centroids = (
        df_work.groupby(cluster_col)
        .agg(
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean"),
            CustomerCount=("CustomerID", "count"),
            TotalMonetary=("Monetary", "sum")
        )
        .reset_index()
    )

    # Compute global dataset averages for benchmarking
    global_avg_r = df_work["Recency"].mean()
    global_avg_f = df_work["Frequency"].mean()
    global_avg_m = df_work["Monetary"].mean()

    # Assign relative ranking scores (higher score = better customer)
    centroids["R_Rank"] = centroids["AvgRecency"].rank(ascending=False)
    centroids["F_Rank"] = centroids["AvgFrequency"].rank(ascending=True)
    centroids["M_Rank"] = centroids["AvgMonetary"].rank(ascending=True)

    # Combined composite RFM score
    centroids["CompositeScore"] = centroids["R_Rank"] + (1.5 * centroids["F_Rank"]) + (2.0 * centroids["M_Rank"])

    assigned_labels = {}
    used_personas = set()

    # Sort centroids by composite score descending to assign personas deterministically
    sorted_centroids = centroids.sort_values("CompositeScore", ascending=False)

    for idx, row in sorted_centroids.iterrows():
        c_id = row[cluster_col]
        avg_r = row["AvgRecency"]
        avg_f = row["AvgFrequency"]
        avg_m = row["AvgMonetary"]
        
        candidate_label = None

        if avg_r <= global_avg_r and avg_m >= global_avg_m and avg_f >= global_avg_f:
            if "Champions" not in used_personas:
                candidate_label = "Champions"
            elif "Loyal Customers" not in used_personas:
                candidate_label = "Loyal Customers"

        elif avg_r <= global_avg_r and avg_f < global_avg_f:
            if "Potential Loyalists / New Buyers" not in used_personas:
                candidate_label = "Potential Loyalists / New Buyers"

        elif avg_r > global_avg_r and avg_m >= global_avg_m:
            if "At-Risk / Needs Attention" not in used_personas:
                candidate_label = "At-Risk / Needs Attention"

        elif avg_r > global_avg_r and avg_m < global_avg_m:
            if "Hibernating / Lost Customers" not in used_personas:
                candidate_label = "Hibernating / Lost Customers"

        if candidate_label is None or candidate_label in used_personas:
            fallback_options = [
                "Champions",
                "Loyal Customers",
                "Potential Loyalists / New Buyers",
                "At-Risk / Needs Attention",
                "Hibernating / Lost Customers"
            ]
            for opt in fallback_options:
                if opt not in used_personas:
                    candidate_label = opt
                    break
            
            if candidate_label is None or candidate_label in used_personas:
                candidate_label = f"Segment {c_id}"

        assigned_labels[c_id] = candidate_label
        used_personas.add(candidate_label)

    df_work["Segment"] = df_work[cluster_col].map(assigned_labels)
    df_work["RecommendedAction"] = df_work["Segment"].map(
        lambda seg: RECOMMENDED_ACTIONS.get(seg, RECOMMENDED_ACTIONS["Generic Segment"])
    )

    total_customers = len(df_work)
    summary_list = []

    for c_id, seg_name in assigned_labels.items():
        sub = df_work[df_work[cluster_col] == c_id]
        c_count = len(sub)
        pct = round((c_count / total_customers) * 100, 1) if total_customers > 0 else 0.0
        
        summary_list.append({
            "Cluster": c_id,
            "Segment": seg_name,
            "CustomerCount": c_count,
            "PercentCustomers": f"{pct}%",
            "AvgRecency": round(float(sub["Recency"].mean()), 1),
            "AvgFrequency": round(float(sub["Frequency"].mean()), 1),
            "AvgMonetary": round(float(sub["Monetary"].mean()), 2),
            "TotalMonetary": round(float(sub["Monetary"].sum()), 2),
            "RecommendedAction": RECOMMENDED_ACTIONS.get(seg_name, RECOMMENDED_ACTIONS["Generic Segment"])
        })

    segment_summary = pd.DataFrame(summary_list)
    segment_summary = segment_summary.sort_values("TotalMonetary", ascending=False).reset_index(drop=True)

    return df_work, segment_summary


def generate_business_insights(segment_summary: pd.DataFrame) -> Dict[str, Any]:
    """
    Extracts deterministic business insights directly from computed segment metrics.
    No fake or hardcoded values are generated.
    
    Args:
        segment_summary: DataFrame output from interpret_rfm_clusters.
        
    Returns:
        Dict[str, Any]: Dictionary of key performance insights.
    """
    if segment_summary is None or segment_summary.empty:
        return {}
        
    df_sorted_monetary = segment_summary.sort_values("AvgMonetary", ascending=False)
    highest_monetary = df_sorted_monetary.iloc[0]
    
    df_sorted_revenue = segment_summary.sort_values("TotalMonetary", ascending=False)
    highest_revenue = df_sorted_revenue.iloc[0]
    
    df_sorted_recency = segment_summary.sort_values("AvgRecency", ascending=False)
    highest_recency = df_sorted_recency.iloc[0]
    
    df_sorted_count = segment_summary.sort_values("CustomerCount", ascending=False)
    largest_segment = df_sorted_count.iloc[0]
    
    total_rev = float(segment_summary["TotalMonetary"].sum())
    total_cust = int(segment_summary["CustomerCount"].sum())
    
    return {
        "highest_monetary": {
            "segment": highest_monetary["Segment"],
            "avg_spend": float(highest_monetary["AvgMonetary"])
        },
        "highest_revenue": {
            "segment": highest_revenue["Segment"],
            "total_revenue": float(highest_revenue["TotalMonetary"]),
            "pct_share": float(highest_revenue["TotalMonetary"]) / total_rev * 100 if total_rev > 0 else 0.0
        },
        "highest_recency": {
            "segment": highest_recency["Segment"],
            "avg_recency_days": float(highest_recency["AvgRecency"])
        },
        "largest_segment": {
            "segment": largest_segment["Segment"],
            "customer_count": int(largest_segment["CustomerCount"]),
            "pct_share": float(largest_segment["CustomerCount"]) / total_cust * 100 if total_cust > 0 else 0.0
        },
        "total_revenue": total_rev,
        "total_customers": total_cust
    }
