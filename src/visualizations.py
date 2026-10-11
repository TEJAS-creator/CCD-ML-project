"""
Plotly Visualizations Module for Customer Segmentation Project.

Generates interactive charts:
- Dataset transaction trend plot
- RFM Feature distributions (Histograms / Boxplots)
- K-Means Elbow Curve (Inertia vs k)
- Silhouette Score evaluation plot (Silhouette vs k)
- 3D Scatter plots for RFM Clusters
- Segment Distribution Donut Chart
"""

from typing import Dict, List, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COLOR_PALETTE = ["#2563EB", "#0D9488", "#4F46E5", "#F59E0B", "#EF4444", "#8B5CF6"]


def plot_transaction_overview(df: pd.DataFrame) -> go.Figure:
    """
    Creates an interactive trend plot of daily transaction quantities.
    
    Args:
        df: Transaction DataFrame.
        
    Returns:
        go.Figure: Plotly figure object.
    """
    df_copy = df.copy()
    df_copy["InvoiceDate"] = pd.to_datetime(df_copy["InvoiceDate"])
    
    daily_sales = df_copy.groupby(df_copy["InvoiceDate"].dt.date)["Quantity"].sum().reset_index()
    daily_sales.columns = ["Date", "TotalQuantity"]
    
    fig = px.line(
        daily_sales,
        x="Date",
        y="TotalQuantity",
        title="Transaction Quantity Trend Over Time",
        labels={"TotalQuantity": "Items Sold", "Date": "Date"},
        template="plotly_white"
    )
    fig.update_traces(line_color="#2563EB", line_width=2)
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plot_rfm_distributions(rfm_df: pd.DataFrame) -> go.Figure:
    """
    Creates subplots showing feature distributions for Recency, Frequency, and Monetary metrics.
    
    Args:
        rfm_df: RFM DataFrame containing [Recency, Frequency, Monetary].
        
    Returns:
        go.Figure: Plotly figure object.
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Recency Distribution (Days)", "Frequency Distribution (Orders)", "Monetary Distribution ($)")
    )

    fig.add_trace(
        go.Histogram(x=rfm_df["Recency"], name="Recency", marker_color="#2563EB", nbinsx=30),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=rfm_df["Frequency"], name="Frequency", marker_color="#0D9488", nbinsx=30),
        row=1, col=2
    )
    fig.add_trace(
        go.Histogram(x=rfm_df["Monetary"], name="Monetary", marker_color="#4F46E5", nbinsx=30),
        row=1, col=3
    )

    fig.update_layout(
        title_text="RFM Feature Metric Distributions",
        template="plotly_white",
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig


def plot_segment_distribution(segment_summary: pd.DataFrame) -> go.Figure:
    """
    Generates a donut chart showing customer distribution by segment.
    
    Args:
        segment_summary: DataFrame output from interpret_rfm_clusters.
        
    Returns:
        go.Figure: Plotly figure object.
    """
    fig = px.pie(
        segment_summary,
        values="CustomerCount",
        names="Segment",
        title="Customer Distribution by Segment Persona",
        hole=0.4,
        color_discrete_sequence=COLOR_PALETTE,
        template="plotly_white"
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plot_elbow_curve(eval_results: Dict[str, List[Any]], selected_k: int = 4) -> go.Figure:
    """
    Generates an interactive Elbow Method plot (Inertia vs Number of Clusters k).
    
    Args:
        eval_results: Dict containing 'k' and 'inertia'.
        selected_k: Currently selected k to highlight.
        
    Returns:
        go.Figure: Plotly figure object.
    """
    k_vals = eval_results.get("k", [])
    inertia_vals = eval_results.get("inertia", [])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=k_vals,
        y=inertia_vals,
        mode="lines+markers",
        name="Inertia",
        line=dict(color="#2563EB", width=3),
        marker=dict(size=8, color="#2563EB")
    ))

    if selected_k in k_vals:
        idx = k_vals.index(selected_k)
        fig.add_trace(go.Scatter(
            x=[selected_k],
            y=[inertia_vals[idx]],
            mode="markers",
            name=f"Selected k={selected_k}",
            marker=dict(size=14, color="#EF4444", symbol="star")
        ))

    fig.update_layout(
        title="Elbow Method (Inertia vs. Number of Clusters)",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Inertia (Within-Cluster Sum of Squares)",
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_silhouette_scores(eval_results: Dict[str, List[Any]], selected_k: int = 4) -> go.Figure:
    """
    Generates an interactive Silhouette Score evaluation plot (Silhouette vs k).
    
    Args:
        eval_results: Dict containing 'k' and 'silhouette'.
        selected_k: Currently selected k to highlight.
        
    Returns:
        go.Figure: Plotly figure object.
    """
    k_vals = eval_results.get("k", [])
    sil_vals = eval_results.get("silhouette", [])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=k_vals,
        y=sil_vals,
        name="Silhouette Score",
        marker_color=["#EF4444" if k == selected_k else "#0D9488" for k in k_vals],
        text=[f"{s:.3f}" for s in sil_vals],
        textposition="auto"
    ))

    fig.update_layout(
        title="Silhouette Score Comparison (Cluster Separation Quality)",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Mean Silhouette Coefficient",
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis=dict(range=[0, 1.0])
    )
    return fig


def plot_rfm_3d_scatter(rfm_df: pd.DataFrame, cluster_col: str = "Segment") -> go.Figure:
    """
    Creates an interactive 3D scatter plot of Recency, Frequency, and Monetary.
    
    Args:
        rfm_df: DataFrame containing RFM metrics and assigned segments/clusters.
        cluster_col: Column to use for coloring data points.
        
    Returns:
        go.Figure: Plotly figure object.
    """
    df_plot = rfm_df.copy()
    df_plot[cluster_col] = df_plot[cluster_col].astype(str)
    
    fig = px.scatter_3d(
        df_plot,
        x="Recency",
        y="Frequency",
        z="Monetary",
        color=cluster_col,
        hover_data=["CustomerID"],
        title="3D Customer Clusters Scatter Plot",
        labels={"Recency": "Recency (Days)", "Frequency": "Frequency (Orders)", "Monetary": "Monetary ($)"},
        color_discrete_sequence=COLOR_PALETTE,
        template="plotly_white"
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig
