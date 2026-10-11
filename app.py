"""
Streamlit Web Application
Cloud-Based Customer Segmentation for E-Commerce Using RFM Analysis and K-Means Clustering

Step 6: Polished Professional E-Commerce Analytics Dashboard
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import streamlit as st

from src.data_loader import generate_synthetic_data, load_csv_data, validate_dataset
from src.preprocessing import clean_transaction_data
from src.rfm import calculate_rfm, get_rfm_summary_statistics
from src.clustering import RFMClusterPipeline, evaluate_cluster_range
from src.segmentation import interpret_rfm_clusters, generate_business_insights, RECOMMENDED_ACTIONS
from src.visualizations import (
    plot_transaction_overview,
    plot_rfm_distributions,
    plot_segment_distribution,
    plot_elbow_curve,
    plot_silhouette_scores,
    plot_rfm_3d_scatter
)

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Platform",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Professional Styling (Light background, dark navy text, blue/teal accents)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #0F172A;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: #FFFFFF;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        color: #FFFFFF;
    }
    .main-header p {
        font-size: 1.0rem;
        color: #94A3B8;
        margin-top: 0.3rem;
        margin-bottom: 0;
    }
    
    .demo-banner {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.88rem;
        font-weight: 600;
        border: 1px solid #F59E0B;
        margin-bottom: 1.5rem;
    }

    .stat-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 4px solid #2563EB;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .stat-card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stat-card-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.2rem;
    }
    
    .section-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    
    .cloud-card {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initializes session state defaults safely."""
    if "df_raw" not in st.session_state:
        st.session_state["df_raw"] = generate_synthetic_data(num_customers=500, seed=42)
        st.session_state["is_demo"] = True
    if "is_demo" not in st.session_state:
        st.session_state["is_demo"] = True


def main():
    init_session_state()

    # Header Title
    st.markdown("""
    <div class="main-header">
        <h1>🛍️ E-Commerce Customer Segmentation Platform</h1>
        <p>RFM Analytics & Machine Learning Clustering Dashboard | College Project Prototype</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Navigation & Controls
    st.sidebar.title("🧭 Navigation & Control")
    nav_selection = st.sidebar.radio(
        "Select Dashboard Section:",
        options=[
            "📊 Executive Overview",
            "📁 Data Management & Audit",
            "📐 RFM Feature Analysis",
            "🤖 Clustering & Evaluation",
            "🎯 Customer Segments & Explorer",
            "💡 Business Insights & Actions"
        ],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Data Source Config")
    
    data_option = st.sidebar.radio(
        "Data Input Option:",
        options=["Synthetic Demo Dataset", "Upload Custom CSV Dataset"],
        index=0 if st.session_state.get("is_demo", True) else 1
    )

    if data_option == "Synthetic Demo Dataset":
        num_customers = st.sidebar.slider("Demo Customer Count", min_value=100, max_value=2000, value=500, step=50)
        seed = st.sidebar.number_input("Random Seed", value=42, min_value=1, max_value=9999)
        
        if st.sidebar.button("⚡ Generate Demo Data", use_container_width=True):
            st.session_state["df_raw"] = generate_synthetic_data(num_customers=num_customers, seed=seed)
            st.session_state["is_demo"] = True

    else:
        uploaded_file = st.sidebar.file_uploader("Upload Transaction CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                uploaded_df = load_csv_data(uploaded_file)
                st.session_state["df_raw"] = uploaded_df
                st.session_state["is_demo"] = False
            except Exception as e:
                st.sidebar.error(f"Error loading CSV: {str(e)}")

    # Reset Button
    if st.sidebar.button("🔄 Reset to Default Demo Data", use_container_width=True):
        st.session_state["df_raw"] = generate_synthetic_data(num_customers=500, seed=42)
        st.session_state["is_demo"] = True
        st.rerun()

    # K-Means Model Settings in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("🤖 Model Parameters")
    n_clusters = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=8, value=4, step=1)
    random_state = st.sidebar.number_input("Random Seed (random_state)", value=42, min_value=1, max_value=9999)

    df_raw = st.session_state.get("df_raw", None)
    is_demo = st.session_state.get("is_demo", True)

    # Demo Banner Notification
    if is_demo:
        st.markdown('<div class="demo-banner">⚠️ USING SYNTHETIC DEMO DATASET (Labeled for Testing & Showcase)</div>', unsafe_allow_html=True)

    if df_raw is None or df_raw.empty:
        st.warning("No dataset loaded. Please select a valid data source from the sidebar.")
        return

    # Data Validation
    is_valid, errors = validate_dataset(df_raw)
    if not is_valid:
        st.error("❌ Dataset Schema Validation Failed:")
        for err in errors:
            st.write(f"- {err}")
        return

    # Pipeline Processing
    df_cleaned, stats = clean_transaction_data(df_raw)
    if df_cleaned.empty:
        st.warning("No clean records available after data cleaning.")
        return

    snapshot_date = df_cleaned["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm_df = calculate_rfm(df_cleaned, reference_date=snapshot_date)

    try:
        pipeline = RFMClusterPipeline(n_clusters=n_clusters, random_state=random_state)
        clustered_rfm, scaled_matrix = pipeline.fit_predict(rfm_df)
        segmented_rfm, segment_summary = interpret_rfm_clusters(clustered_rfm, cluster_col="Cluster")
    except ValueError as err:
        st.error(f"Clustering Error: {str(err)}")
        return

    # =========================================================================
    # SECTION 1: EXECUTIVE OVERVIEW
    # =========================================================================
    if nav_selection == "📊 Executive Overview":
        st.subheader("📊 Executive Overview & High-Level Metrics")

        # 5 Top Overview KPI Cards
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-card-title">Unique Customers</div>
                <div class="stat-card-value">{len(rfm_df):,}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-card-title">Valid Transactions</div>
                <div class="stat-card-value">{len(df_cleaned):,}</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            total_rev = df_cleaned["TotalSpend"].sum()
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-card-title">Total Revenue</div>
                <div class="stat-card-value">${total_rev:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            avg_cust_val = rfm_df["Monetary"].mean()
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-card-title">Avg Spend / Customer</div>
                <div class="stat-card-value">${avg_cust_val:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with c5:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-card-title">Detected Segments</div>
                <div class="stat-card-value">{n_clusters}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Overview Visualizations
        col_chart1, col_chart2 = st.columns([1.2, 1])
        with col_chart1:
            st.plotly_chart(plot_transaction_overview(df_cleaned), use_container_width=True)
        with col_chart2:
            st.plotly_chart(plot_segment_distribution(segment_summary), use_container_width=True)

        # Architecture Banner
        with st.expander("☁️ View Future Cloud Architecture Roadmap (Google Cloud Platform)", expanded=False):
            st.markdown("""
            <div class="cloud-card">
                <h4>Production Cloud Roadmap (Planned Integration)</h4>
                <ul>
                    <li><b>Google Cloud Storage (GCS):</b> Ingest raw daily transaction CSV batch loads.</li>
                    <li><b>Google BigQuery SQL:</b> Scalable data warehouse for automated ETL & RFM views.</li>
                    <li><b>BigQuery ML:</b> Train K-Means clustering natively in SQL across millions of customer records.</li>
                    <li><b>Looker Studio:</b> Executive live dashboard for cross-functional marketing teams.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # =========================================================================
    # SECTION 2: DATA MANAGEMENT & AUDIT
    # =========================================================================
    elif nav_selection == "📁 Data Management & Audit":
        st.subheader("📁 Data Management & Cleaning Audit Report")

        st.markdown(f"""
        **Dataset Summary:** Total **{len(df_cleaned):,}** clean transactions spanning from **{df_cleaned['InvoiceDate'].min().strftime('%Y-%m-%d')}** to **{df_cleaned['InvoiceDate'].max().strftime('%Y-%m-%d')}**.
        """)

        # Cleaning Audit Cards
        ac1, ac2, ac3, ac4, ac5 = st.columns(5)
        with ac1:
            st.metric("Total Records Loaded", f"{stats['total_loaded']:,}")
        with ac2:
            st.metric("Duplicates Dropped", f"{stats['duplicates_removed']:,}")
        with ac3:
            st.metric("Cancellations Dropped", f"{stats['cancellations_removed']:,}")
        with ac4:
            st.metric("Invalid Rows Dropped", f"{stats['invalid_values_removed']:,}")
        with ac5:
            st.metric("Clean Records Retained", f"{stats['total_retained']:,}")

        st.markdown("---")
        st.subheader("🔍 Cleaned Transaction Dataset Preview")
        st.dataframe(df_cleaned.head(100), use_container_width=True)

        # CSV Export Button for Clean Data
        csv_clean = df_cleaned.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Cleaned Transactions CSV",
            data=csv_clean,
            file_name="cleaned_transactions.csv",
            mime="text/csv"
        )

    # =========================================================================
    # SECTION 3: RFM FEATURE ANALYSIS
    # =========================================================================
    elif nav_selection == "📐 RFM Feature Analysis":
        st.subheader("📐 Recency, Frequency, and Monetary (RFM) Metrics")

        st.markdown(f"""
        * **Recency (R):** Days elapsed since customer's last purchase (Reference date: **{snapshot_date.strftime('%Y-%m-%d')}**, 1 day post latest invoice).
        * **Frequency (F):** Total number of distinct valid orders (`InvoiceNo` count) placed.
        * **Monetary (M):** Total monetary revenue generated across all orders ($).
        """)

        # RFM Summary Table
        st.markdown("### 📋 RFM Descriptive Statistics")
        rfm_stats = get_rfm_summary_statistics(rfm_df)
        st.table(rfm_stats)

        # Feature Distributions
        st.markdown("### 📊 Metric Distributions")
        st.plotly_chart(plot_rfm_distributions(rfm_df), use_container_width=True)

        st.markdown("### 🔍 Customer RFM Table Preview")
        st.dataframe(rfm_df.head(100), use_container_width=True)

        # CSV Export Button for RFM Table
        csv_rfm = rfm_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Customer RFM Table CSV",
            data=csv_rfm,
            file_name="customer_rfm_metrics.csv",
            mime="text/csv"
        )

    # =========================================================================
    # SECTION 4: CLUSTERING & EVALUATION
    # =========================================================================
    elif nav_selection == "🤖 Clustering & Evaluation":
        st.subheader("🤖 K-Means Clustering & Model Diagnostics")

        # Evaluate range k=2..8
        eval_results = evaluate_cluster_range(rfm_df, k_min=2, k_max=8, random_state=random_state)

        c_eval1, c_eval2 = st.columns(2)
        with c_eval1:
            st.plotly_chart(plot_elbow_curve(eval_results, selected_k=n_clusters), use_container_width=True)
        with c_eval2:
            st.plotly_chart(plot_silhouette_scores(eval_results, selected_k=n_clusters), use_container_width=True)

        st.info("💡 **Evaluator Note:** The Silhouette score and Elbow plot provide mathematical separation heuristics. Business utility often favors $k=4$ or $k=5$ for actionable personas even if lower $k$ has a slightly higher silhouette coefficient.")

        st.markdown(f"### 🌐 3D Customer Clusters Scatter Plot (k = {n_clusters})")
        st.plotly_chart(plot_rfm_3d_scatter(segmented_rfm, cluster_col="Segment"), use_container_width=True)

        st.markdown("### 📋 Cluster Profile Summary")
        st.table(segment_summary[["Cluster", "Segment", "CustomerCount", "PercentCustomers", "AvgRecency", "AvgFrequency", "AvgMonetary", "TotalMonetary"]])

    # =========================================================================
    # SECTION 5: CUSTOMER SEGMENTS & EXPLORER
    # =========================================================================
    elif nav_selection == "🎯 Customer Segments & Explorer":
        st.subheader("🎯 Customer Segments & Interactive Explorer")

        # Filter controls
        f_col1, f_col2 = st.columns([1, 1])
        with f_col1:
            all_segments = ["All Segments"] + list(segmented_rfm["Segment"].unique())
            selected_segment = st.selectbox("Filter by Persona Segment:", options=all_segments)
        with f_col2:
            search_cid = st.text_input("Search Customer ID:", value="")

        # Apply filtering
        filtered_df = segmented_rfm.copy()
        if selected_segment != "All Segments":
            filtered_df = filtered_df[filtered_df["Segment"] == selected_segment]
        if search_cid:
            filtered_df = filtered_df[filtered_df["CustomerID"].astype(str).str.contains(search_cid, case=False)]

        st.markdown(f"Showing **{len(filtered_df):,}** of **{len(segmented_rfm):,}** customers:")
        st.dataframe(filtered_df[["CustomerID", "Recency", "Frequency", "Monetary", "Cluster", "Segment", "RecommendedAction"]], use_container_width=True)

        # CSV Export Button for Segmented Data
        csv_seg = segmented_rfm.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Full Segmented Customers CSV",
            data=csv_seg,
            file_name="segmented_customers.csv",
            mime="text/csv"
        )

    # =========================================================================
    # SECTION 6: BUSINESS INSIGHTS & ACTIONS
    # =========================================================================
    elif nav_selection == "💡 Business Insights & Actions":
        st.subheader("💡 Deterministic Business Insights & Strategic Action Plan")

        insights = generate_business_insights(segment_summary)
        if not insights:
            st.warning("No segment insights available.")
            return

        # Render 4 Key Insight Cards
        i_col1, i_col2, i_col3, i_col4 = st.columns(4)
        with i_col1:
            high_rev = insights["highest_revenue"]
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-card-title">Top Revenue Segment</div>
                <div class="stat-card-value" style="font-size:1.4rem;">{high_rev['segment']}</div>
                <p style="margin:0; font-size:0.85rem; color:#64748B;">${high_rev['total_revenue']:,.2f} ({high_rev['pct_share']:.1f}% total)</p>
            </div>
            """, unsafe_allow_html=True)
        with i_col2:
            high_mon = insights["highest_monetary"]
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-card-title">Highest Avg Customer Value</div>
                <div class="stat-card-value" style="font-size:1.4rem;">{high_mon['segment']}</div>
                <p style="margin:0; font-size:0.85rem; color:#64748B;">${high_mon['avg_spend']:,.2f} / customer</p>
            </div>
            """, unsafe_allow_html=True)
        with i_col3:
            high_rec = insights["highest_recency"]
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-card-title">Highest Recency (Most Dormant)</div>
                <div class="stat-card-value" style="font-size:1.4rem;">{high_rec['segment']}</div>
                <p style="margin:0; font-size:0.85rem; color:#64748B;">Avg {high_rec['avg_recency_days']:.0f} days since order</p>
            </div>
            """, unsafe_allow_html=True)
        with i_col4:
            large_seg = insights["largest_segment"]
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-card-title">Largest Customer Segment</div>
                <div class="stat-card-value" style="font-size:1.4rem;">{large_seg['segment']}</div>
                <p style="margin:0; font-size:0.85rem; color:#64748B;">{large_seg['customer_count']:,} customers ({large_seg['pct_share']:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🎯 Recommended Marketing Strategy Action Plan")

        for _, row in segment_summary.iterrows():
            seg_name = row["Segment"]
            cnt = row["CustomerCount"]
            pct = row["PercentCustomers"]
            action = row["RecommendedAction"]
            
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-left:5px solid #2563EB; border-radius:8px; padding:1.2rem; margin-bottom:1rem;">
                <h4 style="margin-top:0;">{seg_name} <span style="font-size:0.85rem; color:#64748B; font-weight:normal;">({cnt:,} customers — {pct})</span></h4>
                <p style="margin-bottom:0; color:#1E40AF;"><b>Action Plan:</b> {action}</p>
            </div>
            """, unsafe_allow_html=True)

        st.caption("📌 **Note on Data Ethics:** All insights above are calculated deterministically from computed cluster metrics. No statistics were invented, and no historical campaign performance is claimed.")


if __name__ == "__main__":
    main()
