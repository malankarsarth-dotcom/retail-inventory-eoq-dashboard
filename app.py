"""Streamlit dashboard for Retail Inventory Management & EOQ Optimization."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from eoq_model import (
    compute_cost_curve,
    compute_eoq_metrics,
    load_inventory_data,
    simulate_monthly_depletion,
)

st.set_page_config(
    page_title="Retail Inventory Optimization (EOQ)",
    page_icon="📦",
    layout="wide",
)


@st.cache_data
def get_base_data() -> "pd.DataFrame":
    return load_inventory_data("data/supply_chain_data.csv")


def main() -> None:
    st.title("Retail Inventory Management & Optimization (EOQ Dashboard)")
    st.caption(
        "Dataset: `data/supply_chain_data.csv` (5,001 retail SKUs) | "
        "Model: Economic Order Quantity — $\\text{EOQ} = \\sqrt{\\frac{2DS}{H}}$"
    )

    raw_df = get_base_data()

    # Sidebar controls
    st.sidebar.header("Model & Filter Controls")
    product_types = sorted(raw_df["Product type"].unique().tolist())
    selected_types = st.sidebar.multiselect(
        "Product Type",
        options=product_types,
        default=product_types,
    )

    locations = sorted(raw_df["Location"].unique().tolist())
    selected_locations = st.sidebar.multiselect(
        "Location",
        options=locations,
        default=locations,
    )

    holding_rate_pct = st.sidebar.slider(
        "Annual Holding Cost Rate (% of Price)",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
    )
    ordering_cost_col = st.sidebar.selectbox(
        "Ordering / Setup Cost Column (S)",
        options=["Costs", "Shipping costs"],
        index=0,
    )

    filtered_raw = raw_df[
        raw_df["Product type"].isin(selected_types)
        & raw_df["Location"].isin(selected_locations)
    ]

    if filtered_raw.empty:
        st.warning("No SKUs match the selected filters. Adjust the sidebar filters.")
        return

    df = compute_eoq_metrics(
        filtered_raw,
        holding_rate=holding_rate_pct / 100.0,
        ordering_cost_col=ordering_cost_col,
    )

    # Top-level KPIs
    total_current_cost = df["Current_Total_Cost"].sum()
    total_optimal_cost = df["Optimal_Total_Cost"].sum()
    total_savings = df["Annual_Cost_Savings"].sum()
    savings_pct = (total_savings / total_current_cost * 100.0) if total_current_cost > 0 else 0.0
    reorder_count = int((df["Stock_Status"] == "Reorder Now").sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("SKUs Analyzed", f"{len(df):,}")
    col2.metric("Average Optimal EOQ", f"{df['EOQ'].mean():.1f} units")
    col3.metric(
        "Projected Annual Savings",
        f"${total_savings:,.0f}",
        f"{savings_pct:.1f}% vs. current",
    )
    col4.metric(
        "SKUs Below Reorder Point",
        f"{reorder_count:,}",
        f"{reorder_count / len(df) * 100:.1f}% of selection",
        delta_color="inverse",
    )

    st.divider()

    # SKU-level deep dive (12-Month Depletion + EOQ Cost Curve)
    st.subheader("SKU-Level Inventory Simulation & Cost Trade-off")
    sku_options = df["SKU"].tolist()
    selected_sku = st.selectbox("Select SKU to Inspect", options=sku_options, index=0)
    sku_row = df[df["SKU"] == selected_sku].iloc[0]

    start_mode = st.radio(
        "Starting Inventory for 12-Month Simulation",
        options=["Start at Optimal EOQ", "Start at Current Stock Level"],
        horizontal=True,
    )
    initial_stock = (
        None if start_mode == "Start at Optimal EOQ" else float(sku_row["Stock levels"])
    )

    sim_df = simulate_monthly_depletion(
        annual_demand=float(sku_row["Annual_Demand_D"]),
        eoq=float(sku_row["EOQ"]),
        initial_stock=initial_stock,
        months=12,
    )

    # SKU summary strip
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Annual Demand (D)", f"{sku_row['Annual_Demand_D']:,.0f} units")
    s2.metric("Ordering Cost (S)", f"${sku_row['Ordering_Cost_S']:,.2f}")
    s3.metric("Holding Cost (H)", f"${sku_row['Holding_Cost_H']:,.2f}/unit/yr")
    s4.metric("Optimal EOQ", f"{sku_row['EOQ']:,.1f} units")
    s5.metric("Reorder Point (ROP)", f"{sku_row['Reorder_Point_ROP']:,.1f} units")

    left_chart, right_chart = st.columns(2)

    with left_chart:
        fig_dep = go.Figure()
        fig_dep.add_trace(
            go.Scatter(
                x=sim_df["Month"],
                y=sim_df["Remaining_Units"],
                mode="lines+markers+text",
                name="Remaining Inventory",
                text=[f"{val:.1f}" for val in sim_df["Remaining_Units"]],
                textposition="top center",
                line=dict(color="#2E86AB", width=3),
                marker=dict(size=8),
            )
        )
        fig_dep.add_hline(
            y=float(sku_row["Reorder_Point_ROP"]),
            line_dash="dash",
            line_color="#E63946",
            annotation_text=f"ROP ({sku_row['Reorder_Point_ROP']:.1f})",
            annotation_position="bottom right",
        )
        fig_dep.update_layout(
            title=f"12-Month Inventory Depletion — {selected_sku} ({sku_row['Product type'].title()})",
            xaxis_title="Month",
            yaxis_title="Remaining Inventory (Units)",
            xaxis=dict(tickmode="linear", tick0=1, dtick=1, showgrid=True),
            yaxis=dict(rangemode="tozero", showgrid=True),
            height=420,
            margin=dict(t=50, b=40, l=40, r=20),
        )
        st.plotly_chart(fig_dep, use_container_width=True)

    with right_chart:
        curve_df = compute_cost_curve(
            annual_demand=float(sku_row["Annual_Demand_D"]),
            ordering_cost=float(sku_row["Ordering_Cost_S"]),
            holding_cost=float(sku_row["Holding_Cost_H"]),
            eoq=float(sku_row["EOQ"]),
        )
        fig_cost = go.Figure()
        fig_cost.add_trace(
            go.Scatter(
                x=curve_df["Order_Quantity"],
                y=curve_df["Total_Cost"],
                mode="lines",
                name="Total Annual Cost",
                line=dict(color="#1D3557", width=3),
            )
        )
        fig_cost.add_trace(
            go.Scatter(
                x=curve_df["Order_Quantity"],
                y=curve_df["Ordering_Cost"],
                mode="lines",
                name="Ordering Cost",
                line=dict(color="#F4A261", dash="dot"),
            )
        )
        fig_cost.add_trace(
            go.Scatter(
                x=curve_df["Order_Quantity"],
                y=curve_df["Holding_Cost"],
                mode="lines",
                name="Holding Cost",
                line=dict(color="#2A9D8F", dash="dash"),
            )
        )
        fig_cost.add_vline(
            x=float(sku_row["EOQ"]),
            line_dash="dash",
            line_color="#2A9D8F",
            annotation_text=f"EOQ ({sku_row['EOQ']:.1f})",
        )
        fig_cost.add_vline(
            x=float(sku_row["Order quantities"]),
            line_dash="dot",
            line_color="#E63946",
            annotation_text=f"Current Q ({sku_row['Order quantities']:.0f})",
            annotation_position="bottom right",
        )
        fig_cost.update_layout(
            title=f"EOQ Cost Trade-off Curve — {selected_sku}",
            xaxis_title="Order Quantity (Units)",
            yaxis_title="Annual Cost ($)",
            height=420,
            margin=dict(t=50, b=40, l=40, r=20),
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    st.divider()

    # Portfolio-level charts
    st.subheader("Portfolio Optimization Insights")
    cat_left, cat_right = st.columns(2)

    with cat_left:
        cat_summary = (
            df.groupby("Product type", as_index=False)[
                ["Current_Total_Cost", "Optimal_Total_Cost", "Annual_Cost_Savings"]
            ]
            .sum()
            .melt(
                id_vars="Product type",
                value_vars=["Current_Total_Cost", "Optimal_Total_Cost"],
                var_name="Policy",
                value_name="Annual Cost ($)",
            )
        )
        cat_summary["Policy"] = cat_summary["Policy"].replace(
            {
                "Current_Total_Cost": "Current Order Policy",
                "Optimal_Total_Cost": "Optimal EOQ Policy",
            }
        )
        fig_cat = px.bar(
            cat_summary,
            x="Product type",
            y="Annual Cost ($)",
            color="Policy",
            barmode="group",
            title="Current vs. Optimal EOQ Annual Cost by Product Type",
            color_discrete_sequence=["#E76F51", "#2A9D8F"],
        )
        fig_cat.update_layout(height=400)
        st.plotly_chart(fig_cat, use_container_width=True)

    with cat_right:
        top_savings = df.nlargest(10, "Annual_Cost_Savings").sort_values(
            "Annual_Cost_Savings", ascending=True
        )
        fig_top = px.bar(
            top_savings,
            x="Annual_Cost_Savings",
            y="SKU",
            color="Product type",
            orientation="h",
            title="Top 10 SKUs by Projected Annual Cost Savings",
            labels={"Annual_Cost_Savings": "Annual Savings ($)"},
        )
        fig_top.update_layout(height=400)
        st.plotly_chart(fig_top, use_container_width=True)

    # Data table & CSV export
    st.subheader("EOQ Optimization Table")
    display_cols = [
        "SKU",
        "Product type",
        "Location",
        "Price",
        "Annual_Demand_D",
        "Order quantities",
        "EOQ",
        "Stock levels",
        "Reorder_Point_ROP",
        "Stock_Status",
        "Current_Total_Cost",
        "Optimal_Total_Cost",
        "Annual_Cost_Savings",
        "Savings_Pct",
    ]
    formatted_df = df[display_cols].sort_values("Annual_Cost_Savings", ascending=False)
    st.dataframe(
        formatted_df.style.format(
            {
                "Price": "${:,.2f}",
                "Annual_Demand_D": "{:,.0f}",
                "Order quantities": "{:,.0f}",
                "EOQ": "{:,.1f}",
                "Stock levels": "{:,.0f}",
                "Reorder_Point_ROP": "{:,.1f}",
                "Current_Total_Cost": "${:,.2f}",
                "Optimal_Total_Cost": "${:,.2f}",
                "Annual_Cost_Savings": "${:,.2f}",
                "Savings_Pct": "{:,.1f}%",
            }
        ),
        use_container_width=True,
        height=350,
    )

    csv_bytes = formatted_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Optimized EOQ Table (CSV)",
        data=csv_bytes,
        file_name="eoq_optimized_inventory.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
