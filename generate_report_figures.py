"""Generate high-resolution pictorial report figures from the real EOQ dataset."""

from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eoq_model import (
    compute_cost_curve,
    compute_eoq_metrics,
    load_inventory_data,
    simulate_monthly_depletion,
)

ASSETS_DIR = "report_assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "figure.dpi": 180,
    }
)


def main() -> None:
    raw_df = load_inventory_data("data/supply_chain_data.csv")
    df = compute_eoq_metrics(raw_df, holding_rate=0.20, ordering_cost_col="Costs")

    # Representative SKU: SKU0
    sku_row = df[df["SKU"] == "SKU0"].iloc[0]
    sim_df = simulate_monthly_depletion(
        annual_demand=float(sku_row["Annual_Demand_D"]),
        eoq=float(sku_row["EOQ"]),
        months=12,
    )

    # Figure 1: 12-Month Inventory Depletion (Slide Requirement)
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.plot(
        sim_df["Month"],
        sim_df["Remaining_Units"],
        marker="o",
        color="#2E86AB",
        linewidth=2.5,
        markersize=7,
        label="Remaining Inventory (Units)",
    )
    for m, inv in zip(sim_df["Month"], sim_df["Remaining_Units"]):
        ax.annotate(
            f"{inv:.1f}",
            (m, inv),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="#1D3557",
        )
    rop = float(sku_row["Reorder_Point_ROP"])
    ax.axhline(
        rop,
        color="#E63946",
        linestyle="--",
        linewidth=1.6,
        label=f"Reorder Point (ROP = {rop:.1f} units)",
    )
    ax.set_title(
        f"12-Month Retail Inventory Depletion — {sku_row['SKU']} "
        f"({sku_row['Product type'].title()}, EOQ = {sku_row['EOQ']:.1f} units)"
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Remaining Inventory (Units)")
    ax.set_xticks(sim_df["Month"])
    ax.set_ylim(0, max(sim_df["Remaining_Units"]) * 1.18)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "fig1_monthly_depletion.png"))
    plt.close(fig)

    # Figure 2: EOQ Cost Trade-off Curve for SKU0
    curve_df = compute_cost_curve(
        annual_demand=float(sku_row["Annual_Demand_D"]),
        ordering_cost=float(sku_row["Ordering_Cost_S"]),
        holding_cost=float(sku_row["Holding_Cost_H"]),
        eoq=float(sku_row["EOQ"]),
    )
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.plot(curve_df["Order_Quantity"], curve_df["Total_Cost"], color="#1D3557", linewidth=2.5, label="Total Annual Cost")
    ax.plot(curve_df["Order_Quantity"], curve_df["Ordering_Cost"], color="#F4A261", linestyle=":", linewidth=2, label="Annual Ordering Cost")
    ax.plot(curve_df["Order_Quantity"], curve_df["Holding_Cost"], color="#2A9D8F", linestyle="--", linewidth=2, label="Annual Holding Cost")
    ax.axvline(float(sku_row["EOQ"]), color="#2A9D8F", linestyle="--", linewidth=1.8, label=f"Optimal EOQ = {sku_row['EOQ']:.1f}")
    ax.axvline(float(sku_row["Order quantities"]), color="#E63946", linestyle="-.", linewidth=1.8, label=f"Current Order Q = {sku_row['Order quantities']:.0f}")
    ax.set_title(f"EOQ Cost Trade-Off Analysis — {sku_row['SKU']}")
    ax.set_xlabel("Order Quantity (Units)")
    ax.set_ylabel("Annual Cost ($)")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="upper right", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "fig2_eoq_cost_curve.png"))
    plt.close(fig)

    # Figure 3: Category-Level Annual Cost Comparison
    cat_df = df.groupby("Product type", as_index=False)[["Current_Total_Cost", "Optimal_Total_Cost", "Annual_Cost_Savings"]].sum()
    x = np.arange(len(cat_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    b1 = ax.bar(x - width / 2, cat_df["Current_Total_Cost"] / 1e6, width, label="Current Order Policy", color="#E76F51")
    b2 = ax.bar(x + width / 2, cat_df["Optimal_Total_Cost"] / 1e6, width, label="Optimal EOQ Policy", color="#2A9D8F")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.annotate(f"${h:.2f}M", (bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5)
    ax.set_title("Annual Inventory Cost by Product Category: Current vs. Optimal EOQ")
    ax.set_xlabel("Product Category")
    ax.set_ylabel("Total Annual Cost ($ Millions)")
    ax.set_xticks(x)
    ax.set_xticklabels([c.title() for c in cat_df["Product type"]])
    ax.set_ylim(0, max(cat_df["Current_Total_Cost"] / 1e6) * 1.18)
    ax.grid(True, axis="y", linestyle="--", alpha=0.45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "fig3_category_cost_comparison.png"))
    plt.close(fig)

    # Figure 4: Top 10 SKUs by Annual Cost Savings + Stock Status Pie
    top10 = df.nlargest(10, "Annual_Cost_Savings").sort_values("Annual_Cost_Savings", ascending=True)
    status_counts = df["Stock_Status"].value_counts()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5), gridspec_kw={"width_ratios": [1.5, 1]})
    bars = ax1.barh(top10["SKU"], top10["Annual_Cost_Savings"], color="#2E86AB")
    for bar in bars:
        w = bar.get_width()
        ax1.annotate(f"${w:,.0f}", (w, bar.get_y() + bar.get_height() / 2), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
    ax1.set_title("Top 10 SKUs by Projected Annual Savings")
    ax1.set_xlabel("Annual Cost Savings ($)")
    ax1.set_xlim(0, top10["Annual_Cost_Savings"].max() * 1.22)
    ax1.grid(True, axis="x", linestyle="--", alpha=0.45)

    colors = ["#2A9D8F" if s == "Healthy" else "#E63946" for s in status_counts.index]
    ax2.pie(
        status_counts.values,
        labels=status_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    ax2.set_title("Portfolio Stock Status vs. Reorder Point")

    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "fig4_top_savings_and_status.png"))
    plt.close(fig)

    # Print verified summary metrics
    total_curr = df["Current_Total_Cost"].sum()
    total_opt = df["Optimal_Total_Cost"].sum()
    total_sav = df["Annual_Cost_Savings"].sum()
    print(f"SKUs analyzed: {len(df)}")
    print(f"Mean EOQ: {df['EOQ'].mean():.2f}")
    print(f"Current Total Cost: ${total_curr:,.2f}")
    print(f"Optimal Total Cost: ${total_opt:,.2f}")
    print(f"Projected Annual Savings: ${total_sav:,.2f} ({total_sav / total_curr * 100:.2f}%)")
    print(f"Reorder Now count: {(df['Stock_Status'] == 'Reorder Now').sum()}")
    print(f"Healthy count: {(df['Stock_Status'] == 'Healthy').sum()}")
    print(f"SKU0 -> D={sku_row['Annual_Demand_D']:.0f}, S={sku_row['Ordering_Cost_S']:.2f}, H={sku_row['Holding_Cost_H']:.2f}, EOQ={sku_row['EOQ']:.2f}, ROP={sku_row['Reorder_Point_ROP']:.2f}")


if __name__ == "__main__":
    main()
