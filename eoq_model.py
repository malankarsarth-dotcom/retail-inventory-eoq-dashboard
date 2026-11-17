"""EOQ (Economic Order Quantity) calculations and 12-month inventory simulation."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd


def load_inventory_data(csv_path: str = "data/supply_chain_data.csv") -> pd.DataFrame:
    """Load retail supply chain CSV and ensure positive numeric inputs for EOQ."""
    df = pd.read_csv(csv_path)
    numeric_cols = [
        "Price",
        "Number of products sold",
        "Revenue generated",
        "Stock levels",
        "Lead times",
        "Order quantities",
        "Shipping costs",
        "Manufacturing costs",
        "Costs",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["SKU", "Product type", "Location", "Price", "Number of products sold", "Costs", "Order quantities"])
    df = df[
        (df["Price"] > 0)
        & (df["Number of products sold"] > 0)
        & (df["Costs"] > 0)
        & (df["Order quantities"] > 0)
    ].copy()
    return df


def compute_eoq_metrics(df: pd.DataFrame, holding_rate: float = 0.20, ordering_cost_col: str = "Costs") -> pd.DataFrame:
    """Compute EOQ, Reorder Point, and current vs. optimal annual inventory costs."""
    result = df.copy()
    demand = result["Number of products sold"]
    ordering_cost = result[ordering_cost_col].clip(lower=1.0)
    holding_cost = (result["Price"] * holding_rate).clip(lower=0.01)
    current_q = result["Order quantities"].clip(lower=1.0)

    eoq = np.sqrt((2.0 * demand * ordering_cost) / holding_cost)

    result["Annual_Demand_D"] = demand
    result["Ordering_Cost_S"] = ordering_cost
    result["Holding_Cost_H"] = holding_cost
    result["EOQ"] = eoq
    result["Optimal_Orders_Per_Year"] = demand / eoq
    result["Reorder_Point_ROP"] = (demand / 365.0) * result["Lead times"].fillna(0)
    result["Stock_Status"] = np.where(
        result["Stock levels"] <= result["Reorder_Point_ROP"],
        "Reorder Now",
        "Healthy",
    )

    result["Optimal_Total_Cost"] = (demand / eoq) * ordering_cost + (eoq / 2.0) * holding_cost
    result["Current_Total_Cost"] = (demand / current_q) * ordering_cost + (current_q / 2.0) * holding_cost
    result["Annual_Cost_Savings"] = result["Current_Total_Cost"] - result["Optimal_Total_Cost"]
    result["Savings_Pct"] = np.where(
        result["Current_Total_Cost"] > 0,
        (result["Annual_Cost_Savings"] / result["Current_Total_Cost"]) * 100.0,
        0.0,
    )
    return result


def simulate_monthly_depletion(
    annual_demand: float,
    eoq: float,
    initial_stock: float | None = None,
    months: int = 12,
) -> pd.DataFrame:
    """Simulate monthly inventory depletion over 12 months under EOQ replenishment."""
    monthly_demand = annual_demand / 12.0
    current_inventory = float(eoq if initial_stock is None or initial_stock <= 0 else initial_stock)
    records = []

    for month in range(1, months + 1):
        current_inventory -= monthly_demand
        orders_placed = 0
        while current_inventory <= 0:
            current_inventory += eoq
            orders_placed += 1
        records.append(
            {
                "Month": month,
                "Remaining_Units": round(current_inventory, 2),
                "Orders_Placed": orders_placed,
                "Replenished": orders_placed > 0,
            }
        )

    return pd.DataFrame(records)


def compute_cost_curve(annual_demand: float, ordering_cost: float, holding_cost: float, eoq: float) -> pd.DataFrame:
    """Generate ordering, holding, and total cost curves around the optimal EOQ."""
    q_min = max(5.0, eoq * 0.25)
    q_max = max(q_min + 10.0, eoq * 2.5)
    quantities = np.linspace(q_min, q_max, 60)
    ordering = (annual_demand / quantities) * ordering_cost
    holding = (quantities / 2.0) * holding_cost
    total = ordering + holding
    return pd.DataFrame(
        {
            "Order_Quantity": quantities,
            "Ordering_Cost": ordering,
            "Holding_Cost": holding,
            "Total_Cost": total,
        }
    )
