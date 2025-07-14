#!/usr/bin/env python3
"""
Data Generation Script (V6.3 – Full Script with Debugged sim_reorders)

Generates synthetic data for a multi-hub apparel network:
 - 3 stores per major city, 2 per minor
 - Demand seasonality & store multipliers
 - Partial fulfillment + auto-reorder
 - Inventory-driven replenishment
 - Returns by category
 - Fill-rate logging for visibility
"""

import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from faker import Faker

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & SEEDING
# -----------------------------------------------------------------------------
CFG = {
    "NUM_STORES":        100,
    "NUM_PRODUCTS":      500,
    "NUM_SUPPLIERS":     20,
    "NUM_ORDERS":        6000,
    "START_DATE":        datetime(2024, 1, 1),
    "END_DATE":          datetime(2025, 6, 30),
    "OUT_DIR":           Path("End_to_End_SupplyChain_Data"),
    "SEED":              42,
    "REORDER_THRESHOLD": 300,
}

random.seed(CFG["SEED"])
np.random.seed(CFG["SEED"])
fake = Faker("en_IN")
fake.seed_instance(CFG["SEED"])

CFG["OUT_DIR"].mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(CFG["OUT_DIR"] / "data_gen.log", mode="w")
    ]
)
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 2. CONSTANTS & HELPERS
# -----------------------------------------------------------------------------
DIST_MATRIX: Dict[Tuple[str, str], int] = {
    ("HUB-DEL", "HUB-BOM"): 1400, ("HUB-BOM", "HUB-DEL"): 1400,
    ("HUB-DEL", "HUB-BLR"): 2150, ("HUB-BLR", "HUB-DEL"): 2150,
    ("HUB-BOM", "HUB-BLR"): 1000, ("HUB-BLR", "HUB-BOM"): 1000,
}

MODES = {
    "FTL": {"base": 5000, "km": 12, "wt": 25,   "vol": 1500, "speed": 45},
    "LTL": {"base": 2000, "km": 8,  "wt": 15,   "vol": 1000, "speed": 35},
}

QUARTER_SEAS = {
    1: {"Dress": 0.3, "Shoes": 0.4, "Accessories": 0.3},
    2: {"Dress": 0.4, "Shoes": 0.3, "Accessories": 0.3},
    3: {"Dress": 0.2, "Shoes": 0.3, "Accessories": 0.5},
    4: {"Dress": 0.5, "Shoes": 0.2, "Accessories": 0.3},
}

MONTH_SEAS = {
     1: 0.9,  2: 0.9,  3: 1.0,  4: 1.1,
     5: 1.0,  6: 0.9,  7: 1.2,  8: 1.1,
     9: 1.0, 10: 1.8, 11: 1.5, 12: 1.3
}

WEEKDAY_FACT = {0:1.0,1:1.0,2:1.0,3:1.0,4:1.1,5:1.2,6:1.3}

SUP_SPEC = {
    "Cotton Apparel": "Tiruppur",
    "Leather Goods":   "Kanpur",
    "Jewellery":       "Jaipur"
}

RETURN_PROBS = {
    "Dress":       {"Wrong Size": .4, "Wrong Colour": .2, "Customer Didn’t Like": .3, "Defective Product": .1},
    "Shoes":       {"Wrong Size": .3, "Wrong Colour": .1, "Customer Didn’t Like": .4, "Defective Product": .2},
    "Accessories": {"Customer Didn’t Like": .6, "Defective Product": .4}
}


def transport_cost(mode: str, dist: float, wt: float, vol: float) -> float:
    m = MODES[mode]
    return m["base"] + dist * m["km"] + max(wt * m["wt"], vol * m["vol"])


# -----------------------------------------------------------------------------
# 3. DIMENSION GENERATORS
# -----------------------------------------------------------------------------
def gen_hubs() -> pd.DataFrame:
    hubs = [
        {"Hub_ID": "HUB-DEL", "Hub_Name": "Delhi Apparel Hub",   "Specialty": "Dress"},
        {"Hub_ID": "HUB-BOM", "Hub_Name": "Mumbai Footwear Hub", "Specialty": "Shoes"},
        {"Hub_ID": "HUB-BLR", "Hub_Name": "Bangalore Acc. Hub",  "Specialty": "Accessories"},
    ]
    df = pd.DataFrame(hubs)
    log.info("Generated hubs: %d", len(df))
    return df


def gen_stores() -> Tuple[pd.DataFrame, Dict[str, float]]:
    placements = {
        "HUB-DEL": {"major": ["Delhi","Gurgaon","Noida","Jaipur"],
                    "minor": ["Agra","Mathura","Chandigarh","Kanpur","Bikaner","Dehradun","Lucknow","Kota","Patna","Jodhpur"]},
        "HUB-BOM": {"major": ["Pune","Thane","Ahmedabad","Surat"],
                    "minor": ["Nasik","Nagpur","Bhopal","Raipur","Indore","Rajkot","Panji"]},
        "HUB-BLR": {"major": ["Chennai","Koramangala","Hyderabad"],
                    "minor": ["Kochi","Coimbatore","Tirupati"]}
    }

    rows, mult_map = [], {}
    sid = 1
    for hub, grp in placements.items():
        for city in grp["major"]:
            for _ in range(3):
                sid_str = f"ST{sid:03d}"
                rows.append({"Store_ID": sid_str, "Home_Hub_ID": hub, "City": city})
                mult_map[sid_str] = random.uniform(1.5, 2.5)
                sid += 1
        for city in grp["minor"]:
            for _ in range(2):
                sid_str = f"ST{sid:03d}"
                rows.append({"Store_ID": sid_str, "Home_Hub_ID": hub, "City": city})
                mult_map[sid_str] = random.uniform(0.8, 1.2)
                sid += 1

    df = (
        pd.DataFrame(rows)
        .sample(frac=1, random_state=CFG["SEED"])
        .head(CFG["NUM_STORES"])
        .reset_index(drop=True)
    )
    log.info("Generated stores: %d", len(df))
    return df, mult_map


def gen_products() -> pd.DataFrame:
    specs = {
        "Dress":        (["Cotton Kurta","Silk Anarkali","Linen Shirt Dress","Georgette Saree"],
                         ["Aanya","Riya","Zoya","Elara"], (0.3,1.2), (0.002,0.008), (5,15)),
        "Shoes":        (["Leather Loafers","Canvas Sneakers","Ethnic Juttis","Block Heels"],
                         ["Vector","Orion","Nova","Apex"], (0.5,1.5), (0.005,0.015), (3,10)),
        "Accessories":  (["Leather Handbag","Silver Jhumkas","Analog Watch","Canvas Belt"],
                         ["Aura","Celeste","Eon","Luna"], (0.1,1.0), (0.001,0.020), (8,25)),
    }

    rows = []
    for i in range(1, CFG["NUM_PRODUCTS"] + 1):
        cat, (subs, names, w_rng, v_rng, d_rng) = random.choice(list(specs.items()))
        sub = random.choice(subs)
        name = f"{random.choice(names)} {sub}"
        w = round(random.uniform(*w_rng), 2)
        v = round(random.uniform(*v_rng), 4)
        if "Watch" in sub or "Jhumkas" in sub:
            w = round(random.uniform(0.05,0.2), 2)
            v = round(random.uniform(0.0001,0.0005), 4)

        rows.append({
            "SKU":             f"SKU{i:04d}",
            "Product_Name":    name,
            "Category":        cat,
            "Sub_Category":    sub,
            "Weight_kg":       w,
            "Volume_cbm":      v,
            "Base_Demand_Min": d_rng[0],
            "Base_Demand_Max": d_rng[1]
        })

    df = pd.DataFrame(rows)
    log.info("Generated products: %d", len(df))
    return df


def gen_suppliers() -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for i in range(1, CFG["NUM_SUPPLIERS"] + 1):
        spec = random.choice(list(SUP_SPEC.keys()))
        rows.append({
            "Supplier_ID":       f"SUP{i:03d}",
            "City":              SUP_SPEC[spec],
            "Specialty":         spec,
            "Reliability_Score": round(random.uniform(0.85, 0.98), 3)
        })
    full = pd.DataFrame(rows)
    slim = full.drop(columns=["Reliability_Score"])
    log.info("Generated suppliers: %d", len(slim))
    return slim, full


# -----------------------------------------------------------------------------
# 4. FACT SIMULATIONS
# -----------------------------------------------------------------------------
def sim_orders(
    stores: pd.DataFrame,
    products: pd.DataFrame,
    mult_map: Dict[str, float]
) -> pd.DataFrame:
    hub_map = {"Dress": "HUB-DEL", "Shoes": "HUB-BOM", "Accessories": "HUB-BLR"}
    recs = []
    for i in range(1, CFG["NUM_ORDERS"] + 1):
        dt = fake.date_time_between(CFG["START_DATE"], CFG["END_DATE"])
        store = stores.sample(1).iloc[0]
        prod = products.sample(1).iloc[0]

        base = random.randint(prod.Base_Demand_Min, prod.Base_Demand_Max)
        qtyf = (
            base
            * mult_map[store.Store_ID]
            * QUARTER_SEAS[(dt.month - 1)//3 + 1].get(prod.Category, 1.0)
            * MONTH_SEAS[dt.month]
            * WEEKDAY_FACT[dt.weekday()]
            * random.uniform(0.9, 1.1)
        )
        qty = max(1, int(round(qtyf)))

        recs.append({
            "Order_Line_ID":          f"OL{i:05d}",
            "Store_ID":               store.Store_ID,
            "SKU":                    prod.SKU,
            "Source_Hub_ID":          hub_map[prod.Category],
            "Order_Date":             dt.date(),
            "Required_Delivery_Date": (dt + timedelta(days=random.randint(3,10))).date(),
            "Quantity_Ordered":       qty
        })

    df = pd.DataFrame(recs)
    log.info("Orders: %d", len(df))
    return df


def sim_shipments(
    orders: pd.DataFrame,
    stores: pd.DataFrame,
    products: pd.DataFrame,
    hubs: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = orders.merge(products, on="SKU").merge(stores, on="Store_ID")

    # Lowered inventory to 0–300 for realistic stockouts
    inv = {
        (hub, sku): random.randint(0, 300)
        for hub in hubs.Hub_ID for sku in products.SKU
    }

    lines, unfulfilled = [], []
    for _, r in df.iterrows():
        key = (r.Source_Hub_ID, r.SKU)
        avail = inv.get(key, 0)

        if avail >= r.Quantity_Ordered:
            shipped_qty = r.Quantity_Ordered
            inv[key] -= shipped_qty
        else:
            shipped_qty = avail
            inv[key] = 0
            unfulfilled.append({
                "Order_Line_ID":        r.Order_Line_ID,
                "Unfulfilled_Quantity": r.Quantity_Ordered - shipped_qty
            })

        if shipped_qty > 0:
            lines.append({
                "Order_Line_ID":       r.Order_Line_ID,
                "Quantity_Shipped":    shipped_qty,
                "Order_Date":          r.Order_Date,
                "Source_Hub_ID":       r.Source_Hub_ID,
                "Home_Hub_ID":         r.Home_Hub_ID,
                "Store_ID":            r.Store_ID,
                "SKU":                 r.SKU,
                "Weight":              shipped_qty * r.Weight_kg,
                "Volume":              shipped_qty * r.Volume_cbm,
                "Dispatch_Timestamp":  datetime.combine(r.Order_Date, datetime.min.time())
            })

    shipped_df = pd.DataFrame(lines)
    unful_df   = pd.DataFrame(unfulfilled)

    # Fill-rate logging
    total_ordered = df["Quantity_Ordered"].sum()
    total_shipped = shipped_df["Quantity_Shipped"].sum()
    fill_rate = (total_shipped / total_ordered * 100) if total_ordered else 0
    log.info("Shipment fill rate: %.2f%%", fill_rate)
    log.info("Unfulfilled lines: %d", len(unful_df))

    # Build legs and link
    legs, link = [], []
    cnt = 1

    # Inter-hub legs
    grp1 = shipped_df[shipped_df.Source_Hub_ID != shipped_df.Home_Hub_ID] \
           .groupby(["Order_Date","Source_Hub_ID","Home_Hub_ID"])
    for (d, src, dst), g in grp1:
        leg_id = f"SL{cnt:06d}"
        wt, vol = g.Weight.sum(), g.Volume.sum()
        dist = DIST_MATRIX.get((src, dst), 1500)
        cost = transport_cost("FTL", dist, wt, vol)
        dep = datetime.combine(d, datetime.min.time()) + timedelta(hours=18)
        arr = dep + timedelta(hours=dist / MODES["FTL"]["speed"] + random.randint(6,48))

        legs.append({
            "Shipment_Leg_ID":     leg_id,
            "Leg_Type":            "Inter-Hub",
            "Transport_Mode":      "FTL",
            "Origin":              src,
            "Destination":         dst,
            "Dispatch_Timestamp":  dep,
            "Arrival_Timestamp":   arr,
            "Transportation_Cost": round(cost, 2)
        })
        for _, r2 in g.iterrows():
            link.append({
                "Shipment_Leg_ID":   leg_id,
                "Order_Line_ID":     r2.Order_Line_ID,
                "Quantity_Shipped":  r2.Quantity_Shipped
            })
        cnt += 1

    # Final-mile legs
    grp2 = shipped_df.groupby(["Order_Date","Home_Hub_ID","Store_ID"])
    for (d, hub, store), g in grp2:
        leg_id = f"SL{cnt:06d}"
        wt, vol = g.Weight.sum(), g.Volume.sum()
        dist = random.randint(30,150)
        cost = transport_cost("LTL", dist, wt, vol)
        dep = datetime.combine(d, datetime.min.time()) + timedelta(days=2,hours=10)
        arr = dep + timedelta(hours=dist / MODES["LTL"]["speed"] + random.randint(1,12))

        legs.append({
            "Shipment_Leg_ID":     leg_id,
            "Leg_Type":            "Final-Mile",
            "Transport_Mode":      "LTL",
            "Origin":              hub,
            "Destination":         store,
            "Dispatch_Timestamp":  dep,
            "Arrival_Timestamp":   arr,
            "Transportation_Cost": round(cost, 2)
        })
        for _, r3 in g.iterrows():
            link.append({
                "Shipment_Leg_ID":   leg_id,
                "Order_Line_ID":     r3.Order_Line_ID,
                "Quantity_Shipped":  r3.Quantity_Shipped
            })
        cnt += 1

    legs_df = pd.DataFrame(legs)
    link_df = pd.DataFrame(link)

    log.info("Shipment legs: %d", len(legs_df))
    return shipped_df, legs_df, link_df, unful_df


def sim_reorders(
    unful_df: pd.DataFrame,
    orders: pd.DataFrame,
    sup_full: pd.DataFrame,
    products: pd.DataFrame,
    hubs: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate inbound shipments to cover any unfulfilled order quantities.
    Merges unful_df with orders→SKU and products→Category.
    """
    hub_map = hubs.set_index("Specialty")["Hub_ID"].to_dict()

    df = (
        unful_df
        .merge(orders[["Order_Line_ID", "SKU"]], on="Order_Line_ID")
        .merge(products[["SKU", "Category"]], on="SKU")
    )

    recs = []
    for _, r in df.iterrows():
        sup = sup_full.sample(1).iloc[0]
        hub_id = hub_map[r.Category]
        prom = fake.date_time_between(CFG["START_DATE"], CFG["END_DATE"])
        delay = (
            timedelta(days=random.randint(1,5))
            if random.random() < (1 - sup.Reliability_Score)
            else timedelta(0)
        )
        recs.append({
            "Inbound_Shipment_ID":   f"INB_REO_{r.Order_Line_ID}",
            "Supplier_ID":           sup.Supplier_ID,
            "Destination_Hub_ID":    hub_id,
            "SKU":                   r.SKU,
            "Quantity_Received":     r.Unfulfilled_Quantity,
            "Promised_Arrival_Date": prom.date(),
            "Actual_Arrival_Date":   (prom + delay).date()
        })

    df_reorders = pd.DataFrame(recs)
    log.info("Reorder shipments: %d", len(df_reorders))
    return df_reorders


def sim_inbound(
    sup_full: pd.DataFrame,
    products: pd.DataFrame,
    hubs: pd.DataFrame,
    inv_levels: Dict[Tuple[str,str],int],
    reorder_threshold: int = CFG["REORDER_THRESHOLD"]
) -> pd.DataFrame:
    hub_map = hubs.set_index("Specialty")["Hub_ID"].to_dict()
    recs = []
    count = int(CFG["NUM_ORDERS"] * 0.7)

    for i in range(1, count + 1):
        sup  = sup_full.sample(1).iloc[0]
        prod = products.sample(1).iloc[0]
        hub  = hub_map[prod.Category]
        inv  = inv_levels.get((hub, prod.SKU), 0)

        if inv < reorder_threshold:
            qty = random.randint(500, 2500)
            prom = fake.date_time_between(CFG["START_DATE"], CFG["END_DATE"])
            delay = (
                timedelta(days=random.randint(1, 5))
                if random.random() < (1 - sup.Reliability_Score) else timedelta(0)
            )
            recs.append({
                "Inbound_Shipment_ID":   f"INB{i:05d}",
                "Supplier_ID":           sup.Supplier_ID,
                "Destination_Hub_ID":    hub,
                "SKU":                   prod.SKU,
                "Quantity_Received":     qty,
                "Promised_Arrival_Date": prom.date(),
                "Actual_Arrival_Date":   (prom + delay).date()
            })

    df_inbound = pd.DataFrame(recs)
    log.info("Inbound shipments (threshold): %d", len(df_inbound))
    return df_inbound


def sim_inventory(
    hubs: pd.DataFrame,
    products: pd.DataFrame,
    inbound: pd.DataFrame,
    shipped_df: pd.DataFrame
) -> pd.DataFrame:
    start = CFG["START_DATE"] - timedelta(days=5)
    dates = pd.date_range(start, CFG["END_DATE"], freq="D")

    inv_levels = {
        (hub, sku): random.randint(200, 800)
        for hub in hubs.Hub_ID for sku in products.SKU
    }

    records = []
    for dt in dates:
        today_in = inbound[inbound.Actual_Arrival_Date == dt.date()]
        for _, r in today_in.iterrows():
            key = (r.Destination_Hub_ID, r.SKU)
            inv_levels[key] = inv_levels.get(key, 0) + r.Quantity_Received

        today_sh = shipped_df[shipped_df.Dispatch_Timestamp.dt.date == dt.date()]
        for _, r in today_sh.iterrows():
            key = (r.Home_Hub_ID, r.SKU)
            inv_levels[key] = max(0, inv_levels.get(key, 0) - r.Quantity_Shipped)

        for (hub_id, sku), qty in inv_levels.items():
            records.append({
                "Snapshot_Date":    dt.date(),
                "Hub_ID":           hub_id,
                "SKU":              sku,
                "Quantity_On_Hand": qty
            })

    df_inv = pd.DataFrame(records)
    log.info("Inventory snapshots: %d", len(df_inv))
    return df_inv


def sim_returns(
    link_df: pd.DataFrame,
    orders: pd.DataFrame,
    products: pd.DataFrame
) -> pd.DataFrame:
    sample = link_df.sample(frac=0.07, random_state=CFG["SEED"])
    df = (sample
          .merge(orders, on="Order_Line_ID")
          .merge(products[["SKU", "Category"]], on="SKU"))

    recs = []
    for idx, r in df.iterrows():
        probs  = RETURN_PROBS.get(r.Category, {"Customer Didn’t Like": 1.0})
        reason = random.choices(list(probs), weights=list(probs.values()))[0]
        qty    = r.Quantity_Shipped
        if random.random() < 0.2 and qty > 1:
            qty = random.randint(1, qty - 1)
        if qty == 0:
            continue

        ret_date = (
            datetime.combine(r.Required_Delivery_Date, datetime.min.time())
            + timedelta(days=random.randint(1,21))
        )
        recs.append({
            "Return_ID":         f"RTN{idx+1:05d}",
            "Order_Line_ID":     r.Order_Line_ID,
            "Store_ID":          r.Store_ID,
            "SKU":               r.SKU,
            "Quantity_Returned": qty,
            "Return_Date":       ret_date.date(),
            "Return_Reason":     reason
        })

    df_ret = pd.DataFrame(recs)
    log.info("Returns: %d", len(df_ret))
    return df_ret


# -----------------------------------------------------------------------------
# 5. SAVE UTILITY
# -----------------------------------------------------------------------------
def save(df: pd.DataFrame, fname: str):
    path = CFG["OUT_DIR"] / fname
    df.to_csv(path, index=False)
    log.info("Saved %s (%d rows)", fname, len(df))


# -----------------------------------------------------------------------------
# 6. MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    hubs             = gen_hubs()
    stores, mult_map = gen_stores()
    products         = gen_products()
    sup_dim, sup_full = gen_suppliers()

    inv_levels = {
        (hub.Hub_ID, sku): random.randint(200, 800)
        for _, hub in hubs.iterrows() for sku in products.SKU
    }

    orders = sim_orders(stores, products, mult_map)

    shipped_df, legs_df, link_df, unful_df = sim_shipments(
        orders, stores, products, hubs
    )

    reorders_df = sim_reorders(
        unful_df, orders, sup_full, products, hubs
    )
    initial_inbound = sim_inbound(
        sup_full, products, hubs, inv_levels
    )

    all_inbound     = pd.concat([initial_inbound, reorders_df], ignore_index=True)
    inventory_snap  = sim_inventory(hubs, products, all_inbound, shipped_df)
    returns         = sim_returns(link_df, orders, products)

    save(hubs,             "dim_hubs.csv")
    save(stores,           "dim_stores.csv")
    save(products,         "dim_products.csv")
    save(sup_dim,          "dim_suppliers.csv")
    save(orders,           "fact_orders.csv")
    save(legs_df,          "fact_shipments.csv")
    save(link_df,          "link_shipment_orders.csv")
    save(unful_df,         "fact_unfulfilled_orders.csv")
    save(all_inbound,      "fact_inbound_shipments.csv")
    save(inventory_snap,   "fact_inventory_snapshot.csv")
    save(returns,          "fact_returns.csv")

    log.info("Data generation complete. Directory: %s", CFG["OUT_DIR"])


if __name__ == "__main__":
    main()
