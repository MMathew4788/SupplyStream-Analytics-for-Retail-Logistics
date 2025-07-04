#!/usr/bin/env python3
"""
Data Generation Script (V4.7 – Realistic Demand, No Extra Columns, Fixed Type Hints)

Generates synthetic data for a multi-hub apparel distribution network:
 - Dimensions: hubs, stores, products, suppliers
 - Facts: inbound shipments, inventory snapshots, orders, shipments, returns
 - Realistic demand modeling with internal multipliers, seasonality, no extra columns
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
    "NUM_STORES": 100,
    "NUM_PRODUCTS": 500,
    "NUM_SUPPLIERS": 20,
    "NUM_ORDERS": 6000,
    "START_DATE": datetime(2024, 1, 1),
    "END_DATE": datetime(2025, 6, 30),
    "OUT_DIR": Path("End_to_End_SupplyChain_Data"),
    "SEED": 42
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
        logging.FileHandler(CFG["OUT_DIR"] / "data_gen.log")
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
    "FTL": {"base": 5000, "km": 12, "wt": 25, "vol": 1500, "speed": 45},
    "LTL": {"base": 2000, "km": 8,  "wt": 15, "vol": 1000, "speed": 35},
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

WEEKDAY_FACT = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.1, 5: 1.2, 6: 1.3}

SUP_SPEC = {
    "Cotton Apparel": "Tiruppur",
    "Leather Goods":   "Kanpur",
    "Jewellery":       "Jaipur"
}

RETURN_PROBS = {
    "Dress":       {"Wrong Size": .4, "Wrong Colour": .2, "Customer Did Not Like": .3, "Defective Product": .1},
    "Shoes":       {"Wrong Size": .3, "Wrong Colour": .1, "Customer Did Not Like": .4, "Defective Product": .2},
    "Accessories": {"Customer Did Not Like": .6, "Defective Product": .4}
}


def transport_cost(mode: str, dist: float, wt: float, vol: float) -> float:
    """Compute transport cost based on mode, distance, weight, and volume."""
    m = MODES[mode]
    return m["base"] + dist * m["km"] + max(wt * m["wt"], vol * m["vol"])


# -----------------------------------------------------------------------------
# 3. DIMENSION GENERATORS
# -----------------------------------------------------------------------------
def gen_hubs() -> pd.DataFrame:
    hubs = [
        {"Hub_ID": "HUB-DEL", "Hub_Name": "Delhi Apparel Hub",       "Specialty": "Dress"},
        {"Hub_ID": "HUB-BOM", "Hub_Name": "Mumbai Footwear Hub",    "Specialty": "Shoes"},
        {"Hub_ID": "HUB-BLR", "Hub_Name": "Bangalore Acc. Hub",     "Specialty": "Accessories"},
    ]
    df = pd.DataFrame(hubs)
    log.info("Generated hubs: %d", len(df))
    return df


def gen_stores() -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Returns:
      - DataFrame of stores (Store_ID, Home_Hub_ID, City)
      - Internal demand-multiplier map {Store_ID: multiplier}
    """
    placements = {
        "HUB-DEL": {"major": ["Delhi", "Gurgaon", "Noida", "Jaipur"],
                    "minor": ["Agra", "Mathura", "Chandigarh", "Kanpur", "Bikaner",
                              "Dehradun", "Lucknow", "Kota", "Patna", "Jodhpur"]},
        "HUB-BOM": {"major": ["Pune", "Thane", "Ahmedabad", "Surat"],
                    "minor": ["Nasik", "Nagpur", "Bhopal", "Raipur", "Indore",
                              "Rajkot", "Panji"]},
        "HUB-BLR": {"major": ["Chennai", "Koramangala", "Hyderabad"],
                    "minor": ["Kochi", "Coimbatore", "Tirupati"]}
    }

    rows: list = []
    mult_map: Dict[str, float] = {}
    sid = 1

    for hub, grp in placements.items():
        # Major cities: 4 stores each
        for city in grp["major"]:
            for _ in range(4):
                store_id = f"ST{sid:03d}"
                mult = random.uniform(1.5, 2.5)
                rows.append({"Store_ID": store_id, "Home_Hub_ID": hub, "City": city})
                mult_map[store_id] = mult
                sid += 1
        # Minor cities: 1 store each
        for city in grp["minor"]:
            store_id = f"ST{sid:03d}"
            mult = random.uniform(0.8, 1.2)
            rows.append({"Store_ID": store_id, "Home_Hub_ID": hub, "City": city})
            mult_map[store_id] = mult
            sid += 1

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=CFG["SEED"]).reset_index(drop=True)
    df = df.head(CFG["NUM_STORES"])
    log.info("Generated stores: %d", len(df))
    return df, mult_map


def gen_products() -> pd.DataFrame:
    specs = {
        "Dress": {
            "subs": ["Cotton Kurta", "Silk Anarkali", "Linen Shirt Dress", "Georgette Saree"],
            "names": ["Aanya", "Riya", "Zoya", "Elara"],
            "w": (0.3, 1.2), "v": (0.002, 0.008), "d": (5, 15)
        },
        "Shoes": {
            "subs": ["Leather Loafers", "Canvas Sneakers", "Ethnic Juttis", "Block Heels"],
            "names": ["Vector", "Orion", "Nova", "Apex"],
            "w": (0.5, 1.5), "v": (0.005, 0.015), "d": (3, 10)
        },
        "Accessories": {
            "subs": ["Leather Handbag", "Silver Jhumkas", "Analog Watch", "Canvas Belt"],
            "names": ["Aura", "Celeste", "Eon", "Luna"],
            "w": (0.1, 1.0), "v": (0.001, 0.020), "d": (8, 25)
        }
    }

    rows: list = []
    for i in range(1, CFG["NUM_PRODUCTS"] + 1):
        cat = random.choice(list(specs.keys()))
        sp = specs[cat]
        sub = random.choice(sp["subs"])
        name = f"{random.choice(sp['names'])} {sub}"
        w = round(random.uniform(*sp["w"]), 2)
        v = round(random.uniform(*sp["v"]), 4)
        # Very small items override
        if "Watch" in sub or "Jhumkas" in sub:
            w = round(random.uniform(0.05, 0.2), 2)
            v = round(random.uniform(0.0001, 0.0005), 4)

        rows.append({
            "SKU": f"SKU{i:04d}",
            "Product_Name": name,
            "Category": cat,
            "Sub_Category": sub,
            "Weight_kg": w,
            "Volume_cbm": v,
            "Base_Demand_Min": sp["d"][0],
            "Base_Demand_Max": sp["d"][1]
        })

    df = pd.DataFrame(rows)
    log.info("Generated products: %d", len(df))
    return df


def gen_suppliers() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (dim_suppliers, full_suppliers_with_reliability)."""
    rows: list = []
    for i in range(1, CFG["NUM_SUPPLIERS"] + 1):
        spec = random.choice(list(SUP_SPEC.keys()))
        rows.append({
            "Supplier_ID": f"SUP{i:03d}",
            "City": SUP_SPEC[spec],
            "Specialty": spec,
            "Reliability_Score": round(random.uniform(0.85, 0.98), 3)
        })

    full = pd.DataFrame(rows)
    slim = full.drop(columns=["Reliability_Score"])
    log.info("Generated suppliers: %d", len(slim))
    return slim, full


# -----------------------------------------------------------------------------
# 4. FACT SIMULATIONS
# -----------------------------------------------------------------------------
def sim_inbound(sup_full: pd.DataFrame, prods: pd.DataFrame, hubs: pd.DataFrame) -> pd.DataFrame:
    hub_map = hubs.set_index("Specialty")["Hub_ID"].to_dict()
    records: list = []
    count = int(CFG["NUM_ORDERS"] * 0.7)

    for i in range(1, count + 1):
        sup = sup_full.sample(1).iloc[0]
        prod = prods.sample(1).iloc[0]
        hub_id = hub_map[prod.Category]
        prom = fake.date_time_between(CFG["START_DATE"], CFG["END_DATE"])
        delay = timedelta(days=random.randint(1, 5)) if random.random() < (1 - sup.Reliability_Score) else timedelta(0)

        records.append({
            "Inbound_Shipment_ID": f"INB{i:05d}",
            "Supplier_ID": sup.Supplier_ID,
            "Destination_Hub_ID": hub_id,
            "SKU": prod.SKU,
            "Quantity_Received": random.randint(500, 2500),
            "Promised_Arrival_Date": prom.date(),
            "Actual_Arrival_Date": (prom + delay).date()
        })

    df = pd.DataFrame(records)
    log.info("Inbound shipments: %d", len(df))
    return df


def sim_inventory(hubs: pd.DataFrame, prods: pd.DataFrame, inbound: pd.DataFrame) -> pd.DataFrame:
    start = CFG["START_DATE"] - timedelta(days=5)
    dates = pd.date_range(start, CFG["END_DATE"], freq="D")

    inv_levels: Dict[Tuple[str, str], int] = {
        (hub, sku): random.randint(200, 800)
        for hub in hubs.Hub_ID
        for sku in prods.SKU
    }

    records: list = []
    for dt in dates:
        today_in = inbound[inbound.Actual_Arrival_Date == dt.date()]
        for _, row in today_in.iterrows():
            key = (row.Destination_Hub_ID, row.SKU)
            inv_levels[key] = inv_levels.get(key, 0) + row.Quantity_Received

        for (hub_id, sku), qty in list(inv_levels.items()):
            consumed = random.randint(0, 10)
            inv_levels[(hub_id, sku)] = max(0, qty - consumed)
            records.append({
                "Snapshot_Date": dt.date(),
                "Hub_ID": hub_id,
                "SKU": sku,
                "Quantity_On_Hand": inv_levels[(hub_id, sku)]
            })

    df = pd.DataFrame(records)
    log.info("Inventory snapshots: %d records", len(df))
    return df


def sim_orders(stores: pd.DataFrame, prods: pd.DataFrame, mult_map: Dict[str, float]) -> pd.DataFrame:
    hub_for = {"Dress": "HUB-DEL", "Shoes": "HUB-BOM", "Accessories": "HUB-BLR"}
    records: list = []

    for i in range(1, CFG["NUM_ORDERS"] + 1):
        dt = fake.date_time_between(CFG["START_DATE"], CFG["END_DATE"])
        store = stores.sample(1).iloc[0]
        prod  = prods.sample(1).iloc[0]

        base = random.randint(prod.Base_Demand_Min, prod.Base_Demand_Max)
        q = (base
             * mult_map[store.Store_ID]
             * QUARTER_SEAS[(dt.month - 1)//3 + 1].get(prod.Category, 1.0)
             * MONTH_SEAS[dt.month]
             * WEEKDAY_FACT[dt.weekday()]
             * random.uniform(0.9, 1.1))
        qty = max(1, int(round(q)))

        records.append({  
            "Order_Line_ID": f"OL{i:05d}",
            "Store_ID": store.Store_ID,
            "SKU": prod.SKU,
            "Source_Hub_ID": hub_for[prod.Category],
            "Order_Date": dt.date(),
            "Required_Delivery_Date": (dt + timedelta(days=random.randint(3, 10))).date(),
            "Quantity_Ordered": qty
        })

    df = pd.DataFrame(records)
    log.info("Orders: %d", len(df))
    return df


def sim_shipments(
    orders: pd.DataFrame,
    stores: pd.DataFrame,
    prods: pd.DataFrame,
    hubs: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = orders.merge(prods, on="SKU").merge(stores, on="Store_ID")

    inv = {
        (hub, sku): random.randint(300, 1500)
        for hub in hubs.Hub_ID for sku in prods.SKU
    }

    lines: list = []
    for _, r in df.iterrows():
        key = (r.Source_Hub_ID, r.SKU)
        avail = inv.get(key, 0)
        sh = min(avail, r.Quantity_Ordered)
        inv[key] = avail - sh
        if sh > 0:
            lines.append({
                "Order_Line_ID": r.Order_Line_ID,
                "Quantity_Shipped": sh,
                "Order_Date": r.Order_Date,
                "Source_Hub_ID": r.Source_Hub_ID,
                "Home_Hub_ID": r.Home_Hub_ID,
                "Weight": sh * r.Weight_kg,
                "Volume": sh * r.Volume_cbm,
                "Store_ID": r.Store_ID
            })

    shipped = pd.DataFrame(lines)
    legs, link = [], []
    cnt = 1

    # Inter-hub legs
    grp = shipped[shipped.Source_Hub_ID != shipped.Home_Hub_ID] \
              .groupby(["Order_Date", "Source_Hub_ID", "Home_Hub_ID"])
    for (d, src, dst), g in grp:
        leg_id = f"SL{cnt:06d}"
        wt, vol = g.Weight.sum(), g.Volume.sum()
        dist = DIST_MATRIX.get((src, dst), 1500)
        cost = transport_cost("FTL", dist, wt, vol)
        dep = datetime.combine(d, datetime.min.time()) + timedelta(hours=18)
        arr = dep + timedelta(hours=dist / MODES["FTL"]["speed"] + random.randint(6, 48))

        legs.append({
            "Shipment_Leg_ID":       leg_id,
            "Leg_Type":              "Inter-Hub",
            "Transport_Mode":        "FTL",
            "Origin":                src,
            "Destination":           dst,
            "Dispatch_Timestamp":    dep,
            "Arrival_Timestamp":     arr,
            "Transportation_Cost":   round(cost, 2)
        })

        for _, r2 in g.iterrows():
            link.append({
                "Shipment_Leg_ID": leg_id,
                "Order_Line_ID":   r2.Order_Line_ID,
                "Quantity_Shipped": r2.Quantity_Shipped
            })

        cnt += 1

    # Final-mile legs
    grp2 = shipped.groupby(["Order_Date", "Home_Hub_ID", "Store_ID"])
    for (d, hub, store), g in grp2:
        leg_id = f"SL{cnt:06d}"
        wt, vol = g.Weight.sum(), g.Volume.sum()
        dist = random.randint(30, 150)
        cost = transport_cost("LTL", dist, wt, vol)
        dep = datetime.combine(d, datetime.min.time()) + timedelta(days=2, hours=10)
        arr = dep + timedelta(hours=dist / MODES["LTL"]["speed"] + random.randint(1, 12))

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
                "Shipment_Leg_ID":  leg_id,
                "Order_Line_ID":    r3.Order_Line_ID,
                "Quantity_Shipped": r3.Quantity_Shipped
            })

        cnt += 1

    log.info("Shipment legs: %d", len(legs))
    return pd.DataFrame(legs), pd.DataFrame(link)


def sim_returns(
    link: pd.DataFrame,
    orders: pd.DataFrame,
    prods: pd.DataFrame
) -> pd.DataFrame:
    sample = link.sample(frac=0.07, random_state=CFG["SEED"])
    df = (sample
          .merge(orders, on="Order_Line_ID")
          .merge(prods[["SKU", "Category"]], on="SKU"))

    recs: list = []
    for idx, r in df.iterrows():
        probs = RETURN_PROBS.get(r.Category, {"Customer Did Not Like": 1.0})
        reason = random.choices(list(probs), weights=list(probs.values()))[0]

        qty = r.Quantity_Shipped
        if random.random() < 0.2 and qty > 1:
            qty = random.randint(1, qty - 1)
        if qty == 0:
            continue

        ret_date = (datetime.combine(r.Required_Delivery_Date, datetime.min.time())
                    + timedelta(days=random.randint(1, 21)))

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
    hubs,        = (gen_hubs(),)
    stores, mult = gen_stores()
    products     = gen_products()
    sup, sup_full= gen_suppliers()

    inbound      = sim_inbound(sup_full, products, hubs)
    inventory    = sim_inventory(hubs, products, inbound)
    orders       = sim_orders(stores, products, mult)
    shipments, link = sim_shipments(orders, stores, products, hubs)
    returns      = sim_returns(link, orders, products)

    save(hubs,     "dim_hubs.csv")
    save(stores,   "dim_stores.csv")  # no extra column
    save(products, "dim_products.csv")
    save(sup,      "dim_suppliers.csv")

    save(inventory, "fact_inventory_snapshot.csv")
    save(inbound,   "fact_inbound_shipments.csv")
    save(orders,    "fact_orders.csv")
    save(shipments, "fact_shipments.csv")
    save(link,      "link_shipment_orders.csv")
    save(returns,   "fact_returns.csv")

    log.info("Data generation complete. Output directory: %s", CFG["OUT_DIR"])


if __name__ == "__main__":
    main()
