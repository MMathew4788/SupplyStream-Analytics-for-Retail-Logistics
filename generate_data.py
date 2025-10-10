#!/usr/bin/env python3

import logging, random
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from faker import Faker

# ----------------------------------------------------------------------------
# 1. CONFIG & SEED
# ----------------------------------------------------------------------------
CFG = {
    "NUM_STORES":    100,
    "NUM_PRODUCTS":  500,
    "NUM_SUPPLIERS": 20,
    "NUM_ORDERS":    50000,
    "START_DATE":    datetime(2022, 1, 1),
    "END_DATE":      datetime(2025, 6, 30),
    "OUT_DIR":       Path("SupplyChain_Data"),
    "SEED":          42,
    "SERVICE_FACTOR":1.65,
    "LT_MEAN":       7,
    "LT_STD":        2,
    # Truck capacity constraints
    "TRUCK_MAX_WEIGHT_KG": 10000,
    "TRUCK_MAX_VOLUME_CBM": 60,
    "TRUCK_MIN_LOAD_PCT": 0.30,
    # Operational parameters
    "TRUCK_SPEED_KM_PER_DAY": 500,
    "CROSSDOCK_PROCESSING_HOURS": 8,
    "EXPRESS_URGENCY_DAYS": 3,
}

random.seed(CFG["SEED"])
np.random.seed(CFG["SEED"])
fake = Faker("en_IN"); fake.seed_instance(CFG["SEED"])
CFG["OUT_DIR"].mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(CFG["OUT_DIR"]/ "data_gen.log", mode="w")
    ]
)
log = logging.getLogger()

# ----------------------------------------------------------------------------
# 2. CONSTANTS & HELPERS
# ----------------------------------------------------------------------------
QUARTER_SEAS = {
    1: {"Dress": .3, "Shoes": .4, "Accessories": .3},
    2: {"Dress": .4, "Shoes": .3, "Accessories": .3},
    3: {"Dress": .2, "Shoes": .3, "Accessories": .5},
    4: {"Dress": .5, "Shoes": .2, "Accessories": .3},
}
MONTH_SEAS = {1:0.9,2:0.9,3:1.0,4:1.1,5:1.0,6:0.9,7:1.2,8:1.1,9:1.0,10:1.8,11:1.5,12:1.3}
WEEKDAY_FACT = {i:(1.3 if i==6 else 1.2 if i==5 else 1.1 if i==4 else 1.0) for i in range(7)}

SUP_SPEC        = {"Cotton Apparel":"Tiruppur","Leather Goods":"Kanpur","Jewellery":"Jaipur"}
CAT_TO_SUP_SPEC = {"Dress":"Cotton Apparel","Shoes":"Leather Goods","Accessories":"Jewellery"}

RETURN_PROBS = {
    "Dress":       {"Wrong Size":0.6,"Wrong Colour":0.4},
    "Shoes":       {"Wrong Size":0.6,"Wrong Colour":0.4},
    "Accessories": {"Wrong Size":0.5,"Wrong Colour":0.5}
}

# Inter-hub distances (km)
HUB_DIST_MATRIX = {
    ("HUB-DEL", "HUB-BOM"): 1400,
    ("HUB-BOM", "HUB-DEL"): 1400,
    ("HUB-DEL", "HUB-BLR"): 2100,
    ("HUB-BLR", "HUB-DEL"): 2100,
    ("HUB-BOM", "HUB-BLR"): 1000,
    ("HUB-BLR", "HUB-BOM"): 1000,
}

def sample_lead_time() -> int:
    lt = int(np.random.normal(CFG["LT_MEAN"], CFG["LT_STD"]))
    return max(3, min(14, lt))

def courier_cost(dist_km:int, wt_kg:float, vol_cbm:float, is_inter_hub:bool=False) -> float:
    vol_kg = vol_cbm * 200
    chg_kg = max(wt_kg, vol_kg)
    cost = 300 + 15*dist_km + 8*chg_kg
    if is_inter_hub:
        cost *= 1.2
    return round(cost, 2)

def calculate_transit_days(distance_km: int, is_inter_hub: bool = False) -> int:
    """Calculate realistic transit time based on distance."""
    if is_inter_hub:
        # Truck speed: 500 km/day, minimum 1 day
        days = max(1, int(np.ceil(distance_km / CFG["TRUCK_SPEED_KM_PER_DAY"])))
        # Add small random variation (±1 day for delays/traffic)
        days += random.choice([-1, 0, 0, 0, 1])
        return max(1, days)
    else:
        # Final-mile courier: faster for short distances
        if distance_km <= 50:
            return 1
        elif distance_km <= 100:
            return random.choice([1, 2])
        else:
            return 2

# ----------------------------------------------------------------------------
# 3. DIMENSIONS
# ----------------------------------------------------------------------------
def gen_hubs() -> pd.DataFrame:
    df = pd.DataFrame([
        {"Hub_ID":"HUB-DEL","Hub_Name":"Delhi Apparel Hub","Specialty":"Dress"},
        {"Hub_ID":"HUB-BOM","Hub_Name":"Mumbai Footwear Hub","Specialty":"Shoes"},
        {"Hub_ID":"HUB-BLR","Hub_Name":"Bangalore Acc. Hub","Specialty":"Accessories"},
    ])
    log.info("Hubs: %d", len(df))
    return df

def gen_stores() -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, int]]:
    """Generate stores with distance from home hub."""
    placements = {
        "HUB-DEL": (["Delhi","Gurgaon","Noida","Jaipur"], ["Agra","Lucknow","Kanpur","Patna","Mathura"]),
        "HUB-BOM": (["Pune","Surat","Ahmedabad","Thane"], ["Nagpur","Bhopal","Raipur","Indore","Rajkot"]),
        "HUB-BLR": (["Bengaluru","Chennai","Hyderabad"],   ["Coimbatore","Kochi","Tiruputi"]),
    }
    rows, mult, distances = [], {}, {}
    sid = 1
    for hub, (maj, minr) in placements.items():
        for city in maj:
            for _ in range(3):
                s = f"ST{sid:03d}"
                dist = random.randint(20, 50)
                rows.append({"Store_ID":s,"Home_Hub_ID":hub,"City":city,"Store_Type":"major"})
                mult[s] = random.uniform(1.5, 2.5)
                distances[s] = dist
                sid += 1
        for city in minr:
            for _ in range(2):
                s = f"ST{sid:03d}"
                dist = random.randint(60, 150)
                rows.append({"Store_ID":s,"Home_Hub_ID":hub,"City":city,"Store_Type":"minor"})
                mult[s] = random.uniform(0.8, 1.2)
                distances[s] = dist
                sid += 1

    df = (pd.DataFrame(rows)
          .sample(frac=1, random_state=CFG["SEED"])
          .head(CFG["NUM_STORES"])
          .reset_index(drop=True))
    log.info("Stores: %d", len(df))
    return df, mult, distances

def gen_products() -> pd.DataFrame:
    specs = {
        "Dress": (
            ["Cotton Kurta","Silk Anarkali","Linen Shirt Dress","Georgette Saree"],
            ["Aanya","Riya","Zoya","Elara"], (0.3,1.2), (0.002,0.008), (5,15)
        ),
        "Shoes": (
            ["Leather Loafers","Canvas Sneakers","Ethnic Juttis","Block Heels"],
            ["Vector","Orion","Nova","Apex"], (0.5,1.5), (0.005,0.015), (3,10)
        ),
        "Accessories": (
            ["Leather Handbag","Silver Jhumkas","Analog Watch","Canvas Belt"],
            ["Aura","Celeste","Eon","Luna"], (0.1,1.0), (0.001,0.020), (8,25)
        ),
    }
    rows = []
    for i in range(1, CFG["NUM_PRODUCTS"]+1):
        cat, (subs, names, w_rng, v_rng, d_rng) = random.choice(list(specs.items()))
        sub  = random.choice(subs)
        name = f"{random.choice(names)} {sub}"
        w    = round(random.uniform(*w_rng), 2)
        v    = round(random.uniform(*v_rng), 4)
        rows.append({
            "SKU": f"SKU{i:04d}",
            "Product_Name": name,
            "Category": cat,
            "Sub_Category": sub,
            "Weight_kg": w,
            "Volume_cbm": v,
            "Base_Demand_Min": d_rng[0],
            "Base_Demand_Max": d_rng[1],
        })
    df = pd.DataFrame(rows)
    df["Avg_Daily_Demand"] = ((df.Base_Demand_Min + df.Base_Demand_Max)/2)*0.80
    df["ROP"] = (
        df.Avg_Daily_Demand*CFG["LT_MEAN"]
        + CFG["SERVICE_FACTOR"]*df.Avg_Daily_Demand*CFG["LT_STD"]
    ).round().astype(int)
    df["Target_Level"] = (df.ROP + 7*df.Avg_Daily_Demand).round().astype(int)
    log.info("Products: %d", len(df))
    return df

def gen_suppliers() -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for i in range(1, CFG["NUM_SUPPLIERS"]+1):
        spec = random.choice(list(SUP_SPEC.keys()))
        rows.append({
            "Supplier_ID":       f"SUP{i:03d}",
            "City":              SUP_SPEC[spec],
            "Specialty":         spec,
            "Reliability_Score": round(random.uniform(0.85, 0.98), 3)
        })
    full = pd.DataFrame(rows)
    slim = full.drop(columns=["Reliability_Score"])
    log.info("Suppliers: %d", len(slim))
    return slim, full

# ----------------------------------------------------------------------------
# 4. INITIAL INVENTORY
# ----------------------------------------------------------------------------
def seed_inventory(hubs: pd.DataFrame, prods: pd.DataFrame) -> Dict[Tuple[str, str], int]:
    spec_to_hub = hubs.set_index("Specialty")["Hub_ID"].to_dict()
    target_map  = prods.set_index("SKU")["Target_Level"].to_dict()
    inv: Dict[Tuple[str, str], int] = {}
    for _, r in prods.iterrows():
        tgt = target_map[r.SKU]
        inv[(spec_to_hub[r.Category], r.SKU)] = int(tgt * random.uniform(0.9, 1.1))
    log.info("Seeded inventory: %d hub-SKU pairs", len(inv))
    return inv

# ----------------------------------------------------------------------------
# 5. RETURNS HANDLER
# ----------------------------------------------------------------------------
def handle_returns(
    order_lines: List[Dict],
    arrival_date: date,
    sku_to_cat: Dict[str, str],
    next_rt: int,
    rets: List[Dict],
    inv_levels: Dict[Tuple[str, str], int],
    pending_returns: List[Dict]
) -> int:
    for ol in order_lines:
        shipped_qty = ol.get("Quantity_Shipped", 0)
        if shipped_qty <= 0:
            continue
        return_rate = random.uniform(0.03, 0.06)
        if random.random() < return_rate:
            ret_q = random.randint(1, shipped_qty)
            cat   = sku_to_cat[ol["SKU"]]
            reason = random.choices(
                list(RETURN_PROBS[cat].keys()),
                weights=list(RETURN_PROBS[cat].values())
            )[0]
            ret_date = arrival_date + timedelta(days=random.randint(1, 21))
            rets.append({
                "Return_ID":         f"RET{next_rt:05d}",
                "Order_Line_ID":     ol["Order_Line_ID"],
                "SKU":               ol["SKU"],
                "Quantity_Returned": ret_q,
                "Return_Date":       ret_date,
                "Return_Reason":     reason
            })
            pending_returns.append({
                "Return_Date":  ret_date,
                "Hub_ID":       ol["Source_Hub_ID"],
                "SKU":          ol["SKU"],
                "Quantity":     ret_q
            })
            next_rt += 1
    return next_rt

# ----------------------------------------------------------------------------
# 6. IMPROVED TRUCK CONSOLIDATION WITH URGENCY LOGIC
# ----------------------------------------------------------------------------
class TruckLoadManager:
    """Manages consolidation of inter-hub shipments with urgency-based priority."""

    def __init__(self, sku_to_wt: Dict, sku_to_vol: Dict):
        self.sku_to_wt = sku_to_wt
        self.sku_to_vol = sku_to_vol
        self.pending_loads: Dict[Tuple[str, str, date], List[Dict]] = defaultdict(list)

    def add_shipment_item(self, origin: str, destination: str, dispatch_date: date, 
                          order_line_id: str, sku: str, quantity: int, order_id: str,
                          required_delivery_date: date, is_express: bool = False):
        """Add an item to pending consolidated shipment with urgency info."""
        key = (origin, destination, dispatch_date)
        self.pending_loads[key].append({
            "Order_Line_ID": order_line_id,
            "SKU": sku,
            "Quantity": quantity,
            "Order_ID": order_id,
            "Required_Delivery_Date": required_delivery_date,
            "Is_Express": is_express
        })

    def get_load_metrics(self, items: List[Dict]) -> Tuple[float, float]:
        total_weight = sum(item["Quantity"] * self.sku_to_wt[item["SKU"]] for item in items)
        total_volume = sum(item["Quantity"] * self.sku_to_vol[item["SKU"]] for item in items)
        return total_weight, total_volume

    def check_capacity(self, weight: float, volume: float) -> bool:
        return (weight <= CFG["TRUCK_MAX_WEIGHT_KG"] and 
                volume <= CFG["TRUCK_MAX_VOLUME_CBM"])

    def check_minimum_threshold(self, weight: float, volume: float) -> bool:
        min_weight = CFG["TRUCK_MAX_WEIGHT_KG"] * CFG["TRUCK_MIN_LOAD_PCT"]
        min_volume = CFG["TRUCK_MAX_VOLUME_CBM"] * CFG["TRUCK_MIN_LOAD_PCT"]
        return weight >= min_weight or volume >= min_volume

    def has_express_items(self, items: List[Dict]) -> bool:
        """Check if any items in the load are express/urgent."""
        return any(item.get("Is_Express", False) for item in items)

    def process_ready_trucks(self, current_date: date, next_leg: int, 
                           legs: List[Dict], links: List[Dict],
                           pending_crossdock: Dict, delay_buffer: List[Dict]) -> int:
        routes_to_remove = []

        for (origin, destination, dispatch_date), items in self.pending_loads.items():
            if dispatch_date > current_date:
                continue

            total_weight, total_volume = self.get_load_metrics(items)
            has_express = self.has_express_items(items)

            # Express shipments bypass minimum threshold
            if not has_express and not self.check_minimum_threshold(total_weight, total_volume):
                next_dispatch = dispatch_date + timedelta(days=1)
                for item in items:
                    delay_buffer.append({
                        "origin": origin,
                        "destination": destination,
                        "dispatch_date": next_dispatch,
                        "item": item
                    })
                routes_to_remove.append((origin, destination, dispatch_date))
                continue

            trucks = self._split_into_trucks(items)

            for truck_items in trucks:
                truck_weight, truck_volume = self.get_load_metrics(truck_items)
                lid = f"SL{next_leg:06d}"
                next_leg += 1

                dist_key = (origin, destination)
                dist = HUB_DIST_MATRIX.get(dist_key, random.randint(200, 1000))

                # Realistic transit time based on distance
                transit_days = calculate_transit_days(dist, is_inter_hub=True)
                arrival_date = dispatch_date + timedelta(days=transit_days)

                cost = courier_cost(dist, truck_weight, truck_volume, is_inter_hub=True)

                leg = {
                    "Shipment_Leg_ID": lid,
                    "Leg_Type": "Inter-Hub",
                    "Transport_Mode": "Truck",
                    "Origin": origin,
                    "Destination": destination,
                    "Dispatch_Date": dispatch_date,
                    "Arrival_Date": arrival_date,
                    "Transportation_Cost": cost
                }
                legs.append(leg)

                for item in truck_items:
                    links.append({
                        "Shipment_Leg_ID": lid,
                        "Order_Line_ID": item["Order_Line_ID"],
                        "Quantity_Shipped": item["Quantity"]
                    })

                    pending_crossdock.setdefault(destination, []).append({
                        "Order_ID": item["Order_ID"],
                        "SKU": item["SKU"],
                        "Quantity": item["Quantity"],
                        "Arrival_Date": arrival_date,
                        "From_Hub": origin,
                        "Required_Delivery_Date": item["Required_Delivery_Date"]
                    })

            routes_to_remove.append((origin, destination, dispatch_date))

        for key in routes_to_remove:
            del self.pending_loads[key]

        return next_leg

    def _split_into_trucks(self, items: List[Dict]) -> List[List[Dict]]:
        trucks = []
        current_truck = []
        current_weight = 0.0
        current_volume = 0.0

        for item in items:
            item_weight = item["Quantity"] * self.sku_to_wt[item["SKU"]]
            item_volume = item["Quantity"] * self.sku_to_vol[item["SKU"]]

            if (current_weight + item_weight > CFG["TRUCK_MAX_WEIGHT_KG"] or
                current_volume + item_volume > CFG["TRUCK_MAX_VOLUME_CBM"]):
                if current_truck:
                    trucks.append(current_truck)
                    current_truck = []
                    current_weight = 0.0
                    current_volume = 0.0

            current_truck.append(item)
            current_weight += item_weight
            current_volume += item_volume

        if current_truck:
            trucks.append(current_truck)

        return trucks

# ----------------------------------------------------------------------------
# 7. BACKORDER TRACKING
# ----------------------------------------------------------------------------
def track_backorder(order_line: Dict, backorders: List[Dict], next_bo: int) -> int:
    """Track unfulfilled order lines as backorders."""
    backorders.append({
        "Backorder_ID": f"BO{next_bo:05d}",
        "Order_Line_ID": order_line["Order_Line_ID"],
        "Order_ID": order_line["Order_ID"],
        "SKU": order_line["SKU"],
        "Quantity_Backordered": order_line["Quantity_Ordered"] - order_line.get("Quantity_Shipped", 0),
        "Backorder_Date": order_line["Order_Date"],
        "Status": "Pending"
    })
    return next_bo + 1

# ----------------------------------------------------------------------------
# 8. MAIN SIMULATION
# ----------------------------------------------------------------------------
def main():
    hubs                = gen_hubs()
    stores, mult_map, store_distances = gen_stores()
    prods               = gen_products()
    sup_dim, sup_full   = gen_suppliers()

    hub_map    = hubs.set_index("Specialty")["Hub_ID"].to_dict()
    sku_to_cat = prods.set_index("SKU")["Category"].to_dict()
    sku_to_rop = prods.set_index("SKU")["ROP"].to_dict()
    sku_to_tgt = prods.set_index("SKU")["Target_Level"].to_dict()
    sku_to_wt  = prods.set_index("SKU")["Weight_kg"].to_dict()
    sku_to_vol = prods.set_index("SKU")["Volume_cbm"].to_dict()

    inv_levels       = seed_inventory(hubs, prods)
    pending_inb: List[Dict] = []
    pending_crossdock: Dict[str, List[Dict]] = {}
    pending_returns: List[Dict] = []

    truck_manager = TruckLoadManager(sku_to_wt, sku_to_vol)
    delay_buffer: List[Dict] = []

    next_ord, next_ol, next_leg, next_inb, next_rt, next_bo = 1, 1, 1, 1, 1, 1

    orders: List[Dict] = []
    legs:   List[Dict] = []
    links:  List[Dict] = []
    inbs:   List[Dict] = []
    rets:   List[Dict] = []
    snaps:  List[Dict] = []
    backorders: List[Dict] = []

    total_days = (CFG["END_DATE"] - CFG["START_DATE"]).days + 1
    avg_daily  = CFG["NUM_ORDERS"] / total_days

    years = range(CFG["START_DATE"].year, CFG["END_DATE"].year + 1)
    year_growth = {year: random.uniform(-0.05, 0.15) for year in years}
    log.info("Yearly growth rates (YOY): %s", year_growth)

    cum_multiplier: Dict[int, float] = {}
    running = 1.0
    for y in years:
        running *= (1.0 + year_growth[y])
        cum_multiplier[y] = running

    calendar = pd.date_range(CFG["START_DATE"], CFG["END_DATE"], freq="D")
    for ts in calendar:
        today = ts.date()

        # Process delayed items
        for delayed in delay_buffer[:]:
            truck_manager.add_shipment_item(
                delayed["origin"], delayed["destination"], 
                delayed["dispatch_date"], delayed["item"]["Order_Line_ID"],
                delayed["item"]["SKU"], delayed["item"]["Quantity"],
                delayed["item"]["Order_ID"], 
                delayed["item"]["Required_Delivery_Date"],
                delayed["item"]["Is_Express"]
            )
        delay_buffer.clear()

        # Apply returns
        for pr in pending_returns[:]:
            if pr["Return_Date"] == today:
                key = (pr["Hub_ID"], pr["SKU"])
                inv_levels[key] = inv_levels.get(key, 0) + pr["Quantity"]
                pending_returns.remove(pr)

        # Inbound arrivals
        for ev in pending_inb[:]:
            if ev["Actual_Arrival_Date"] == today:
                inv_levels[(ev["Destination_Hub_ID"], ev["SKU"])] = inv_levels.get((ev["Destination_Hub_ID"], ev["SKU"]), 0) + ev["Quantity_Received"]
                inbs.append({**ev, "Actual_Arrival_Date": today})
                pending_inb.remove(ev)

        # Generate daily orders
        adjusted_avg_daily = avg_daily * cum_multiplier[today.year]
        nords = np.random.poisson(adjusted_avg_daily)
        daily_orders = []
        for _ in range(nords):
            store_row = stores.sample(1).iloc[0]
            home_hub  = store_row["Home_Hub_ID"]
            num_lines = random.randint(1, 5)
            order_id  = f"ORD{next_ord:05d}"

            # Calculate Required_Delivery_Date
            required_delivery_date = today + timedelta(days=random.randint(3, 10))
            urgency_days = (required_delivery_date - today).days
            is_express = urgency_days <= CFG["EXPRESS_URGENCY_DAYS"]

            next_ord += 1
            order_lines = []
            for __ in range(num_lines):
                prod = prods.sample(1).iloc[0]
                base = random.randint(prod.Base_Demand_Min, prod.Base_Demand_Max)
                qtyf = (
                    base * mult_map[store_row.Store_ID]
                    * QUARTER_SEAS[(today.month-1)//3 + 1][prod.Category]
                    * MONTH_SEAS[today.month]
                    * WEEKDAY_FACT[today.weekday()]
                    * random.uniform(0.9, 1.1)
                )
                q = max(1, int(round(qtyf)))
                ol = {
                    "Order_Line_ID": f"OL{next_ol:05d}",
                    "Order_ID": order_id,
                    "Store_ID": store_row.Store_ID,
                    "SKU": prod.SKU,
                    "Source_Hub_ID": hub_map[prod.Category],
                    "Order_Date": today,
                    "Required_Delivery_Date": required_delivery_date,
                    "Quantity_Ordered": q,
                    "Is_Express": is_express
                }
                order_lines.append(ol)
                next_ol += 1
            daily_orders.append({
                "Order_ID": order_id,
                "Store_ID": store_row.Store_ID,
                "Home_Hub_ID": home_hub,
                "Lines": order_lines,
                "Required_Delivery_Date": required_delivery_date,
                "Is_Express": is_express
            })

        # Process orders
        for order in daily_orders:
            home_hub = order["Home_Hub_ID"]
            source_hubs = set(ol["Source_Hub_ID"] for ol in order["Lines"])
            is_direct = len(source_hubs) == 1 and home_hub in source_hubs

            crossdock_items: Dict[str, int] = {}
            inter_hub_created = False
            shipped_lines: List[Dict] = []

            for ol in order["Lines"]:
                src_hub = ol["Source_Hub_ID"]
                key = (src_hub, ol["SKU"])
                avail = inv_levels.get(key, 0)
                ship = min(avail, ol["Quantity_Ordered"])

                ol["Quantity_Shipped"] = ship
                orders.append(ol)

                if ship > 0:
                    inv_levels[key] -= ship
                    shipped_lines.append(ol)

                    if src_hub == home_hub:
                        crossdock_items[ol["SKU"]] = crossdock_items.get(ol["SKU"], 0) + ship
                    else:
                        inter_hub_created = True
                        truck_manager.add_shipment_item(
                            origin=src_hub,
                            destination=home_hub,
                            dispatch_date=today,
                            order_line_id=ol["Order_Line_ID"],
                            sku=ol["SKU"],
                            quantity=ship,
                            order_id=order["Order_ID"],
                            required_delivery_date=ol["Required_Delivery_Date"],
                            is_express=ol["Is_Express"]
                        )

                # Track backorders for unfulfilled quantities
                if ship < ol["Quantity_Ordered"]:
                    next_bo = track_backorder(ol, backorders, next_bo)

            # Direct final-mile
            if is_direct and crossdock_items:
                lid = f"SL{next_leg:06d}"
                next_leg += 1
                wt  = sum(qty * sku_to_wt[sku] for sku, qty in crossdock_items.items())
                vol = sum(qty * sku_to_vol[sku] for sku, qty in crossdock_items.items())

                dist = store_distances[order["Store_ID"]]
                transit_days = calculate_transit_days(dist, is_inter_hub=False)
                arrival_date = today + timedelta(days=transit_days)

                cost = courier_cost(dist, wt, vol)
                leg = {
                    "Shipment_Leg_ID": lid,
                    "Leg_Type": "Final-Mile",
                    "Transport_Mode": "Courier",
                    "Origin": home_hub,
                    "Destination": order["Store_ID"],
                    "Dispatch_Date": today,
                    "Arrival_Date": arrival_date,
                    "Transportation_Cost": cost
                }
                legs.append(leg)
                for ol in shipped_lines:
                    links.append({
                        "Shipment_Leg_ID": lid,
                        "Order_Line_ID": ol["Order_Line_ID"],
                        "Quantity_Shipped": ol["Quantity_Shipped"]
                    })
                next_rt = handle_returns(shipped_lines, arrival_date, sku_to_cat, next_rt, rets, inv_levels, pending_returns)

            # Mixed orders
            if crossdock_items and inter_hub_created:
                for sku, qty in crossdock_items.items():
                    pending_crossdock.setdefault(home_hub, []).append({
                        "Order_ID": order["Order_ID"],
                        "SKU": sku,
                        "Quantity": qty,
                        "Arrival_Date": today,
                        "From_Hub": home_hub,
                        "Required_Delivery_Date": order["Required_Delivery_Date"]
                    })

        # Process consolidated trucks
        next_leg = truck_manager.process_ready_trucks(
            today, next_leg, legs, links, pending_crossdock, delay_buffer
        )

        # Cross-dock consolidation with processing time - FIXED
        if pending_crossdock:
            for hub in list(pending_crossdock.keys()):
                order_ids = set(item["Order_ID"] for item in pending_crossdock[hub])
                for ord_id in list(order_ids):
                    all_for_order = [item for item in pending_crossdock[hub] if item["Order_ID"] == ord_id]
                    arrived_for_order = [item for item in all_for_order if item["Arrival_Date"] <= today]

                    if len(arrived_for_order) == len(all_for_order) and len(arrived_for_order) > 0:
                        # Add cross-dock processing time - FIXED DATE HANDLING
                        last_arrival = max(item["Arrival_Date"] for item in arrived_for_order)

                        # Calculate processing days (8 hours = 0.33 days, round up to 1 day)
                        processing_days = 1 if CFG["CROSSDOCK_PROCESSING_HOURS"] >= 4 else 0
                        earliest_dispatch = last_arrival + timedelta(days=processing_days)

                        # Only dispatch if processing is complete
                        if today >= earliest_dispatch:
                            order_items: Dict[str, int] = defaultdict(int)
                            required_delivery = None
                            for item in arrived_for_order:
                                order_items[item["SKU"]] += item["Quantity"]
                                if required_delivery is None:
                                    required_delivery = item.get("Required_Delivery_Date")

                            store_id = next(ol["Store_ID"] for ol in orders if ol["Order_ID"] == ord_id)
                            lid = f"SL{next_leg:06d}"
                            next_leg += 1
                            wt  = sum(qty * sku_to_wt[sku] for sku, qty in order_items.items())
                            vol = sum(qty * sku_to_vol[sku] for sku, qty in order_items.items())

                            dist = store_distances[store_id]
                            transit_days = calculate_transit_days(dist, is_inter_hub=False)
                            arrival_date = today + timedelta(days=transit_days)

                            cost = courier_cost(dist, wt, vol)
                            leg = {
                                "Shipment_Leg_ID": lid,
                                "Leg_Type": "Final-Mile",
                                "Transport_Mode": "Courier",
                                "Origin": hub,
                                "Destination": store_id,
                                "Dispatch_Date": today,
                                "Arrival_Date": arrival_date,
                                "Transportation_Cost": cost
                            }
                            legs.append(leg)
                            order_lines = [ol for ol in orders if ol["Order_ID"] == ord_id]
                            for ol in order_lines:
                                links.append({
                                    "Shipment_Leg_ID": lid,
                                    "Order_Line_ID": ol["Order_Line_ID"],
                                    "Quantity_Shipped": ol["Quantity_Shipped"]
                                })
                            next_rt = handle_returns(order_lines, arrival_date, sku_to_cat, next_rt, rets, inv_levels, pending_returns)
                            pending_crossdock[hub] = [item for item in pending_crossdock[hub] if item["Order_ID"] != ord_id]

                if hub in pending_crossdock and not pending_crossdock[hub]:
                    del pending_crossdock[hub]

        # Replenishment
        for hub in hubs.Hub_ID:
            lows = [sku for (h,sku), qty in inv_levels.items() if h == hub and qty < sku_to_rop[sku]]
            if not lows:
                continue
            if any(ev["Destination_Hub_ID"] == hub for ev in pending_inb):
                continue
            spec = CAT_TO_SUP_SPEC[sku_to_cat[lows[0]]]
            sup  = sup_full[sup_full.Specialty == spec].sample(1).iloc[0]
            cid  = f"INB{next_inb:05d}"; next_inb += 1
            for sku in lows:
                need = max(1, sku_to_tgt[sku] - inv_levels[(hub, sku)])
                lt   = sample_lead_time()
                ev = {
                    "Inbound_Shipment_ID":  cid,
                    "Supplier_ID":          sup.Supplier_ID,
                    "Destination_Hub_ID":   hub,
                    "SKU":                  sku,
                    "Quantity_Received":    need,
                    "Expected_Arrival_Date": today + timedelta(days=lt - random.randint(0, 5)),
                    "Actual_Arrival_Date":   today + timedelta(days=lt)
                }
                pending_inb.append(ev)

        # Daily inventory snapshot
        for (hub, sku), qty in inv_levels.items():
            snaps.append({
                "Snapshot_Date":    today,
                "Hub_ID":           hub,
                "SKU":              sku,
                "Quantity_On_Hand": qty
            })

    # Save outputs
    tables = [
        (hubs,   "dim_hubs.csv"),
        (stores, "dim_stores.csv"),
        (prods,  "dim_products.csv"),
        (sup_dim,"dim_suppliers.csv"),
        (pd.DataFrame(orders), "fact_orders.csv"),
        (pd.DataFrame(legs),   "fact_shipments.csv"),
        (pd.DataFrame(links),  "link_shipment_orders.csv"),
        (pd.DataFrame(inbs),   "fact_inbound_shipments.csv"),
        (pd.DataFrame(snaps),  "fact_inventory_snapshot.csv"),
        (pd.DataFrame(rets),   "fact_returns.csv"),
        (pd.DataFrame(backorders), "fact_backorders.csv"),
    ]
    for df, name in tables:
        df.to_csv(CFG["OUT_DIR"]/name, index=False)
        log.info("Saved %s (%d rows)", name, len(df))

    log.info("Generation complete.")

if __name__ == "__main__":
    main()
