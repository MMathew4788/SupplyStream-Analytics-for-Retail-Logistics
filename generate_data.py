#!/usr/bin/env python3
"""
V15.3 – Realistic Distance & Courier Cost Integration
"""

import logging, random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

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
    "NUM_ORDERS":    6000,
    "START_DATE":    datetime(2022,1,1),
    "END_DATE":      datetime(2025,6,30),
    "OUT_DIR":       Path("SupplyChain_Data"),
    "SEED":          42,
    "SERVICE_FACTOR":1.65,
    "LT_MEAN":       7,
    "LT_STD":        2,
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
    1:{"Dress":.3,"Shoes":.4,"Accessories":.3},
    2:{"Dress":.4,"Shoes":.3,"Accessories":.3},
    3:{"Dress":.2,"Shoes":.3,"Accessories":.5},
    4:{"Dress":.5,"Shoes":.2,"Accessories":.3},
}
MONTH_SEAS    = {1:0.9,2:0.9,3:1,4:1.1,5:1,6:.9,7:1.2,8:1.1,9:1,10:1.8,11:1.5,12:1.3}
WEEKDAY_FACT = {i:(1.3 if i==6 else 1.2 if i==5 else 1.1 if i==4 else 1) for i in range(7)}

SUP_SPEC       = {"Cotton Apparel":"Tiruppur","Leather Goods":"Kanpur","Jewellery":"Jaipur"}
CAT_TO_SUP_SPEC= {"Dress":"Cotton Apparel","Shoes":"Leather Goods","Accessories":"Jewellery"}

RETURN_PROBS = {
    "Dress":       {"Wrong Size":.4,"Wrong Colour":.2,"Customer Did Not Like":.3,"Defective Product":.1},
    "Shoes":       {"Wrong Size":.3,"Wrong Colour":.1,"Customer Did Not Like":.4,"Defective Product":.2},
    "Accessories":{"Customer Did Not Like":.6,"Defective Product":.4}
}

def sample_lead_time()->int:
    lt = int(np.random.normal(CFG["LT_MEAN"], CFG["LT_STD"]))
    return max(3, min(14, lt))

def courier_cost(dist_km:int, wt_kg:float, vol_cbm:float)->float:
    """
    base ₹200 + ₹10/km + ₹5 per chargeable kg
    chargeable_kg = max(actual_kg, volumetric_kg)
    volumetric_kg = vol_cbm * 200
    """
    vol_kg = vol_cbm * 200
    chg_kg = max(wt_kg, vol_kg)
    return round(200 + 10*dist_km + 5*chg_kg, 2)

# ----------------------------------------------------------------------------
# 3. DIMENSIONS
# ----------------------------------------------------------------------------
def gen_hubs()->pd.DataFrame:
    df = pd.DataFrame([
        {"Hub_ID":"HUB-DEL","Hub_Name":"Delhi Apparel Hub","Specialty":"Dress"},
        {"Hub_ID":"HUB-BOM","Hub_Name":"Mumbai Footwear Hub","Specialty":"Shoes"},
        {"Hub_ID":"HUB-BLR","Hub_Name":"Bangalore Acc. Hub","Specialty":"Accessories"},
    ])
    log.info("Hubs: %d", len(df))
    return df

def gen_stores()->Tuple[pd.DataFrame,Dict[str,float]]:
    """
    Adds Store_Type=major/minor so we can pick distances realistically.
    """
    placements = {
        "HUB-DEL": (["Delhi","Gurgaon","Noida","Jaipur"],["Agra","Lucknow","Kanpur","Patna","Mathura"]),
        "HUB-BOM": (["Pune","Surat","Ahmedabad","Thane"],["Nagpur","Bhopal","Raipur","Indore","Rajkot"]),
        "HUB-BLR": (["Bengaluru","Chennai","Hyderabad"],["Coimbatore","Kochi","Tirupati"]),
    }
    rows, mult = [], {}
    sid=1
    for hub,(maj, minr) in placements.items():
        for city in maj:
            for _ in range(3):
                s=f"ST{sid:03d}"
                rows.append({"Store_ID":s,"Home_Hub_ID":hub,"City":city,"Store_Type":"major"})
                mult[s]=random.uniform(1.5,2.5); sid+=1
        for city in minr:
            for _ in range(2):
                s=f"ST{sid:03d}"
                rows.append({"Store_ID":s,"Home_Hub_ID":hub,"City":city,"Store_Type":"minor"})
                mult[s]=random.uniform(0.8,1.2); sid+=1

    df = (pd.DataFrame(rows)
          .sample(frac=1, random_state=CFG["SEED"])
          .head(CFG["NUM_STORES"])
          .reset_index(drop=True))
    log.info("Stores: %d", len(df))
    return df, mult

def gen_products()->pd.DataFrame:
    specs = {
        "Dress":        (["Cotton Kurta","Silk Anarkali","Linen Shirt Dress","Georgette Saree"],
                         ["Aanya","Riya","Zoya","Elara"], (0.3,1.2), (0.002,0.008), (5,15)),
        "Shoes":        (["Leather Loafers","Canvas Sneakers","Ethnic Juttis","Block Heels"],
                         ["Vector","Orion","Nova","Apex"], (0.5,1.5), (0.005,0.015), (3,10)),
        "Accessories":  (["Leather Handbag","Silver Jhumkas","Analog Watch","Canvas Belt"],
                         ["Aura","Celeste","Eon","Luna"], (0.1,1.0), (0.001,0.020), (8,25)),
    }
    rows=[]
    for i in range(1, CFG["NUM_PRODUCTS"]+1):
        cat,(subs,names,w_rng,v_rng,d_rng)=random.choice(list(specs.items()))
        sub=random.choice(subs)
        name=f"{random.choice(names)} {sub}"
        w=round(random.uniform(*w_rng),2)
        v=round(random.uniform(*v_rng),4)
        rows.append({
            "SKU":f"SKU{i:04d}",
            "Product_Name":name,
            "Category":cat,
            "Sub_Category":sub,
            "Weight_kg":w,
            "Volume_cbm":v,
            "Base_Demand_Min":d_rng[0],
            "Base_Demand_Max":d_rng[1],
        })
    df=pd.DataFrame(rows)
    # Pilot demand at 80% of base → faster depletion
    df["Avg_Daily_Demand"] = ((df.Base_Demand_Min+df.Base_Demand_Max)/2)*0.80
    # ROP = μ*LT + z*μ*σ
    df["ROP"] = (
        df.Avg_Daily_Demand*CFG["LT_MEAN"]
        + CFG["SERVICE_FACTOR"]*df.Avg_Daily_Demand*CFG["LT_STD"]
    ).round().astype(int)
    # Target = ROP + 7-day demand
    df["Target_Level"] = (df.ROP + 7*df.Avg_Daily_Demand).round().astype(int)
    log.info("Products: %d", len(df))
    return df

def gen_suppliers()->Tuple[pd.DataFrame,pd.DataFrame]:
    rows=[]
    for i in range(1, CFG["NUM_SUPPLIERS"]+1):
        spec=random.choice(list(SUP_SPEC.keys()))
        rows.append({
            "Supplier_ID":       f"SUP{i:03d}",
            "City":              SUP_SPEC[spec],
            "Specialty":         spec,
            "Reliability_Score": round(random.uniform(0.85,0.98),3)
        })
    full=pd.DataFrame(rows)
    slim=full.drop(columns=["Reliability_Score"])
    log.info("Suppliers: %d", len(slim))
    return slim, full

# ----------------------------------------------------------------------------
# 4. INITIAL INVENTORY
# ----------------------------------------------------------------------------
def seed_inventory(
    hubs:pd.DataFrame, prods:pd.DataFrame
)->Dict[Tuple[str,str],int]:
    m = hubs.set_index("Specialty")["Hub_ID"].to_dict()
    inv={}
    for _,r in prods.iterrows():
        inv[(m[r.Category], r.SKU)] = random.randint(100,400)
    log.info("Seeded inventory: %d hub-SKU pairs", len(inv))
    return inv

# ----------------------------------------------------------------------------
# 5. DAILY PIPELINE
# ----------------------------------------------------------------------------
def main():
    hubs      = gen_hubs()
    stores, mult_map = gen_stores()
    prods     = gen_products()
    sup_dim, sup_full= gen_suppliers()

    # Lookups
    hub_map    = hubs.set_index("Specialty")["Hub_ID"].to_dict()
    sku_to_cat = prods.set_index("SKU")["Category"].to_dict()
    sku_to_rop = prods.set_index("SKU")["ROP"].to_dict()
    sku_to_tgt = prods.set_index("SKU")["Target_Level"].to_dict()

    inv_levels  = seed_inventory(hubs, prods)
    pending_inb = []
    next_ol, next_leg, next_inb, next_rt = 1, 1, 1, 1

    orders, legs, links, inbs, rets, snaps = ([] for _ in range(6))
    total_days = (CFG["END_DATE"] - CFG["START_DATE"]).days + 1
    avg_daily  = CFG["NUM_ORDERS"] / total_days
    calendar   = pd.date_range(CFG["START_DATE"], CFG["END_DATE"], freq="D")

    for ts in calendar:
        today = ts.date()

        # 1) Inbound arrivals
        for ev in pending_inb[:]:
            if ev["Actual_Arrival_Date"] == today:
                inv_levels[(ev["Destination_Hub_ID"], ev["SKU"])] += ev["Quantity_Received"]
                inbs.append({**ev, "Actual_Arrival_Date":today})
                pending_inb.remove(ev)

        # 2) Poisson daily orders
        nords = np.random.poisson(avg_daily)
        day_lines = []
        for _ in range(nords):
            ol = {"Order_Line_ID": f"OL{next_ol:05d}"}; next_ol+=1
            store = stores.sample(1).iloc[0]
            prod  = prods.sample(1).iloc[0]
            base  = random.randint(prod.Base_Demand_Min, prod.Base_Demand_Max)
            qtyf  = (base * mult_map[store.Store_ID]
                     * QUARTER_SEAS[(today.month-1)//3+1][prod.Category]
                     * MONTH_SEAS[today.month]
                     * WEEKDAY_FACT[today.weekday()]
                     * random.uniform(0.9,1.1))
            q = max(1,int(round(qtyf)))
            ol.update({
                "Store_ID":store.Store_ID,
                "SKU":prod.SKU,
                "Source_Hub_ID":hub_map[prod.Category],
                "Order_Date":today,
                "Required_Delivery_Date":today + timedelta(days=random.randint(3,10)),
                "Quantity_Ordered":q
            })
            key   = (ol["Source_Hub_ID"], ol["SKU"])
            avail = inv_levels.get(key,0)
            ship  = min(avail, q)
            inv_levels[key] = avail - ship
            if ship>0:
                orders.append(ol)
                day_lines.append({**ol, "Quantity_Shipped": ship})

        # 3) Consolidated final-mile
        df_lines = pd.DataFrame(day_lines)
        if not df_lines.empty:
            grp = df_lines.groupby(["Order_Date","Source_Hub_ID","Store_ID"])
            for (d,src,dst), g in grp:
                lid = f"SL{next_leg:06d}"; next_leg+=1
                wt  = (g.Quantity_Shipped *
                       g.SKU.map(prods.set_index("SKU")["Weight_kg"])).sum()
                vol = (g.Quantity_Shipped *
                       g.SKU.map(prods.set_index("SKU")["Volume_cbm"])).sum()
                # realistic distance by store type
                st_type = stores.set_index("Store_ID").loc[dst,"Store_Type"]
                dist = random.randint(20,50) if st_type=="major" else random.randint(50,120)
                cost = courier_cost(dist, wt, vol)
                leg = {
                    "Shipment_Leg_ID":     lid,
                    "Leg_Type":            "Final-Mile",
                    "Transport_Mode":      "Courier",
                    "Origin":              src,
                    "Destination":         dst,
                    "Dispatch_Date":       d,
                    "Arrival_Date":        d + timedelta(days=1),
                    "Transportation_Cost": cost
                }
                legs.append(leg)
                for _,r in g.iterrows():
                    links.append({
                        "Shipment_Leg_ID":   lid,
                        "Order_Line_ID":     r.Order_Line_ID,
                        "Quantity_Shipped":  r.Quantity_Shipped
                    })
                    # returns ~7%
                    if random.random()<0.07:
                        ret_q = random.randint(1, r.Quantity_Shipped)
                        cat   = sku_to_cat[r.SKU]
                        reason= random.choices(list(RETURN_PROBS[cat].keys()),
                                              weights=list(RETURN_PROBS[cat].values()))[0]
                        rets.append({
                            "Return_ID":         f"RTN{next_rt:05d}",
                            "Order_Line_ID":     r.Order_Line_ID,
                            "SKU":               r.SKU,
                            "Quantity_Returned": ret_q,
                            "Return_Date":       leg["Arrival_Date"]+timedelta(days=random.randint(1,21)),
                            "Return_Reason":     reason
                        })
                        next_rt+=1

        # 4) Reorder per hub in one container
        for hub in hubs.Hub_ID:
            lows = [sku for (h,sku),qty in inv_levels.items() if h==hub and qty<sku_to_rop[sku]]
            if not lows: continue
            if any(ev["Destination_Hub_ID"]==hub for ev in pending_inb): continue
            spec = CAT_TO_SUP_SPEC[sku_to_cat[lows[0]]]
            sup  = sup_full[sup_full.Specialty==spec].sample(1).iloc[0]
            cid  = f"INB{next_inb:05d}"; next_inb+=1
            for sku in lows:
                need = max(1, sku_to_tgt[sku]-inv_levels[(hub,sku)])
                lt   = sample_lead_time()
                ev = {
                    "Inbound_Shipment_ID":  cid,
                    "Supplier_ID":          sup.Supplier_ID,
                    "Destination_Hub_ID":   hub,
                    "SKU":                  sku,
                    "Quantity_Received":    need,
                    "Promised_Arrival_Date":today,
                    "Actual_Arrival_Date":  today + timedelta(days=lt)
                }
                pending_inb.append(ev)

        # 5) Daily snapshot
        for (hub,sku),qty in inv_levels.items():
            snaps.append({
                "Snapshot_Date":    today,
                "Hub_ID":           hub,
                "SKU":              sku,
                "Quantity_On_Hand": qty
            })

    # 6) Save
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
    ]
    for df,name in tables:
        df.to_csv(CFG["OUT_DIR"]/name, index=False)
        log.info("Saved %s (%d rows)", name, len(df))

    log.info("Generation complete.")

if __name__=="__main__":
    main()
