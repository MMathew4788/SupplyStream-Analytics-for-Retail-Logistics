# 📦 SupplyStream: Synthetic Retail Supply Chain Simulator + Analytics Dashboard

## ✨ Overview

This repo have a Python-based simulator that generates realistic supply chain datasets tailored for an Indian retail business selling apparel, shoes, and accessories. Built on a hub-and-spoke model with cross-docking, this framework feeds directly into a Power BI dashboard for performance tracking across fulfillment, delivery, inventory, transport costs, and reverse logistics.

Designed for:

- 📊 Dashboard prototyping (Power BI, Tableau)
- 🧠 Machine learning model training
- 🎓 Educational use in operations, logistics, analytics

---

## 🛠️ Architecture Summary

```
Python Data generation → SupplyChain_Data/*.csv → GitHub Raw URLs → Power BI DAX Modeling → Data Visualization
```

---

## 🚚 Logistic Model

This synthetic framework mimics a multi-echelon, multi-SKU retail network:

- Hubs serve as central replenishment points, using continuous review inventory logic (s, Q) with safety stock anchored to service level.
- Stores draw stock via cross-docking or direct shipment, capturing variability from seasonality, weekday effects, and store size factors.
- Transport flows distinguish inter-hub bulk truck legs from final-mile courier legs, applying realistic costing and lead-time distributions.
- Returns logic inserts post-delivery reversals, making the dataset suitable for reverse-logistics analysis.

#### Shipment & Transportation Logic

The model simulates two transport tiers:

##### Inter-Hub Shipments

- Legs: Dispatch to hub cross-dock, transit 1–3 days by truck.
- Cost = ₹300 + ₹15/km + ₹8·chargeable_kg, with 1.2× multiplier for bulk inter-hub.
- Distances for DEL-BOM, DEL-BLR, BOM-BLR are hard-coded; others are randomized.

##### Final-Mile Shipments

- Direct store shipments: local pick if all lines from same hub or cross-docked consolidation.
- Delivery lead time = 1 day, distance 20–120 km by courier.
- Cost uses the same cost function without the inter-hub multiplier.

##### Returns are simulated with a 3–6% chance per line, random quantity, and category-based reason probabilities.

---

## 🏗️ Synthetic Data Generator

### 📂 Output Structure

| Type       | Files                                                                                                          |
| ---------- | -------------------------------------------------------------------------------------------------------------- | --- | ----- | ------------------------ |
| Dimensions | dim_hubs.csv, dim_stores.csv, dim_products.csv, dim_suppliers.csv                                              |
| Facts      | fact_orders.csv, fact_shipments.csv, fact_inventory_snapshot.csv, fact_returns.csv, fact_inbound_shipments.csv |     | Links | link_shipment_orders.csv |
| Logs       | SupplyChain_Data/data_gen.log                                                                                  |

### ⚙️ Simulation Parameters (CFG block)

- NUM_STORES, NUM_PRODUCTS, NUM_SUPPLIERS, NUM_ORDERS
- START_DATE, END_DATE
- Lead times (inter-hub, final-mile, inbound)
- Reorder points (ROP), return rates, supplier reliability

### 🧮 Modeling Assumptions

- Poisson-distributed demand with seasonal spikes
- Realistic logistics modeling (e.g., Delhi-Mumbai shipping distances)
- Supplier variability: 85–98% reliability
- Returns categorized by cause (“Wrong Size”, “Wrong Colour”)
- Courier costing: ₹300 base + ₹15/km distance

### 🚀 How to Run

- #### Install dependencies

```
pip install numpy pandas faker
```

- #### Clone and run

```
git clone https://github.com/MMathew4788/SupplyStream-Analytics-for-Retail-Logistics.git

cd SupplyStream-Analytics-for-Retail-Logistics

python generate_data.py

```

## 🌐 Hosting CSV Files via GitHub

Once generated, upload `SupplyChain_Data/*.csv` to your GitHub repository. Access the raw file links like:
https://raw.githubusercontent.com/your-username/repo-name/main/SupplyChain_Data/fact_orders.csv

---

## 📊 Power BI Dashboard – Analytics Layer

### 🔧 Data Import

- Open Power BI Desktop
- Use Get Data → Web
- Paste GitHub raw URL for each CSV
- Name your tables clearly (fact_orders, dim_products, etc.)
- Define relationships in Model View

### 🗂️ Manage Relationships

| Table Name                                 | Relationship Type | Related Table                        |
| ------------------------------------------ | ----------------- | ------------------------------------ |
| `fact_inbound_shipments` (`SKU`)           | \* → 1            | `dim_products` (`SKU`)               |
| `fact_inbound_shipments` (`Supplier_ID`)   | \* → 1            | `dim_suppliers` (`Supplier_ID`)      |
| `fact_inventory_snapshot` (`Hub_ID`)       | \* → 1            | `dim_hubs` (`Hub_ID`)                |
| `fact_inventory_snapshot` (`SKU`)          | \* → 1            | `dim_products` (`SKU`)               |
| `fact_orders` (`SKU`)                      | \* → 1            | `dim_products` (`SKU`)               |
| `fact_orders` (`Store_ID`)                 | \* → 1            | `dim_stores` (`Store_ID`)            |
| `fact_returns` (`Order_Line_ID`)           | 1 ↔ 1             | `fact_orders` (`Order_Line_ID`)      |
| `fact_shipments` (`Origin`)                | \* → 1            | `dim_hubs` (`Hub_ID`)                |
| `link_shipment_orders` (`Order_Line_ID`)   | \* → 1            | `fact_orders` (`Order_Line_ID`)      |
| `link_shipment_orders` (`Shipment_Leg_ID`) | \* → 1            | `fact_shipments` (`Shipment_Leg_ID`) |

---

### 📐 DAX Measures

#### 🚚 Order Fulfillment

```
Total Quantity Ordered = SUM('fact_orders'[Quantity_Ordered])
Total Quantity Shipped = SUM('fact_orders'[Quantity_Shipped])
Order Fulfillment Rate % = DIVIDE([Total Quantity Shipped], [Total Quantity Ordered], 0)
```

#### 🕒 Delivery Performance

```
Actual Delivery Date =
MAXX(
    FILTER(link_shipment_orders, RELATED(fact_shipments[Leg_Type]) = "Final-Mile"),
    RELATED(fact_shipments[Arrival_Date])
)

Number of On Time Deliveries =
COUNTROWS(
    FILTER(
        ADDCOLUMNS(fact_orders, "ActDate", [Actual Delivery Date]),
        [ActDate] <= fact_orders[Required_Delivery_Date]
    )
)

On Time Delivery % = DIVIDE([Number of On Time Deliveries], COUNTROWS(fact_orders), 0)
```

#### 🧮 Inventory Insights

```
Stockout Risk =
COUNTROWS(
    FILTER(
        fact_inventory_snapshot,
        fact_inventory_snapshot[Quantity_On_Hand] < RELATED(dim_products[ROP])
    )
)

Average Product On Hand =
AVERAGEX(
    SUMMARIZE(fact_inventory_snapshot, fact_inventory_snapshot[SKU]),
    CALCULATE(AVERAGE(fact_inventory_snapshot[Quantity_On_Hand]))
)
```

#### 🚛 Transport Economics

```
Total Transport Cost = SUM(fact_shipments[Transportation_Cost])
Cost Per Unit =
DIVIDE(
    [Total Transport Cost],
    SUM(link_shipment_orders[Quantity_Shipped]),
    0
)
```

#### 🔁 Returns & Reverse Logistics

```
Total Returns = COUNTROWS(fact_returns)

Return Rate % =
DIVIDE(
    SUM(fact_returns[Quantity_Returned]),
    SUM(fact_orders[Quantity_Shipped]),
    0
) * 100
```

### 📦 Project File Reference

The `Analysis.pbix` file is available in the root folder of this repository.

---

## 🙌 Contribute, Fork, and Share

If you find this project useful, feel free to **star ⭐**, **fork 🍴**, or adapt it for your own supply chain analytics workflows. Contributions, enhancements, and feedback are always welcome.

Let’s make retail logistics smarter, together 🚀
