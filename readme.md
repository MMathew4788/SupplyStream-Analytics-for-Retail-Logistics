## 📦 SupplyStream: Synthetic Retail Supply Chain Simulator + Analytics Dashboard

## ✨ Overview

This repo have a Python-based simulator that generates realistic supply chain datasets tailored for an Indian retail business selling apparel, shoes, and accessories. Built on a hub-and-spoke model with cross-docking, this framework feeds directly into a Power BI dashboard for performance tracking across fulfillment, delivery, inventory, transport costs, and reverse logistics.

Designed for:

- 📊 Dashboard prototyping (Power BI, Tableau)
- 🧠 Machine learning model training
- 🎓 Educational use in operations, logistics, analytics

---

## 🔍 View the live interactive preview of Dashboard

🔗[Live Preview](https://app.powerbi.com/view?r=eyJrIjoiZWM1ZjcwYjgtODVlMy00ZDYzLWJlYmUtNDBkYjBkZDZkN2JiIiwidCI6ImM2ZTU0OWIzLTVmNDUtNDAzMi1hYWU5LWQ0MjQ0ZGM1YjJjNCJ9)

---

## 🛠️ Architecture Summary

```
Python Data generation → SupplyChain_Data/*.csv → GitHub Raw URLs → Power BI DAX Modeling → Data Visualization
```

---

## 🚚 Logistic Model

#### Network Topology

- 20 specialized suppliers mapped to key manufacturing clusters
- 3 regional hubs:
  HUB-DEL (Apparel), HUB-BOM (Footwear), HUB-BLR (Accessories)
- ~100 stores (major/minor) assigned to a “home” hub

#### Demand Generation

- ~40 daily orders (Poisson process) over Jan 2022–Jun 2025
- Seasonality factors: quarter, month, weekday
- Multi-SKU orders (1–5 lines), 3–10 day delivery window
- Inventory & Replenishment
- Reorder Point (ROP) = lead-time demand + safety stock (95% service level)
- Lead times sampled 3–14 days (μ=7, σ=2)
- Single inbound batch per hub when any SKU falls below ROP

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

### 📐 Define Relationships & DAX Measures

- Use Model View to link tables
- Create DAX measures for KPIs

### 📦 Project File Reference

The `Analysis.pbix` file is available in the root folder of this repository.

---

## 📖 Learn More About This Project

🌐[View the project web page](https://www.manojmathew.com/project2.html)

## 🙌 Contribute, Fork, and Share

If you find this project useful, feel free to **star ⭐**, **fork 🍴**, or adapt it for your own supply chain analytics workflows. Contributions, enhancements, and feedback are always welcome.

Let’s make retail logistics smarter, together 🚀
