# End-to-End Supply Chain Data Generator

This repository contains a **Python** script to generate a full synthetic dataset for a multi-hub apparel distribution network.  
It produces dimensions and facts mirroring real supply-chain processes, enabling you to prototype analytics, dashboards, or ML models without access to confidential operational data.

## Features

- **Reproducible**: Seeds for Python, NumPy, and Faker guarantee identical outputs.
- **Multi-Hub Network**: Three regional hubs (Delhi, Mumbai, Bangalore) with specialty categories.
- **Demand Modeling**: Seasonality (quarter, month, weekday) + store-level multipliers.
- **Partial Fulfillment & Auto-Reorder**:
  - Ship up to available stock, record shortfalls.
  - Trigger inbound replenishment for unmet orders.
- **Inventory Snapshots**: Daily on-hand levels driven by real inbounds & shipments.
- **Return Simulation**: Category-specific return reasons and volumes.
- **Configurable**: Central `CFG` dictionary for counts, date ranges, thresholds.

## Generated Files

| Filename                      | Description                               |
| ----------------------------- | ----------------------------------------- |
| `dim_hubs.csv`                | Hub master data                           |
| `dim_stores.csv`              | Store master data                         |
| `dim_products.csv`            | Product master data                       |
| `dim_suppliers.csv`           | Supplier master data                      |
| `fact_orders.csv`             | Customer order lines                      |
| `fact_shipments.csv`          | Aggregated shipment legs                  |
| `link_shipment_orders.csv`    | Shipment-to-order line link               |
| `fact_unfulfilled_orders.csv` | Partial fulfillment records               |
| `fact_inbound_shipments.csv`  | Inbound replenishment shipments           |
| `fact_inventory_snapshot.csv` | Daily hub–SKU on-hand inventory snapshots |
| `fact_returns.csv`            | Customer returns                          |
