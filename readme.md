# SupplyStream: Synthetic Supply Chain Data Generator

## Overview

This code generates synthetic supply chain datasets for retail logistics, focused on an Indian apparel, shoes, and accessories business. It simulates a hub-and-spoke model with cross-docking, inventory management, orders, shipments, returns, and inbound logistics. The data is realistic, incorporating seasonality, stochastic demand (Poisson distribution), lead times, and supplier reliability.

This script is ideal for:

- Testing analytics dashboards (e.g., Power BI, Tableau).
- Training ML models on supply chain data.
- Educational purposes in logistics and operations research.

Key features:

- **Hub-and-Spoke with Cross-Docking**: Hubs act as warehouses and cross-docks for multi-category orders.
- **Stochastic Elements**: Demand variability, returns (3-6%), lead times (3-14 days), and supplier reliability (85-98% on-time delivery).
- **Output**: CSV files for dimensions (hubs, stores, products, suppliers) and facts (orders, shipments, inventory snapshots, returns, inbound shipments).

## Installation

1. **Prerequisites**:

   - Python 3.8+
   - Required libraries: `numpy`, `pandas`, `faker`

   Install via pip:
   pip install numpy pandas faker

2. **Clone the Repository**:
   git clone https://github.com/MMathew4788/SupplyStream-Analytics-for-Retail-Logistics.git
   cd SupplyStream-Analytics-for-Retail-Logistics

## Usage

1. **Run the Script**:
   python generate_data.py

- This generates data in the `SupplyChain_Data/` directory.
- Logs are saved to `SupplyChain_Data/data_gen.log`.

2. **Configuration** (Edit `CFG` in the script):

- `NUM_STORES`: Number of stores (default: 100).
- `NUM_PRODUCTS`: Number of SKUs (default: 500).
- `NUM_SUPPLIERS`: Number of suppliers (default: 20).
- `NUM_ORDERS`: Total orders over the period (default: 50,000 for realism).
- `START_DATE` / `END_DATE`: Simulation period (default: 2022-01-01 to 2025-06-30).
- Other params: Lead times, service factors, etc.

3. **Output Files**:

- **Dimensions**: `dim_hubs.csv`, `dim_stores.csv`, `dim_products.csv`, `dim_suppliers.csv`.
- **Facts**: `fact_orders.csv` (order lines), `fact_shipments.csv` (shipment legs), `link_shipment_orders.csv` (links), `fact_inbound_shipments.csv` (supplier deliveries), `fact_inventory_snapshot.csv` (daily stock), `fact_returns.csv` (returns).

## How It Works

- **Simulation Loop**: Runs daily from start to end date.
- Generates multi-line orders with seasonal demand.
- Handles cross-docking: Inter-hub shipments for non-local items, consolidation at home hub, then final-mile delivery.
- Inventory: Reorders when below ROP, with lead times and reliability-based delays.
- Returns: 3-6% rate, limited to "Wrong Size" or "Wrong Colour".
- **Realism**: Based on Indian geography (e.g., distances Delhi-Mumbai: 1400km), courier costs (₹300 base + ₹15/km), and retail trends (e.g., festival spikes in October).

## Contributing

1. Fork the repo.
2. Create a feature branch (`git checkout -b feature/new-feature`).
3. Commit changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a Pull Request.

For issues or suggestions, open a GitHub issue!
