# Prozorro Ukraine Public Procurement Dataset (2022-2025)

## Overview

**13.1+ million completed public procurement tenders** from Ukraine's Prozorro transparency system, covering **January 1, 2022 through December 31, 2025**.

**Dataset Highlights:**
- 13.1M+ tenders across 4 years
- 359K suppliers and 36K buyers
- 2.6M bids from competitive procedures
- 72K unique bidders (with EDRPOU - joinable with suppliers)
- Pre-war baseline (Jan-Feb 2022) for comparative analysis
- Year-based files for memory-efficient analysis

---

## What is Prozorro?

**Prozorro** is Ukraine's electronic public procurement system launched in 2016. It is one of the world's most transparent government procurement platforms:

- 100% open data via API
- $40+ billion processed annually (pre-war)
- Used by 60,000+ government institutions
- Based on OCDS (Open Contracting Data Standard)

---

## File Structure

```
prozorro_dataset_clean/
├── tenders_2022.csv    # 2.4M tenders
├── tenders_2023.csv    # 3.6M tenders
├── tenders_2024.csv    # 3.4M tenders
├── tenders_2025.csv    # 3.7M tenders
├── bids_2022.csv       # 242K bids
├── bids_2023.csv       # 489K bids
├── bids_2024.csv       # 847K bids
├── bids_2025.csv       # 1.1M bids
├── buyers.csv          # 36K buyers (16 columns)
├── suppliers.csv       # 359K suppliers (6 columns)
└── bidders.csv         # 72K bidders (7 columns)
```

---

## Key Features

### Tenders Table (55 columns)

**Identification**
- `tender_id`, `ocid`

**Temporal Features**
- Raw dates: `published_date`, `tender_start_date`, `tender_end_date`, `award_date`
- Normalized: `year`, `month`, `quarter`, `day_of_week`, `is_q4`, `is_december`, `is_weekend`

**Parties**
- `buyer_id` → buyers.csv
- `supplier_id` → suppliers.csv

**Procurement Details**
- `procurement_method`, `main_cpv_code`, `tender_value`, `award_value`, `currency`

**Price Analysis**
- `price_change_amount`, `price_change_pct`

**Competition**
- `number_of_tenderers`, `number_of_bids`, `is_competitive`

**Awards Aggregated (12 cols)**
- `award_value_total`, `award_value_max`, `award_value_min`, `award_value_mean`, `award_value_std`
- `award_concentration`, `discount_percentage_avg`, `discount_percentage_max`
- `has_multiple_awards`, `has_unsuccessful_awards`, `has_cancelled_awards`, `active_awards_count`

**Flags**
- `is_cross_region`, `is_single_bidder`, `is_buyer_masked`, `is_supplier_masked`

### Buyers Table (16 columns)

**Base Stats**
- `buyer_id`, `buyer_name`, `buyer_region`, `total_tenders`, `total_value`

**Market Diversity**
- `unique_suppliers`, `supplier_diversity_index`

**Competition Metrics**
- `avg_tenderers_per_tender`, `competitive_tenders_count`, `competitive_rate`, `single_bidder_rate`

**Efficiency Metrics**
- `avg_discount_pct`, `total_savings`, `avg_tender_value`

**Temporal**
- `first_tender_date`, `last_tender_date`

### Bids Table (8 columns)
- `tender_id`, `bid_id`, `bidder_id` (EDRPOU), `bid_date`, `bid_status`
- `bid_amount` (bid value in UAH), `is_winner` (1=won), `is_bidder_masked`

### Suppliers Table (6 columns)
- `supplier_id`, `supplier_name`, `supplier_region`, `total_awards`, `total_value`, `is_masked`

### Bidders Table (7 columns)
- `bidder_id` (EDRPOU — can join with suppliers), `total_bids`, `unique_tenders`, `unique_buyers`
- `first_bid_date`, `last_bid_date`, `is_masked`

---

## Quick Start

### Load Single Year

```python
import pandas as pd

tenders_2023 = pd.read_csv('tenders_2023.csv')
print(f"Loaded {len(tenders_2023):,} tenders")
```

### Load All Years

```python
years = [2022, 2023, 2024, 2025]
tenders = pd.concat([
    pd.read_csv(f'tenders_{year}.csv')
    for year in years
], ignore_index=True)
```

### Join Tables

```python
buyers = pd.read_csv('buyers.csv')
suppliers = pd.read_csv('suppliers.csv')

tenders_enriched = tenders.merge(buyers, on='buyer_id', how='left')
tenders_enriched = tenders_enriched.merge(suppliers, on='supplier_id', how='left')
```

---

## Research Ideas

1. **War Impact Analysis** - Compare pre-war (Jan-Feb 2022) vs wartime procurement
2. **Competition Analysis** - Single bidder patterns, market concentration
3. **Regional Analysis** - Procurement by region over time
4. **Fraud Detection** - Anomalous patterns, price manipulation

---

## Statistics

| Year | Tenders |
|------|---------|
| 2022 | 2,401,937 |
| 2023 | 3,599,565 |
| 2024 | 3,437,381 |
| 2025 | 3,657,528 |
| **Total** | **13,096,411** |

---

## Links

- **Data Source**: https://prozorro.gov.ua
- **Standard**: OCDS (Open Contracting Data Standard)
- **License**: CC0 1.0 (Public Domain)

---

**See also:** DATA_DICTIONARY.md for complete column descriptions.
