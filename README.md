# Anomaly Detection in Ukrainian Public Procurement

Master's thesis project analyzing 13M+ ProZorro tenders for suspicious patterns and potential fraud.

## Structure

| Folder | Description |
|--------|-------------|
| `thesis/` | Academic documents (thesis PDF/DOCX) |
| `notebooks/` | Jupyter notebooks for EDA and modeling |
| `src/` | Python source code for models |
| `results/` | Figures, metrics, experiment results |
| `data/` | Dataset reference |

## Dataset

**Source:** ProZorro Ukrainian Public Procurement System
**Period:** 2022-2025
**Size:** 13.1M completed tenders, ~5.5 GB

## Quick Start

```python
import pandas as pd

# Load data
tenders = pd.read_csv('data/tenders_2023.csv', low_memory=False)
buyers = pd.read_csv('data/buyers.csv')
suppliers = pd.read_csv('data/suppliers.csv')
```

## Research Focus

1. **Single-bidder analysis** - 91% of tenders have no competition
2. **Price anomalies** - Unusual discounts/markups
3. **Buyer-supplier networks** - Concentration and relationships
4. **Temporal patterns** - War impact on procurement

## Requirements

```bash
pip install -r requirements.txt
```

## Related

- [prozorro-parser](../prozorro-parser/) - Dataset parser and documentation
