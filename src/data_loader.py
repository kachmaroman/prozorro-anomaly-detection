"""
Data loading utilities for ProZorro dataset.

Optimized for memory efficiency with 13M+ records.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union

from .config import (
    DATA_DIR, YEARS, TENDER_FILES, BID_FILES,
    BUYERS_FILE, SUPPLIERS_FILE, BIDDERS_FILE,
    ProcurementMethod
)


# === Column dtypes for memory optimization ===
TENDER_DTYPES = {
    "tender_id": "string",
    "ocid": "string",
    "buyer_id": "string",
    "procuring_entity_id": "string",
    "supplier_id": "string",
    "locality": "string",
    "postal_code": "string",
    "procurement_method": "category",
    "main_procurement_category": "category",
    "award_criteria": "category",
    "main_cpv_code": "string",
    "currency": "category",
    "year": "int16",
    "month": "int8",
    "quarter": "int8",
    "day_of_week": "int8",
    "is_q4": "int8",
    "is_december": "int8",
    "is_weekend": "int8",
    "is_single_bidder": "int8",
    "is_competitive": "int8",
    "is_cross_region": "int8",
    "has_enquiries": "int8",
    "is_buyer_masked": "int8",
    "is_supplier_masked": "int8",
    "has_multiple_awards": "int8",
    "has_unsuccessful_awards": "int8",
    "has_cancelled_awards": "int8",
    "number_of_items": "int32",
    "number_of_tenderers": "int16",
    "number_of_bids": "int16",
    "number_of_awards": "int16",
    "number_of_contracts": "int16",
    "number_of_documents": "int16",
    "active_awards_count": "int16",
    "main_cpv_2_digit": "float32",
    "main_cpv_4_digit": "float32",
    "tender_value": "float64",
    "award_value": "float64",
    "price_change_amount": "float64",
    "price_change_pct": "float64",
    "award_value_total": "float64",
    "award_value_max": "float64",
    "award_value_min": "float64",
    "award_value_mean": "float64",
    "award_value_std": "float64",
    "award_concentration": "float32",
    "discount_percentage_avg": "float32",
    "discount_percentage_max": "float32",
}

BID_DTYPES = {
    "tender_id": "string",
    "bid_id": "string",
    "bidder_id": "string",
    "bid_status": "category",
    "bid_amount": "float64",
    "is_winner": "int8",
    "is_bidder_masked": "int8",
}

DATE_COLUMNS = ["published_date", "award_date"]


def load_tenders(
    years: Optional[Union[int, list]] = None,
    procurement_method: Optional[str] = None,
    columns: Optional[list] = None,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load tender data with optional filtering.

    Args:
        years: Year(s) to load. None = all years.
        procurement_method: Filter by method ('limited', 'open', 'selective').
        columns: Columns to load. None = all columns.
        sample_frac: Random sample fraction (0-1). None = full data.
        random_state: Random seed for sampling.

    Returns:
        DataFrame with tender data.

    Examples:
        >>> # Load all 2023 tenders
        >>> df = load_tenders(years=2023)

        >>> # Load only Open tenders from 2023-2024
        >>> df = load_tenders(years=[2023, 2024], procurement_method='open')

        >>> # Load 10% sample of all data
        >>> df = load_tenders(sample_frac=0.1)
    """
    if years is None:
        years = YEARS
    elif isinstance(years, int):
        years = [years]

    dfs = []
    for year in years:
        file_path = TENDER_FILES.get(year)
        if file_path is None or not file_path.exists():
            print(f"Warning: File for {year} not found, skipping")
            continue

        # Read with optimized dtypes
        usecols = columns if columns else None
        df = pd.read_csv(
            file_path,
            usecols=usecols,
            dtype={k: v for k, v in TENDER_DTYPES.items() if usecols is None or k in usecols},
            low_memory=False,
        )

        # Parse dates
        for col in DATE_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

        # Filter by procurement method
        if procurement_method and "procurement_method" in df.columns:
            df = df[df["procurement_method"] == procurement_method]

        print(f"Loaded {year}: {len(df):,} records")
        dfs.append(df)

    if not dfs:
        raise ValueError("No data loaded. Check years and file paths.")

    result = pd.concat(dfs, ignore_index=True)

    # Sample if requested
    if sample_frac is not None:
        result = result.sample(frac=sample_frac, random_state=random_state)
        print(f"Sampled to {len(result):,} records ({sample_frac*100:.0f}%)")

    return result


def load_bids(
    years: Optional[Union[int, list]] = None,
    columns: Optional[list] = None,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load bid data with optional filtering.

    Args:
        years: Year(s) to load. None = all years.
        columns: Columns to load. None = all columns.
        sample_frac: Random sample fraction (0-1).
        random_state: Random seed for sampling.

    Returns:
        DataFrame with bid data.
    """
    if years is None:
        years = YEARS
    elif isinstance(years, int):
        years = [years]

    dfs = []
    for year in years:
        file_path = BID_FILES.get(year)
        if file_path is None or not file_path.exists():
            print(f"Warning: Bids file for {year} not found, skipping")
            continue

        usecols = columns if columns else None
        df = pd.read_csv(
            file_path,
            usecols=usecols,
            dtype={k: v for k, v in BID_DTYPES.items() if usecols is None or k in usecols},
            low_memory=False,
        )

        # Parse bid_date
        if "bid_date" in df.columns:
            df["bid_date"] = pd.to_datetime(df["bid_date"], errors="coerce", utc=True)

        print(f"Loaded bids {year}: {len(df):,} records")
        dfs.append(df)

    if not dfs:
        raise ValueError("No bid data loaded.")

    result = pd.concat(dfs, ignore_index=True)

    if sample_frac is not None:
        result = result.sample(frac=sample_frac, random_state=random_state)

    return result


def load_buyers() -> pd.DataFrame:
    """Load buyers reference table."""
    df = pd.read_csv(BUYERS_FILE, dtype={"buyer_id": "string"})
    print(f"Loaded buyers: {len(df):,}")
    return df


def load_suppliers() -> pd.DataFrame:
    """Load suppliers reference table."""
    df = pd.read_csv(SUPPLIERS_FILE, dtype={"supplier_id": "string"})
    print(f"Loaded suppliers: {len(df):,}")
    return df


def load_bidders() -> pd.DataFrame:
    """Load bidders reference table."""
    df = pd.read_csv(BIDDERS_FILE, dtype={"bidder_id": "string"})
    print(f"Loaded bidders: {len(df):,}")
    return df


def load_open_tenders(years: Optional[Union[int, list]] = None, **kwargs) -> pd.DataFrame:
    """Shortcut: Load only Open tenders (5.5% of data)."""
    return load_tenders(years=years, procurement_method=ProcurementMethod.OPEN, **kwargs)


def load_limited_tenders(years: Optional[Union[int, list]] = None, **kwargs) -> pd.DataFrame:
    """Shortcut: Load only Limited tenders (91% of data)."""
    return load_tenders(years=years, procurement_method=ProcurementMethod.LIMITED, **kwargs)


def load_selective_tenders(years: Optional[Union[int, list]] = None, **kwargs) -> pd.DataFrame:
    """Shortcut: Load only Selective tenders (3.3% of data)."""
    return load_tenders(years=years, procurement_method=ProcurementMethod.SELECTIVE, **kwargs)


def merge_with_buyers(tenders: pd.DataFrame, buyers: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Merge tenders with buyer information.

    Adds: buyer_name, buyer_region, single_bidder_rate, competitive_rate,
          supplier_diversity_index, etc.
    """
    if buyers is None:
        buyers = load_buyers()

    return tenders.merge(
        buyers,
        on="buyer_id",
        how="left",
        suffixes=("", "_buyer"),
    )


def merge_with_suppliers(tenders: pd.DataFrame, suppliers: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Merge tenders with supplier information.

    Adds: supplier_name, supplier_region, total_awards, total_value.
    """
    if suppliers is None:
        suppliers = load_suppliers()

    return tenders.merge(
        suppliers,
        on="supplier_id",
        how="left",
        suffixes=("", "_supplier"),
    )


def get_tenders_with_bids(
    years: Optional[Union[int, list]] = None,
    min_bids: int = 2,
) -> pd.DataFrame:
    """
    Load tenders that have associated bids data.

    Useful for statistical screens that require bid-level analysis.

    Args:
        years: Year(s) to load.
        min_bids: Minimum number of bids per tender.

    Returns:
        DataFrame with tenders that have >= min_bids.
    """
    tenders = load_tenders(years=years)
    bids = load_bids(years=years)

    # Count bids per tender
    bid_counts = bids.groupby("tender_id").size().reset_index(name="actual_bid_count")

    # Merge and filter
    result = tenders.merge(bid_counts, on="tender_id", how="inner")
    result = result[result["actual_bid_count"] >= min_bids]

    print(f"Tenders with >= {min_bids} bids: {len(result):,}")
    return result


def memory_usage(df: pd.DataFrame) -> str:
    """Get human-readable memory usage of DataFrame."""
    mem = df.memory_usage(deep=True).sum()
    if mem < 1024**2:
        return f"{mem / 1024:.1f} KB"
    elif mem < 1024**3:
        return f"{mem / 1024**2:.1f} MB"
    else:
        return f"{mem / 1024**3:.2f} GB"


# === Quick data overview ===
def data_overview() -> dict:
    """
    Get quick overview of available data without loading everything.

    Returns:
        Dictionary with file sizes and estimated record counts.
    """
    overview = {}

    # Tender files
    for year, path in TENDER_FILES.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024**2)
            overview[f"tenders_{year}"] = f"{size_mb:.0f} MB"

    # Bid files
    for year, path in BID_FILES.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024**2)
            overview[f"bids_{year}"] = f"{size_mb:.0f} MB"

    # Reference tables
    for name, path in [("buyers", BUYERS_FILE), ("suppliers", SUPPLIERS_FILE), ("bidders", BIDDERS_FILE)]:
        if path.exists():
            size_mb = path.stat().st_size / (1024**2)
            overview[name] = f"{size_mb:.1f} MB"

    return overview
