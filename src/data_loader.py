"""
Data loading utilities for ProZorro dataset.

Optimized for performance with Polars (13M+ records).
"""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List

from .config import (
    DATA_DIR, YEARS, TENDER_FILES, BID_FILES,
    BUYERS_FILE, SUPPLIERS_FILE, BIDDERS_FILE,
    ProcurementMethod
)


# === Schema definitions for Polars ===
TENDER_SCHEMA = {
    "tender_id": pl.Utf8,
    "ocid": pl.Utf8,
    "buyer_id": pl.Utf8,
    "procuring_entity_id": pl.Utf8,
    "supplier_id": pl.Utf8,
    "locality": pl.Utf8,
    "postal_code": pl.Utf8,
    "procurement_method": pl.Categorical,
    "main_procurement_category": pl.Categorical,
    "award_criteria": pl.Categorical,
    "main_cpv_code": pl.Utf8,
    "currency": pl.Categorical,
    "year": pl.Int16,
    "month": pl.Int8,
    "quarter": pl.Int8,
    "day_of_week": pl.Int8,
    "is_q4": pl.Int8,
    "is_december": pl.Int8,
    "is_weekend": pl.Int8,
    "is_single_bidder": pl.Int8,
    "is_competitive": pl.Int8,
    "is_cross_region": pl.Int8,
    "has_enquiries": pl.Int8,
    "is_buyer_masked": pl.Int8,
    "is_supplier_masked": pl.Int8,
    "has_multiple_awards": pl.Int8,
    "has_unsuccessful_awards": pl.Int8,
    "has_cancelled_awards": pl.Int8,
    "number_of_items": pl.Int32,
    "number_of_tenderers": pl.Int16,
    "number_of_bids": pl.Int16,
    "number_of_awards": pl.Int16,
    "number_of_contracts": pl.Int16,
    "number_of_documents": pl.Int16,
    "active_awards_count": pl.Int16,
    "main_cpv_2_digit": pl.Float32,
    "main_cpv_4_digit": pl.Float32,
    "tender_value": pl.Float64,
    "award_value": pl.Float64,
    "price_change_amount": pl.Float64,
    "price_change_pct": pl.Float64,
    "award_value_total": pl.Float64,
    "award_value_max": pl.Float64,
    "award_value_min": pl.Float64,
    "award_value_mean": pl.Float64,
    "award_value_std": pl.Float64,
    "award_concentration": pl.Float32,
    "discount_percentage_avg": pl.Float32,
    "discount_percentage_max": pl.Float32,
    "published_date": pl.Utf8,  # Will parse separately
    "award_date": pl.Utf8,
}

BID_SCHEMA = {
    "tender_id": pl.Utf8,
    "bid_id": pl.Utf8,
    "bidder_id": pl.Utf8,
    "bid_status": pl.Categorical,
    "bid_amount": pl.Float64,
    "is_winner": pl.Int8,
    "is_bidder_masked": pl.Int8,
    "bid_date": pl.Utf8,
}

DATE_COLUMNS = ["published_date", "award_date"]


def load_tenders(
    years: Optional[Union[int, List[int]]] = None,
    procurement_method: Optional[str] = None,
    columns: Optional[List[str]] = None,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
    return_polars: bool = False,
) -> Union[pl.DataFrame, pd.DataFrame]:
    """
    Load tender data with optional filtering using Polars.

    Args:
        years: Year(s) to load. None = all years.
        procurement_method: Filter by method ('limited', 'open', 'selective').
        columns: Columns to load. None = all columns.
        sample_frac: Random sample fraction (0-1). None = full data.
        random_state: Random seed for sampling.
        return_polars: If True, return Polars DataFrame. Default False (Pandas).

    Returns:
        DataFrame with tender data (Polars or Pandas).

    Examples:
        >>> # Load all 2023 tenders
        >>> df = load_tenders(years=2023)

        >>> # Load only Open tenders from 2023-2024
        >>> df = load_tenders(years=[2023, 2024], procurement_method='open')

        >>> # Load 10% sample, return Polars
        >>> df = load_tenders(sample_frac=0.1, return_polars=True)
    """
    if years is None:
        years = YEARS
    elif isinstance(years, int):
        years = [years]

    lazy_frames = []
    for year in years:
        file_path = TENDER_FILES.get(year)
        if file_path is None or not file_path.exists():
            print(f"Warning: File for {year} not found, skipping")
            continue

        # Lazy scan for efficiency
        lf = pl.scan_csv(
            file_path,
            schema_overrides={k: v for k, v in TENDER_SCHEMA.items()},
            ignore_errors=True,
        )

        # Select columns if specified
        if columns:
            available_cols = [c for c in columns if c in lf.columns]
            lf = lf.select(available_cols)

        # Filter by procurement method
        if procurement_method:
            lf = lf.filter(pl.col("procurement_method") == procurement_method)

        lazy_frames.append(lf)
        print(f"Scanning {year}...")

    if not lazy_frames:
        raise ValueError("No data found. Check years and file paths.")

    # Concatenate all lazy frames
    combined = pl.concat(lazy_frames)

    # Sample if requested (before collect for efficiency)
    if sample_frac is not None:
        combined = combined.collect().sample(fraction=sample_frac, seed=random_state)
        print(f"Sampled to {len(combined):,} records ({sample_frac*100:.0f}%)")
    else:
        combined = combined.collect()
        print(f"Loaded {len(combined):,} records")

    # Parse date columns
    for col in DATE_COLUMNS:
        if col in combined.columns:
            combined = combined.with_columns(
                pl.col(col).str.to_datetime(strict=False).alias(col)
            )

    if return_polars:
        return combined
    return combined.to_pandas()


def load_bids(
    years: Optional[Union[int, List[int]]] = None,
    columns: Optional[List[str]] = None,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
    return_polars: bool = False,
) -> Union[pl.DataFrame, pd.DataFrame]:
    """
    Load bid data with optional filtering using Polars.

    Args:
        years: Year(s) to load. None = all years.
        columns: Columns to load. None = all columns.
        sample_frac: Random sample fraction (0-1).
        random_state: Random seed for sampling.
        return_polars: If True, return Polars DataFrame.

    Returns:
        DataFrame with bid data.
    """
    if years is None:
        years = YEARS
    elif isinstance(years, int):
        years = [years]

    lazy_frames = []
    for year in years:
        file_path = BID_FILES.get(year)
        if file_path is None or not file_path.exists():
            print(f"Warning: Bids file for {year} not found, skipping")
            continue

        lf = pl.scan_csv(
            file_path,
            schema_overrides={k: v for k, v in BID_SCHEMA.items()},
            ignore_errors=True,
        )

        if columns:
            available_cols = [c for c in columns if c in lf.columns]
            lf = lf.select(available_cols)

        lazy_frames.append(lf)
        print(f"Scanning bids {year}...")

    if not lazy_frames:
        raise ValueError("No bid data found.")

    combined = pl.concat(lazy_frames)

    if sample_frac is not None:
        combined = combined.collect().sample(fraction=sample_frac, seed=random_state)
    else:
        combined = combined.collect()

    print(f"Loaded {len(combined):,} bids")

    # Parse bid_date
    if "bid_date" in combined.columns:
        combined = combined.with_columns(
            pl.col("bid_date").str.to_datetime(strict=False).alias("bid_date")
        )

    if return_polars:
        return combined
    return combined.to_pandas()


def load_buyers(return_polars: bool = False) -> Union[pl.DataFrame, pd.DataFrame]:
    """Load buyers reference table."""
    df = pl.read_csv(BUYERS_FILE)
    print(f"Loaded buyers: {len(df):,}")

    if return_polars:
        return df
    return df.to_pandas()


def load_suppliers(return_polars: bool = False) -> Union[pl.DataFrame, pd.DataFrame]:
    """Load suppliers reference table."""
    df = pl.read_csv(SUPPLIERS_FILE)
    print(f"Loaded suppliers: {len(df):,}")

    if return_polars:
        return df
    return df.to_pandas()


def load_bidders(return_polars: bool = False) -> Union[pl.DataFrame, pd.DataFrame]:
    """Load bidders reference table."""
    df = pl.read_csv(BIDDERS_FILE)
    print(f"Loaded bidders: {len(df):,}")

    if return_polars:
        return df
    return df.to_pandas()


def load_open_tenders(years: Optional[Union[int, List[int]]] = None, **kwargs) -> pd.DataFrame:
    """Shortcut: Load only Open tenders (5.5% of data)."""
    return load_tenders(years=years, procurement_method=ProcurementMethod.OPEN, **kwargs)


def load_limited_tenders(years: Optional[Union[int, List[int]]] = None, **kwargs) -> pd.DataFrame:
    """Shortcut: Load only Limited tenders (91% of data)."""
    return load_tenders(years=years, procurement_method=ProcurementMethod.LIMITED, **kwargs)


def load_selective_tenders(years: Optional[Union[int, List[int]]] = None, **kwargs) -> pd.DataFrame:
    """Shortcut: Load only Selective tenders (3.3% of data)."""
    return load_tenders(years=years, procurement_method=ProcurementMethod.SELECTIVE, **kwargs)


def merge_with_buyers(
    tenders: Union[pl.DataFrame, pd.DataFrame],
    buyers: Optional[Union[pl.DataFrame, pd.DataFrame]] = None,
    return_polars: bool = False,
) -> Union[pl.DataFrame, pd.DataFrame]:
    """
    Merge tenders with buyer information.

    Adds: buyer_name, buyer_region, single_bidder_rate, competitive_rate,
          supplier_diversity_index, etc.
    """
    # Convert to Polars if needed
    if isinstance(tenders, pd.DataFrame):
        tenders_pl = pl.from_pandas(tenders)
    else:
        tenders_pl = tenders

    if buyers is None:
        buyers_pl = load_buyers(return_polars=True)
    elif isinstance(buyers, pd.DataFrame):
        buyers_pl = pl.from_pandas(buyers)
    else:
        buyers_pl = buyers

    result = tenders_pl.join(buyers_pl, on="buyer_id", how="left", suffix="_buyer")

    if return_polars:
        return result
    return result.to_pandas()


def merge_with_suppliers(
    tenders: Union[pl.DataFrame, pd.DataFrame],
    suppliers: Optional[Union[pl.DataFrame, pd.DataFrame]] = None,
    return_polars: bool = False,
) -> Union[pl.DataFrame, pd.DataFrame]:
    """
    Merge tenders with supplier information.

    Adds: supplier_name, supplier_region, total_awards, total_value.
    """
    if isinstance(tenders, pd.DataFrame):
        tenders_pl = pl.from_pandas(tenders)
    else:
        tenders_pl = tenders

    if suppliers is None:
        suppliers_pl = load_suppliers(return_polars=True)
    elif isinstance(suppliers, pd.DataFrame):
        suppliers_pl = pl.from_pandas(suppliers)
    else:
        suppliers_pl = suppliers

    result = tenders_pl.join(suppliers_pl, on="supplier_id", how="left", suffix="_supplier")

    if return_polars:
        return result
    return result.to_pandas()


def get_tenders_with_bids(
    years: Optional[Union[int, List[int]]] = None,
    min_bids: int = 2,
    return_polars: bool = False,
) -> Union[pl.DataFrame, pd.DataFrame]:
    """
    Load tenders that have associated bids data.

    Useful for statistical screens that require bid-level analysis.

    Args:
        years: Year(s) to load.
        min_bids: Minimum number of bids per tender.
        return_polars: If True, return Polars DataFrame.

    Returns:
        DataFrame with tenders that have >= min_bids.
    """
    tenders = load_tenders(years=years, return_polars=True)
    bids = load_bids(years=years, return_polars=True)

    # Count bids per tender
    bid_counts = bids.group_by("tender_id").agg(
        pl.count().alias("actual_bid_count")
    )

    # Join and filter
    result = tenders.join(bid_counts, on="tender_id", how="inner")
    result = result.filter(pl.col("actual_bid_count") >= min_bids)

    print(f"Tenders with >= {min_bids} bids: {len(result):,}")

    if return_polars:
        return result
    return result.to_pandas()


def memory_usage(df: Union[pl.DataFrame, pd.DataFrame]) -> str:
    """Get human-readable memory usage of DataFrame."""
    if isinstance(df, pl.DataFrame):
        mem = df.estimated_size()
    else:
        mem = df.memory_usage(deep=True).sum()

    if mem < 1024**2:
        return f"{mem / 1024:.1f} KB"
    elif mem < 1024**3:
        return f"{mem / 1024**2:.1f} MB"
    else:
        return f"{mem / 1024**3:.2f} GB"


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


# === Polars-native aggregation functions ===

def aggregate_by_buyer(
    tenders: Union[pl.DataFrame, pd.DataFrame],
    return_polars: bool = False,
) -> Union[pl.DataFrame, pd.DataFrame]:
    """
    Aggregate tender data by buyer for clustering/analysis.

    Returns buyer-level features:
    - total_tenders, total_value, avg_value
    - single_bidder_rate, competitive_rate
    - unique_suppliers, supplier_diversity
    """
    if isinstance(tenders, pd.DataFrame):
        tenders_pl = pl.from_pandas(tenders)
    else:
        tenders_pl = tenders

    result = tenders_pl.group_by("buyer_id").agg([
        pl.count().alias("total_tenders"),
        pl.col("tender_value").sum().alias("total_value"),
        pl.col("tender_value").mean().alias("avg_value"),
        pl.col("tender_value").median().alias("median_value"),
        pl.col("is_single_bidder").mean().alias("single_bidder_rate"),
        pl.col("is_competitive").mean().alias("competitive_rate"),
        pl.col("price_change_pct").mean().alias("avg_discount_pct"),
        pl.col("supplier_id").n_unique().alias("unique_suppliers"),
        pl.col("number_of_tenderers").mean().alias("avg_tenderers"),
    ])

    # Calculate supplier diversity index (unique suppliers / total tenders)
    result = result.with_columns(
        (pl.col("unique_suppliers") / pl.col("total_tenders")).alias("supplier_diversity_index")
    )

    if return_polars:
        return result
    return result.to_pandas()


def aggregate_by_supplier(
    tenders: Union[pl.DataFrame, pd.DataFrame],
    return_polars: bool = False,
) -> Union[pl.DataFrame, pd.DataFrame]:
    """
    Aggregate tender data by supplier for clustering/analysis.

    Returns supplier-level features:
    - total_awards, total_value, avg_value
    - unique_buyers, buyer_concentration
    - single_bidder_rate, avg_competitors
    """
    if isinstance(tenders, pd.DataFrame):
        tenders_pl = pl.from_pandas(tenders)
    else:
        tenders_pl = tenders

    result = tenders_pl.group_by("supplier_id").agg([
        pl.count().alias("total_awards"),
        pl.col("tender_value").sum().alias("total_value"),
        pl.col("tender_value").mean().alias("avg_award_value"),
        pl.col("buyer_id").n_unique().alias("buyer_count"),
        pl.col("is_single_bidder").mean().alias("single_bidder_rate"),
        pl.col("number_of_tenderers").mean().alias("avg_competitors"),
    ])

    if return_polars:
        return result
    return result.to_pandas()


def aggregate_by_pair(
    tenders: Union[pl.DataFrame, pd.DataFrame],
    min_contracts: int = 1,
    return_polars: bool = False,
) -> Union[pl.DataFrame, pd.DataFrame]:
    """
    Aggregate tender data by buyer-supplier pair for relationship analysis.

    Returns pair-level features:
    - contracts_count, total_value, avg_value
    - single_bidder_rate
    - exclusivity metrics
    """
    if isinstance(tenders, pd.DataFrame):
        tenders_pl = pl.from_pandas(tenders)
    else:
        tenders_pl = tenders

    # Aggregate by pair
    pair_agg = tenders_pl.group_by(["buyer_id", "supplier_id"]).agg([
        pl.count().alias("contracts_count"),
        pl.col("tender_value").sum().alias("total_value"),
        pl.col("tender_value").mean().alias("avg_value"),
        pl.col("is_single_bidder").mean().alias("single_bidder_rate"),
    ])

    # Filter by minimum contracts
    pair_agg = pair_agg.filter(pl.col("contracts_count") >= min_contracts)

    # Calculate exclusivity metrics
    buyer_totals = tenders_pl.group_by("buyer_id").agg(
        pl.count().alias("buyer_total")
    )
    supplier_totals = tenders_pl.group_by("supplier_id").agg(
        pl.count().alias("supplier_total")
    )

    pair_agg = pair_agg.join(buyer_totals, on="buyer_id", how="left")
    pair_agg = pair_agg.join(supplier_totals, on="supplier_id", how="left")

    pair_agg = pair_agg.with_columns([
        (pl.col("contracts_count") / pl.col("buyer_total")).alias("exclusivity_buyer"),
        (pl.col("contracts_count") / pl.col("supplier_total")).alias("exclusivity_supplier"),
    ]).drop(["buyer_total", "supplier_total"])

    if return_polars:
        return pair_agg
    return pair_agg.to_pandas()
