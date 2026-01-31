"""
Autoencoder-based Anomaly Detection for Public Procurement.

Uses reconstruction error as anomaly score:
- Train autoencoder on "normal" data patterns
- High reconstruction error = anomaly

Author: Roman Kachmar
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from typing import Optional, List, Dict, Union, Literal
import polars as pl

from ..data_loader import aggregate_by_buyer, aggregate_by_supplier, aggregate_by_pair

# Features to log-transform
LOG_TRANSFORM_FEATURES = [
    "total_value", "tender_value", "award_value", "avg_value", "avg_tender_value",
    "avg_award_value", "total_savings", "median_value",
    "total_awards", "total_tenders", "contracts_count", "buyer_count",
]


class Autoencoder(nn.Module):
    """Simple autoencoder architecture."""

    def __init__(self, input_dim: int, encoding_dim: int = 8, hidden_dim: int = 32):
        super().__init__()

        # Encoder: input -> hidden -> encoding
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, encoding_dim),
            nn.ReLU(),
        )

        # Decoder: encoding -> hidden -> output
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


class AutoencoderDetector:
    """
    Autoencoder-based anomaly detector for tender-level data.

    Uses reconstruction error as anomaly score:
    - Low error = normal pattern (autoencoder learned it well)
    - High error = anomaly (doesn't fit learned patterns)

    Usage:
        detector = AutoencoderDetector(encoding_dim=8, epochs=50)
        results = detector.fit_detect(tenders, buyers_df=buyers)
        anomalies = detector.get_anomalies(min_score=0.9)
    """

    def __init__(
        self,
        encoding_dim: int = 8,
        hidden_dim: int = 32,
        epochs: int = 50,
        batch_size: int = 1024,
        learning_rate: float = 0.001,
        contamination: float = 0.05,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        """
        Initialize Autoencoder detector.

        Args:
            encoding_dim: Size of the bottleneck layer
            hidden_dim: Size of hidden layers
            epochs: Training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            contamination: Expected proportion of anomalies
            device: 'cuda' or 'cpu' (auto-detect if None)
            random_state: Random seed
        """
        self.encoding_dim = encoding_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.contamination = contamination
        self.random_state = random_state

        # Auto-detect device (with compatibility check)
        if device is None:
            self.device = self._get_device()
        else:
            self.device = torch.device(device)

        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_names_ = None
        self.results = None
        self.training_loss_ = []

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

    @staticmethod
    def _get_device():
        """Get available device with compatibility check.

        Checks CUDA compute capability to ensure compatibility.
        RTX 5070 Ti (Blackwell, sm_120) is not yet supported by PyTorch.
        Supported compute capabilities: 3.5 - 9.0 (as of PyTorch 2.x)
        """
        if not torch.cuda.is_available():
            return torch.device("cpu")

        try:
            # Check compute capability - sm_120 (12.0) is NOT supported
            # PyTorch 2.x supports up to sm_90 (9.0) - Hopper architecture
            capability = torch.cuda.get_device_capability(0)
            major, minor = capability

            # Blackwell (sm_120) = major 12, not supported
            # Hopper (sm_90) = major 9, supported
            # Ada Lovelace (sm_89) = major 8.9, supported
            if major >= 10:
                print(f"  Note: GPU compute capability {major}.{minor} (sm_{major}{minor}) not yet supported by PyTorch, using CPU")
                return torch.device("cpu")

            # Additional test: try to actually run something on GPU
            test_tensor = torch.zeros(1, device="cuda")
            del test_tensor
            torch.cuda.empty_cache()
            return torch.device("cuda")
        except Exception as e:
            print(f"  Note: CUDA not compatible ({e}), using CPU")
            return torch.device("cpu")

    def _preprocess(self, df: pd.DataFrame, feature_cols: List[str], fit: bool = True) -> np.ndarray:
        """Log-transform, impute, and scale features."""
        X = df[feature_cols].copy()

        # Log-transform skewed features
        for col in X.columns:
            if col in LOG_TRANSFORM_FEATURES:
                X[col] = np.log1p(X[col].clip(lower=0))

        if fit:
            self.imputer = SimpleImputer(strategy="median")
            X_imputed = self.imputer.fit_transform(X)
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X_imputed)
        else:
            X_imputed = self.imputer.transform(X)
            X_scaled = self.scaler.transform(X_imputed)

        return X_scaled.astype(np.float32)

    def fit_detect(
        self,
        tenders: pd.DataFrame,
        buyers_df: Optional[pd.DataFrame] = None,
        suppliers_df: Optional[pd.DataFrame] = None,
        sample_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fit autoencoder and detect anomalies.

        Args:
            tenders: Tenders DataFrame
            buyers_df: Optional buyers DataFrame
            suppliers_df: Optional suppliers DataFrame
            sample_size: If set, train on sample (for large datasets)

        Returns:
            DataFrame with anomaly scores and labels
        """
        print(f"AutoencoderDetector")
        print(f"  Device: {self.device}")
        print(f"  Architecture: input -> {self.hidden_dim} -> {self.encoding_dim} -> {self.hidden_dim} -> output")
        print(f"Processing {len(tenders):,} tenders...")

        # Sample if needed
        if sample_size and len(tenders) > sample_size:
            print(f"  Training on sample of {sample_size:,} tenders...")
            tenders_train = tenders.sample(sample_size, random_state=self.random_state)
        else:
            tenders_train = tenders

        # Prepare features
        print("Step 1/4: Preparing features...")
        feature_cols = self._get_feature_cols(tenders, buyers_df, suppliers_df)
        df_merged = self._merge_features(tenders, buyers_df, suppliers_df, feature_cols)
        df_train = self._merge_features(tenders_train, buyers_df, suppliers_df, feature_cols)

        self.feature_names_ = feature_cols
        print(f"  Features: {len(feature_cols)}")

        # Preprocess
        print("Step 2/4: Preprocessing...")
        X_train = self._preprocess(df_train, feature_cols, fit=True)
        X_all = self._preprocess(df_merged, feature_cols, fit=False)
        print(f"  Shape: {X_all.shape}")

        # Build model
        input_dim = X_train.shape[1]
        self.model = Autoencoder(input_dim, self.encoding_dim, self.hidden_dim).to(self.device)

        # Train
        print(f"Step 3/4: Training ({self.epochs} epochs)...")
        self._train(X_train)

        # Compute reconstruction errors
        print("Step 4/4: Computing reconstruction errors...")
        errors = self._compute_errors(X_all)

        # Normalize to 0-1
        errors_norm = (errors - errors.min()) / (errors.max() - errors.min() + 1e-10)

        # Determine anomaly threshold
        threshold = np.percentile(errors_norm, 100 * (1 - self.contamination))

        # Build results
        result = tenders[["tender_id"]].copy()
        result["score"] = errors_norm
        result["anomaly"] = (errors_norm >= threshold).astype(int)
        result["risk_level"] = pd.cut(
            errors_norm,
            bins=[0, 0.5, 0.75, 0.9, 1.01],
            labels=["low", "medium", "high", "critical"]
        )

        self.results = result

        n_anomalies = result["anomaly"].sum()
        print(f"\nAutoencoder complete!")
        print(f"  Final loss: {self.training_loss_[-1]:.6f}")
        print(f"  Anomalies: {n_anomalies:,} ({n_anomalies/len(result)*100:.2f}%)")

        return result

    def _get_feature_cols(self, tenders, buyers_df, suppliers_df):
        """Get list of feature columns."""
        feature_cols = []

        # Tender features
        tender_features = [
            "tender_value", "price_change_pct", "number_of_tenderers",
            "is_single_bidder", "is_competitive", "is_weekend", "is_q4", "is_december"
        ]
        for col in tender_features:
            if col in tenders.columns:
                feature_cols.append(col)

        # Buyer features
        if buyers_df is not None:
            buyer_features = ["single_bidder_rate", "competitive_rate", "avg_discount_pct", "supplier_diversity_index"]
            for col in buyer_features:
                if col in buyers_df.columns:
                    feature_cols.append(col)

        # Supplier features
        if suppliers_df is not None:
            supplier_features = ["total_awards", "total_value"]
            for col in supplier_features:
                if col in suppliers_df.columns:
                    feature_cols.append(col)

        return feature_cols

    def _merge_features(self, tenders, buyers_df, suppliers_df, feature_cols):
        """Merge buyer and supplier features into tenders."""
        df = tenders.copy()

        if buyers_df is not None and "buyer_id" in df.columns:
            buyer_cols = [c for c in feature_cols if c in buyers_df.columns]
            if buyer_cols:
                df = df.merge(buyers_df[["buyer_id"] + buyer_cols], on="buyer_id", how="left")

        if suppliers_df is not None and "supplier_id" in df.columns:
            supplier_cols = [c for c in feature_cols if c in suppliers_df.columns]
            if supplier_cols:
                df = df.merge(suppliers_df[["supplier_id"] + supplier_cols], on="supplier_id", how="left")

        return df

    def _train(self, X: np.ndarray):
        """Train the autoencoder."""
        dataset = TensorDataset(torch.tensor(X))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        self.training_loss_ = []

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                x = batch[0].to(self.device)

                optimizer.zero_grad()
                output = self.model(x)
                loss = criterion(output, x)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            self.training_loss_.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")

    def _compute_errors(self, X: np.ndarray) -> np.ndarray:
        """Compute reconstruction errors for all samples."""
        self.model.eval()

        dataset = TensorDataset(torch.tensor(X))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        errors = []
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].to(self.device)
                output = self.model(x)
                # MSE per sample
                batch_errors = ((x - output) ** 2).mean(dim=1).cpu().numpy()
                errors.extend(batch_errors)

        return np.array(errors)

    def get_anomalies(self, min_score: float = 0.75) -> pd.DataFrame:
        """Get tenders with score above threshold."""
        if self.results is None:
            raise ValueError("Run fit_detect() first")
        return self.results[self.results["score"] >= min_score]

    def summary(self) -> pd.DataFrame:
        """Get summary of results."""
        if self.results is None:
            raise ValueError("Run fit_detect() first")

        total = len(self.results)
        risk_counts = self.results["risk_level"].value_counts()

        summary_data = [
            {"metric": "total_tenders", "value": total},
            {"metric": "anomalies", "value": self.results["anomaly"].sum()},
            {"metric": "anomaly_rate", "value": self.results["anomaly"].mean() * 100},
            {"metric": "final_loss", "value": self.training_loss_[-1] if self.training_loss_ else 0},
        ]

        for risk in ["critical", "high", "medium", "low"]:
            count = risk_counts.get(risk, 0)
            summary_data.append({"metric": f"risk_{risk}", "value": count})

        return pd.DataFrame(summary_data)


class AggregatedAutoencoder:
    """
    Autoencoder-based anomaly detection at aggregated levels.

    Detects anomalous buyers, suppliers, and buyer-supplier pairs
    based on their behavioral patterns.

    Usage:
        detector = AggregatedAutoencoder(encoding_dim=4, epochs=30)

        buyer_results = detector.detect_buyers(tenders, buyers)
        supplier_results = detector.detect_suppliers(tenders)
        pair_results = detector.detect_pairs(tenders, min_contracts=3)
    """

    def __init__(
        self,
        encoding_dim: int = 4,
        hidden_dim: int = 16,
        epochs: int = 30,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        contamination: float = 0.05,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.encoding_dim = encoding_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.contamination = contamination
        self.random_state = random_state

        # Auto-detect device (with compatibility check)
        if device is None:
            self.device = AutoencoderDetector._get_device()
        else:
            self.device = torch.device(device)

        self.buyer_results_ = None
        self.supplier_results_ = None
        self.pair_results_ = None

        torch.manual_seed(random_state)
        np.random.seed(random_state)

    def _preprocess(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """Log-transform, impute, and scale."""
        X = df[feature_cols].copy()

        for col in X.columns:
            if col in LOG_TRANSFORM_FEATURES:
                X[col] = np.log1p(X[col].clip(lower=0))

        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X)

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        return X_scaled.astype(np.float32)

    def _train_and_score(self, X: np.ndarray, verbose: bool = True) -> np.ndarray:
        """Train autoencoder and return reconstruction errors."""
        input_dim = X.shape[1]
        model = Autoencoder(input_dim, self.encoding_dim, self.hidden_dim).to(self.device)

        dataset = TensorDataset(torch.tensor(X))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                x = batch[0].to(self.device)
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if verbose and (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/len(dataloader):.6f}")

        # Compute errors
        model.eval()
        dataloader_eval = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        errors = []
        with torch.no_grad():
            for batch in dataloader_eval:
                x = batch[0].to(self.device)
                output = model(x)
                batch_errors = ((x - output) ** 2).mean(dim=1).cpu().numpy()
                errors.extend(batch_errors)

        return np.array(errors)

    def detect_buyers(
        self,
        tenders: Union[pd.DataFrame, pl.DataFrame],
        buyers_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Detect anomalous buyers."""
        print(f"AggregatedAutoencoder: Detecting anomalous BUYERS...")
        print(f"  Device: {self.device}")

        # Get buyer features
        if buyers_df is not None and all(col in buyers_df.columns for col in
            ["single_bidder_rate", "competitive_rate", "supplier_diversity_index"]):
            buyer_agg = buyers_df.copy()
            print(f"  Using pre-computed buyer features")
        else:
            print(f"  Computing buyer features from tenders...")
            buyer_agg = aggregate_by_buyer(tenders, return_polars=False)

        # Select features
        feature_cols = []
        for col in ["single_bidder_rate", "competitive_rate", "avg_discount_pct",
                    "supplier_diversity_index", "total_tenders", "avg_tender_value", "total_value"]:
            if col in buyer_agg.columns:
                feature_cols.append(col)

        print(f"  Features: {feature_cols}")
        print(f"  Buyers: {len(buyer_agg):,}")

        # Train and score
        X = self._preprocess(buyer_agg, feature_cols)
        errors = self._train_and_score(X)

        # Normalize
        scores = (errors - errors.min()) / (errors.max() - errors.min() + 1e-10)
        threshold = np.percentile(scores, 100 * (1 - self.contamination))

        # Build results
        result = buyer_agg[["buyer_id"]].copy()
        result["score"] = scores
        result["anomaly"] = (scores >= threshold).astype(int)
        result["risk_level"] = pd.cut(
            scores, bins=[0, 0.5, 0.75, 0.9, 1.01],
            labels=["low", "medium", "high", "critical"]
        )

        for col in feature_cols:
            result[col] = buyer_agg[col].values

        self.buyer_results_ = result

        n_anomalies = result["anomaly"].sum()
        print(f"  Anomalies: {n_anomalies:,} ({n_anomalies/len(result)*100:.1f}%)")

        return result

    def detect_suppliers(
        self,
        tenders: Union[pd.DataFrame, pl.DataFrame],
        suppliers_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Detect anomalous suppliers."""
        print(f"AggregatedAutoencoder: Detecting anomalous SUPPLIERS...")
        print(f"  Device: {self.device}")

        print(f"  Computing supplier features from tenders...")
        supplier_agg = aggregate_by_supplier(tenders, return_polars=False)

        # Select features
        feature_cols = []
        for col in ["total_awards", "total_value", "avg_award_value",
                    "buyer_count", "single_bidder_rate", "avg_competitors"]:
            if col in supplier_agg.columns:
                feature_cols.append(col)

        print(f"  Features: {feature_cols}")
        print(f"  Suppliers: {len(supplier_agg):,}")

        # Train and score
        X = self._preprocess(supplier_agg, feature_cols)
        errors = self._train_and_score(X)

        # Normalize
        scores = (errors - errors.min()) / (errors.max() - errors.min() + 1e-10)
        threshold = np.percentile(scores, 100 * (1 - self.contamination))

        # Build results
        result = supplier_agg[["supplier_id"]].copy()
        result["score"] = scores
        result["anomaly"] = (scores >= threshold).astype(int)
        result["risk_level"] = pd.cut(
            scores, bins=[0, 0.5, 0.75, 0.9, 1.01],
            labels=["low", "medium", "high", "critical"]
        )

        for col in feature_cols:
            result[col] = supplier_agg[col].values

        self.supplier_results_ = result

        n_anomalies = result["anomaly"].sum()
        print(f"  Anomalies: {n_anomalies:,} ({n_anomalies/len(result)*100:.1f}%)")

        return result

    def detect_pairs(
        self,
        tenders: Union[pd.DataFrame, pl.DataFrame],
        min_contracts: int = 3,
    ) -> pd.DataFrame:
        """Detect anomalous buyer-supplier pairs."""
        print(f"AggregatedAutoencoder: Detecting anomalous PAIRS...")
        print(f"  Device: {self.device}")

        print(f"  Computing pair features from tenders...")
        pair_agg = aggregate_by_pair(tenders, min_contracts=min_contracts, return_polars=False)
        print(f"  Pairs with {min_contracts}+ contracts: {len(pair_agg):,}")

        if len(pair_agg) < 100:
            print(f"  Not enough pairs for training. Returning empty.")
            return pd.DataFrame()

        # Select features
        feature_cols = [
            "contracts_count", "total_value", "avg_value",
            "single_bidder_rate", "exclusivity_buyer", "exclusivity_supplier"
        ]

        print(f"  Features: {feature_cols}")

        # Train and score
        X = self._preprocess(pair_agg, feature_cols)
        errors = self._train_and_score(X)

        # Normalize
        scores = (errors - errors.min()) / (errors.max() - errors.min() + 1e-10)
        threshold = np.percentile(scores, 100 * (1 - self.contamination))

        # Build results
        result = pair_agg[["buyer_id", "supplier_id"]].copy()
        result["score"] = scores
        result["anomaly"] = (scores >= threshold).astype(int)
        result["risk_level"] = pd.cut(
            scores, bins=[0, 0.5, 0.75, 0.9, 1.01],
            labels=["low", "medium", "high", "critical"]
        )

        for col in feature_cols:
            result[col] = pair_agg[col].values

        self.pair_results_ = result

        n_anomalies = result["anomaly"].sum()
        print(f"  Anomalies: {n_anomalies:,} ({n_anomalies/len(result)*100:.1f}%)")

        return result

    def get_anomalies(
        self,
        level: Literal["buyers", "suppliers", "pairs"],
        min_score: float = 0.5,
    ) -> pd.DataFrame:
        """Get entities with score above threshold."""
        results_map = {
            "buyers": self.buyer_results_,
            "suppliers": self.supplier_results_,
            "pairs": self.pair_results_,
        }

        results = results_map.get(level)
        if results is None:
            raise ValueError(f"Run detect_{level}() first")

        return results[results["score"] >= min_score]

    def summary(self) -> Dict[str, pd.DataFrame]:
        """Get summary of all detection results."""
        summaries = {}

        for name, results in [
            ("buyers", self.buyer_results_),
            ("suppliers", self.supplier_results_),
            ("pairs", self.pair_results_),
        ]:
            if results is not None:
                summaries[name] = pd.DataFrame([
                    {"metric": "total", "value": len(results)},
                    {"metric": "anomalies", "value": results["anomaly"].sum()},
                    {"metric": "anomaly_pct", "value": results["anomaly"].mean() * 100},
                    {"metric": "critical", "value": (results["risk_level"] == "critical").sum()},
                    {"metric": "high", "value": (results["risk_level"] == "high").sum()},
                ])

        return summaries
