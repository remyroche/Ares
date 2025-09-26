"""Tree-based decision layer powered by PatchTST feature extraction.

This module replaces the legacy CLVSA wrapper with a modern PatchTST-based
sequence encoder that produces multi-horizon forecasts, embeddings, and
uncertainty estimates which are then consumed by tree-based decision models.
"""

from __future__ import annotations

import copy
import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.preprocessing import RobustScaler, StandardScaler


logger = logging.getLogger(__name__)


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class PatchTSTTreeConfig:
    """Configuration for the PatchTST + tree hybrid pipeline."""

    lookback: int = 512
    patch_size: int = 16
    patch_stride: int = 8
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1
    forecast_horizons: Tuple[int, ...] = (3, 6, 12)
    task_type: str = "both"  # "regression", "classification", "both"
    quantile_levels: Tuple[float, ...] = (0.1, 0.5, 0.9)
    use_quantile_heads: bool = True
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 40
    patience: int = 7
    smoothing: float = 0.05
    winsorize_pct: float = 0.005
    huber_delta: Optional[float] = None
    huber_delta_scale: float = 1.5
    train_val_split: float = 0.8
    embargo: int = 12
    device: str = field(default_factory=_default_device)
    use_amp: bool = True
    detach_embeddings: bool = True
    min_samples: int = 128
    feature_scaler: str = "standard"  # "standard" or "robust"
    channel_scaler: str = "robust"  # currently robust per channel
    num_workers: int = 0
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.lookback <= 0:
            raise ValueError("lookback must be positive")
        if self.patch_size <= 0 or self.patch_size > self.lookback:
            raise ValueError("patch_size must be in (0, lookback]")
        if self.patch_stride <= 0:
            raise ValueError("patch_stride must be positive")
        if any(h <= 0 for h in self.forecast_horizons):
            raise ValueError("forecast horizons must be positive")
        if self.train_val_split <= 0 or self.train_val_split >= 1:
            raise ValueError("train_val_split must be in (0,1)")


class PatchTSTSequenceDataset(Dataset):
    """Dataset of sliding windows for PatchTST training and inference."""

    def __init__(
        self,
        sequences: np.ndarray,
        handcrafted_features: np.ndarray,
        indices: Sequence[pd.Timestamp],
        regression_targets: Optional[np.ndarray] = None,
        classification_targets: Optional[np.ndarray] = None,
    ) -> None:
        if sequences.ndim != 3:
            raise ValueError("sequences must have shape (N, lookback, channels)")
        self.sequences = torch.from_numpy(sequences.astype(np.float32))
        self.handcrafted_features = torch.from_numpy(handcrafted_features.astype(np.float32))
        self.indices = np.array(indices)
        self.regression_targets = (
            torch.from_numpy(regression_targets.astype(np.float32))
            if regression_targets is not None
            else None
        )
        self.classification_targets = (
            torch.from_numpy(classification_targets.astype(np.float32))
            if classification_targets is not None
            else None
        )

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "inputs": self.sequences[idx],
            "context": self.handcrafted_features[idx],
        }
        if self.regression_targets is not None:
            item["regression"] = self.regression_targets[idx]
        if self.classification_targets is not None:
            item["classification"] = self.classification_targets[idx]
        return item

    @property
    def has_regression(self) -> bool:
        return self.regression_targets is not None

    @property
    def has_classification(self) -> bool:
        return self.classification_targets is not None


class ChannelIndependentPatchEmbedding(nn.Module):
    """Patch embedding that keeps channels independent before mixing."""

    def __init__(self, input_dim: int, patch_size: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.d_model = d_model
        self.channel_proj = nn.Linear(patch_size, d_model)
        self.channel_dropout = nn.Dropout(dropout)
        self.mixer = nn.Linear(input_dim * d_model, d_model)

    def forward(self, x: torch.Tensor, patch_stride: int) -> torch.Tensor:
        patches = x.unfold(dimension=1, size=self.patch_size, step=patch_stride)
        patches = patches.permute(0, 3, 1, 2)
        channel_embeddings = self.channel_proj(patches)
        channel_embeddings = self.channel_dropout(channel_embeddings)
        channel_embeddings = channel_embeddings.permute(0, 2, 1, 3)
        batch, num_patches, channels, dim = channel_embeddings.shape
        mixed = channel_embeddings.reshape(batch, num_patches, channels * dim)
        return self.mixer(mixed)


class PatchTSTModel(nn.Module):
    """Minimal PatchTST encoder with regression/classification heads."""

    def __init__(
        self,
        input_dim: int,
        num_patches: int,
        config: PatchTSTTreeConfig,
        num_horizons: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_horizons = num_horizons
        self.embedding = ChannelIndependentPatchEmbedding(
            input_dim=input_dim,
            patch_size=config.patch_size,
            d_model=config.d_model,
            dropout=config.dropout,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches, config.d_model))
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        if config.task_type in {"regression", "both"}:
            self.regression_head = nn.Linear(config.d_model, num_horizons)
        else:
            self.regression_head = None
        if config.task_type in {"classification", "both"}:
            self.classification_head = nn.Linear(config.d_model, num_horizons)
        else:
            self.classification_head = None
        if config.use_quantile_heads:
            self.quantile_head = nn.Linear(config.d_model, num_horizons * len(config.quantile_levels))
        else:
            self.quantile_head = None

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        num_patches = 1 + (inputs.shape[1] - self.config.patch_size) // self.config.patch_stride
        patch_tokens = self.embedding(inputs, self.config.patch_stride)
        if patch_tokens.shape[1] != num_patches:
            patch_tokens = patch_tokens[:, :num_patches]
        tokens = patch_tokens + self.positional_embedding[:, :num_patches]
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        outputs: Dict[str, torch.Tensor] = {"embedding": pooled, "tokens": encoded}
        if self.regression_head is not None:
            outputs["regression"] = self.regression_head(pooled)
        if self.classification_head is not None:
            outputs["classification"] = self.classification_head(pooled)
        if self.quantile_head is not None:
            quantiles = self.quantile_head(pooled)
            outputs["quantiles"] = quantiles.view(-1, self.num_horizons, len(self.config.quantile_levels))
        return outputs


def _pinball_loss(pred: torch.Tensor, target: torch.Tensor, taus: Sequence[float]) -> torch.Tensor:
    diff = target.unsqueeze(-1) - pred
    losses = []
    for i, tau in enumerate(taus):
        error = diff[..., i]
        losses.append(torch.maximum(tau * error, (tau - 1) * error))
    stacked = torch.stack(losses, dim=-1)
    return stacked.mean()


class TreePatchTSTWrapper(BaseEstimator, RegressorMixin, ClassifierMixin):
    """Wrapper that marries PatchTST feature extraction with tree models."""

    def __init__(
        self,
        base_model: Any,
        config: PatchTSTTreeConfig,
        classification_model: Optional[Any] = None,
    ) -> None:
        self.base_model = base_model
        self.classification_model = classification_model
        self.config = config
        self.horizons = tuple(sorted(config.forecast_horizons))
        self.patch_model: Optional[PatchTSTModel] = None
        self.patch_state: Optional[Dict[str, Any]] = None
        self.tree_models_reg: Dict[int, Any] = {}
        self.tree_models_cls: Dict[int, Any] = {}
        self.channel_scalers: Dict[str, RobustScaler] = {}
        self.tree_feature_scaler: Optional[StandardScaler] = None
        self.tree_feature_columns: List[str] = []
        self.context_columns: List[str] = []
        self.handcrafted_feature_names: List[str] = []
        self.training_metadata: Dict[str, Any] = {}
        self.last_patch_outputs_: Dict[str, pd.DataFrame] = {}
        self.last_tree_features_: Optional[pd.DataFrame] = None
        self.is_fitted: bool = False
        torch.manual_seed(self.config.random_state)

    # ------------------------------------------------------------------
    # Data preparation utilities
    # ------------------------------------------------------------------
    def _ensure_dataframe(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X)
        raise TypeError("X must be a pandas DataFrame or numpy array")

    def _winsorize(self, series: pd.Series) -> pd.Series:
        lower = series.quantile(self.config.winsorize_pct)
        upper = series.quantile(1 - self.config.winsorize_pct)
        return series.clip(lower, upper)

    def _compute_targets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if "close" not in df.columns:
            raise ValueError("market data must contain a 'close' column for return targets")
        close = df["close"].astype(float)
        regression_targets = {}
        classification_targets = {}
        for horizon in self.horizons:
            log_ret = np.log(close.shift(-horizon) / close)
            log_ret = self._winsorize(log_ret)
            regression_targets[horizon] = log_ret
            direction = pd.Series(self.config.smoothing, index=df.index, dtype=float)
            direction[log_ret > 0] = 1.0 - self.config.smoothing
            direction[log_ret.isna()] = np.nan
            classification_targets[horizon] = direction
        reg_df = pd.DataFrame({h: regression_targets[h] for h in self.horizons})
        cls_df = pd.DataFrame({h: classification_targets[h] for h in self.horizons})
        return reg_df, cls_df

    def _compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df.copy()
        if "close" in features.columns:
            close = features["close"].astype(float)
            for horizon in (1, 3, 6, 12):
                features[f"log_return_{horizon}"] = np.log(close / close.shift(horizon))
            look = min(96, max(16, self.config.lookback // 4))
            rolling_mean = close.rolling(look).mean()
            rolling_std = close.rolling(look).std()
            features["close_z"] = (close - rolling_mean) / (rolling_std + 1e-8)
            returns_1 = close.pct_change()
            features["realized_vol"] = returns_1.rolling(look).std()
            features["momentum"] = returns_1.rolling(max(look // 2, 4)).mean()
        if {"high", "low", "close"}.issubset(features.columns):
            high = features["high"].astype(float)
            low = features["low"].astype(float)
            close = features["close"].astype(float)
            prev_close = close.shift(1)
            tr_components = pd.concat(
                [
                    high - low,
                    (high - prev_close).abs(),
                    (low - prev_close).abs(),
                ],
                axis=1,
            )
            true_range = tr_components.max(axis=1)
            atr = true_range.rolling(14).mean()
            features["atr_norm"] = atr / (close + 1e-8)
        if {"bid_price", "ask_price"}.issubset(features.columns):
            ask = features["ask_price"].astype(float)
            bid = features["bid_price"].astype(float)
            spread = ask - bid
            mid = (ask + bid) / 2
            features["spread"] = spread
            features["relative_spread"] = spread / (mid + 1e-8)
            features["mid"] = mid
        if {"bid_volume", "ask_volume"}.issubset(features.columns):
            bid_vol = features["bid_volume"].astype(float)
            ask_vol = features["ask_volume"].astype(float)
            features["order_imbalance"] = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)
        return features

    def _apply_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.index, pd.DatetimeIndex):
            minute_of_day = df.index.hour * 60 + df.index.minute
            minutes_per_day = 24 * 60
            df["sin_time"] = np.sin(2 * math.pi * minute_of_day / minutes_per_day)
            df["cos_time"] = np.cos(2 * math.pi * minute_of_day / minutes_per_day)
            day_of_week = df.index.dayofweek
            df["sin_week"] = np.sin(2 * math.pi * day_of_week / 7.0)
            df["cos_week"] = np.cos(2 * math.pi * day_of_week / 7.0)
        return df

    def _transform_channels(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        processed = df.replace([np.inf, -np.inf], np.nan)
        processed = processed.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
        scaled_data = {}
        for column in processed.columns:
            values = processed[column].values.reshape(-1, 1)
            if fit or column not in self.channel_scalers:
                if not fit and column not in self.channel_scalers:
                    warnings.warn(
                        f"Column '{column}' was not seen during training; fitting a new scaler on inference data.",
                        RuntimeWarning,
                    )
                scaler = RobustScaler() if self.config.channel_scaler == "robust" else StandardScaler()
                scaler.fit(values)
                self.channel_scalers[column] = scaler
            scaler = self.channel_scalers[column]
            scaled = scaler.transform(values).reshape(-1)
            scaled_data[column] = scaled
        return pd.DataFrame(scaled_data, index=processed.index)

    def _build_sequence_dataset(
        self,
        scaled_df: pd.DataFrame,
        regression_targets: Optional[pd.DataFrame],
        classification_targets: Optional[pd.DataFrame],
        require_targets: bool,
    ) -> PatchTSTSequenceDataset:
        values = scaled_df.values
        num_rows, num_features = values.shape
        if num_rows < self.config.lookback:
            raise ValueError("Not enough rows to build sequences for the specified lookback")
        reg_array = None
        cls_array = None
        if regression_targets is not None:
            reg_array = regression_targets[self.horizons].reindex(scaled_df.index).values
        if classification_targets is not None:
            cls_array = classification_targets[self.horizons].reindex(scaled_df.index).values
        windows: List[np.ndarray] = []
        contexts: List[np.ndarray] = []
        indices: List[pd.Timestamp] = []
        reg_samples: List[np.ndarray] = [] if reg_array is not None else []
        cls_samples: List[np.ndarray] = [] if cls_array is not None else []
        max_h = max(self.horizons) if self.horizons else 1
        last_index = num_rows - max_h if require_targets else num_rows
        start_idx = self.config.lookback - 1
        if last_index <= start_idx:
            raise ValueError("Not enough samples after accounting for horizon embargo")
        for target_idx in range(start_idx, last_index):
            window = values[target_idx - self.config.lookback + 1 : target_idx + 1]
            if np.isnan(window).any():
                continue
            reg_row = reg_array[target_idx] if reg_array is not None else None
            cls_row = cls_array[target_idx] if cls_array is not None else None
            if reg_row is not None and (np.isnan(reg_row).any() or np.isinf(reg_row).any()):
                continue
            if cls_row is not None and (np.isnan(cls_row).any() or np.isinf(cls_row).any()):
                continue
            context = scaled_df.iloc[target_idx][self.context_columns].values
            windows.append(window)
            contexts.append(context)
            indices.append(scaled_df.index[target_idx])
            if reg_array is not None:
                reg_samples.append(reg_row)
            if cls_array is not None:
                cls_samples.append(cls_row)
        if not windows:
            raise ValueError("No valid sequences were generated; check data quality and NaNs")
        sequences = np.stack(windows)
        handcrafted = np.stack(contexts)
        regression_targets_arr = np.stack(reg_samples) if reg_samples else None
        classification_targets_arr = np.stack(cls_samples) if cls_samples else None
        return PatchTSTSequenceDataset(
            sequences=sequences,
            handcrafted_features=handcrafted,
            indices=indices,
            regression_targets=regression_targets_arr,
            classification_targets=classification_targets_arr,
        )

    def _prepare_dataloaders(
        self, dataset: PatchTSTSequenceDataset
    ) -> Tuple[DataLoader, Optional[DataLoader], np.ndarray, np.ndarray]:
        num_samples = len(dataset)
        indices = np.arange(num_samples)
        val_count = max(1, int(num_samples * (1 - self.config.train_val_split)))
        train_end = max(num_samples - val_count - self.config.embargo, 1)
        val_start = min(train_end + self.config.embargo, num_samples)
        train_indices = indices[:train_end]
        val_indices = indices[val_start:]
        if val_indices.size == 0:
            val_indices = indices[train_end:]
        train_subset = Subset(dataset, train_indices.tolist())
        val_subset = Subset(dataset, val_indices.tolist()) if val_indices.size > 0 else None
        pin_memory = self.config.device.startswith("cuda")
        train_loader = DataLoader(
            train_subset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.config.num_workers,
            pin_memory=pin_memory,
        )
        val_loader = (
            DataLoader(
                val_subset,
                batch_size=self.config.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.config.num_workers,
                pin_memory=pin_memory,
            )
            if val_subset is not None and len(val_subset) > 0
            else None
        )
        return train_loader, val_loader, train_indices, val_indices

    def _initialise_patch_model(self, input_dim: int) -> None:
        num_patches = 1 + (self.config.lookback - self.config.patch_size) // self.config.patch_stride
        self.patch_model = PatchTSTModel(input_dim, num_patches, self.config, len(self.horizons))
        self.patch_model.to(self.config.device)

    def _train_patch_model(
        self,
        dataset: PatchTSTSequenceDataset,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        train_indices: np.ndarray,
    ) -> None:
        if self.patch_model is None:
            raise RuntimeError("PatchTST model has not been initialised")
        optimizer = torch.optim.AdamW(
            self.patch_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scaler = GradScaler(enabled=self.config.use_amp and self.config.device.startswith("cuda"))
        if dataset.has_regression:
            train_targets = dataset.regression_targets[train_indices].numpy()
            if self.config.huber_delta is not None:
                huber_beta = self.config.huber_delta
            else:
                median_abs = np.median(np.abs(train_targets))
                if not np.isfinite(median_abs) or median_abs == 0:
                    median_abs = 1.0
                huber_beta = self.config.huber_delta_scale * median_abs
        else:
            huber_beta = 1.0
        best_state = copy.deepcopy(self.patch_model.state_dict())
        best_loss = float("inf")
        patience_counter = 0
        for epoch in range(self.config.max_epochs):
            self.patch_model.train()
            train_loss_accum = 0.0
            train_count = 0
            for batch in train_loader:
                inputs = batch["inputs"].to(self.config.device)
                targets_reg = batch.get("regression")
                targets_cls = batch.get("classification")
                if targets_reg is not None:
                    targets_reg = targets_reg.to(self.config.device)
                if targets_cls is not None:
                    targets_cls = targets_cls.to(self.config.device)
                optimizer.zero_grad()
                with autocast(enabled=self.config.use_amp and self.config.device.startswith("cuda")):
                    outputs = self.patch_model(inputs)
                    loss = torch.zeros((), device=self.config.device)
                    if targets_reg is not None and "regression" in outputs:
                        reg_loss = F.smooth_l1_loss(outputs["regression"], targets_reg, beta=huber_beta)
                        loss = loss + reg_loss
                    if targets_cls is not None and "classification" in outputs:
                        cls_loss = F.binary_cross_entropy_with_logits(outputs["classification"], targets_cls)
                        loss = loss + cls_loss
                    if (
                        targets_reg is not None
                        and self.config.use_quantile_heads
                        and "quantiles" in outputs
                    ):
                        quant_loss = _pinball_loss(outputs["quantiles"], targets_reg, self.config.quantile_levels)
                        loss = loss + quant_loss
                if torch.isnan(loss):
                    raise RuntimeError("NaN encountered in PatchTST loss")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss_accum += loss.item() * inputs.size(0)
                train_count += inputs.size(0)
            train_loss = train_loss_accum / max(train_count, 1)
            if val_loader is not None:
                self.patch_model.eval()
                val_loss_accum = 0.0
                val_count = 0
                with torch.no_grad():
                    for batch in val_loader:
                        inputs = batch["inputs"].to(self.config.device)
                        targets_reg = batch.get("regression")
                        targets_cls = batch.get("classification")
                        if targets_reg is not None:
                            targets_reg = targets_reg.to(self.config.device)
                        if targets_cls is not None:
                            targets_cls = targets_cls.to(self.config.device)
                        outputs = self.patch_model(inputs)
                        loss = torch.zeros((), device=self.config.device)
                        if targets_reg is not None and "regression" in outputs:
                            loss = loss + F.smooth_l1_loss(outputs["regression"], targets_reg, beta=huber_beta)
                        if targets_cls is not None and "classification" in outputs:
                            loss = loss + F.binary_cross_entropy_with_logits(outputs["classification"], targets_cls)
                        if (
                            targets_reg is not None
                            and self.config.use_quantile_heads
                            and "quantiles" in outputs
                        ):
                            loss = loss + _pinball_loss(outputs["quantiles"], targets_reg, self.config.quantile_levels)
                        val_loss_accum += loss.item() * inputs.size(0)
                        val_count += inputs.size(0)
                val_loss = val_loss_accum / max(val_count, 1)
            else:
                val_loss = train_loss
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_state = copy.deepcopy(self.patch_model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    break
            logger.debug(
                "PatchTST epoch %d/%d - train_loss=%.5f val_loss=%.5f",
                epoch + 1,
                self.config.max_epochs,
                train_loss,
                val_loss,
            )
        self.patch_model.load_state_dict(best_state)
        self.patch_state = copy.deepcopy(best_state)
        self.patch_model.eval()

    def _extract_patch_outputs(self, dataset: PatchTSTSequenceDataset) -> Dict[str, pd.DataFrame]:
        if self.patch_model is None:
            raise RuntimeError("PatchTST model is not initialised")
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.device.startswith("cuda"),
        )
        forecasts_list: List[torch.Tensor] = []
        class_list: List[torch.Tensor] = []
        quantile_list: List[torch.Tensor] = []
        embedding_list: List[torch.Tensor] = []
        context_list: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch["inputs"].to(self.config.device)
                outputs = self.patch_model(inputs)
                if "regression" in outputs:
                    forecasts_list.append(outputs["regression"].detach().cpu())
                if "classification" in outputs:
                    class_list.append(torch.sigmoid(outputs["classification"].detach().cpu()))
                if "quantiles" in outputs:
                    quantile_list.append(outputs["quantiles"].detach().cpu())
                embedding_list.append(outputs["embedding"].detach().cpu())
                context_list.append(batch["context"])
        index = pd.Index(dataset.indices, name="timestamp")
        results: Dict[str, pd.DataFrame] = {}
        if forecasts_list:
            forecasts = torch.cat(forecasts_list, dim=0).numpy()
            forecast_cols = [f"patch_forecast_h{h}" for h in self.horizons]
            results["patch_forecasts"] = pd.DataFrame(forecasts, columns=forecast_cols, index=index)
        if class_list:
            probs = torch.cat(class_list, dim=0).numpy()
            class_cols = [f"patch_direction_prob_h{h}" for h in self.horizons]
            results["patch_direction"] = pd.DataFrame(probs, columns=class_cols, index=index)
        if quantile_list:
            quantiles = torch.cat(quantile_list, dim=0).numpy()
            quantile_cols: List[str] = []
            for horizon in self.horizons:
                for tau in self.config.quantile_levels:
                    quantile_cols.append(f"patch_quantile_{tau:.2f}_h{horizon}")
            reshaped = quantiles.reshape(len(index), -1)
            results["patch_quantiles"] = pd.DataFrame(reshaped, columns=quantile_cols, index=index)
        embeddings = torch.cat(embedding_list, dim=0).numpy()
        emb_cols = [f"patch_emb_{i}" for i in range(embeddings.shape[1])]
        results["patch_embeddings"] = pd.DataFrame(embeddings, columns=emb_cols, index=index)
        contexts = torch.cat(context_list, dim=0).numpy()
        results["context"] = pd.DataFrame(contexts, columns=self.handcrafted_feature_names, index=index)
        if "patch_quantiles" in results and len(self.config.quantile_levels) >= 2:
            low_tau = self.config.quantile_levels[0]
            high_tau = self.config.quantile_levels[-1]
            quantile_df = results["patch_quantiles"]
            uncertainty = {}
            for horizon in self.horizons:
                low_col = f"patch_quantile_{low_tau:.2f}_h{horizon}"
                high_col = f"patch_quantile_{high_tau:.2f}_h{horizon}"
                if low_col in quantile_df.columns and high_col in quantile_df.columns:
                    uncertainty[f"patch_iqr_h{horizon}"] = quantile_df[high_col] - quantile_df[low_col]
            if uncertainty:
                results["patch_uncertainty"] = pd.DataFrame(uncertainty, index=index)
        return results

    def _assemble_tree_features(self, patch_outputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for key in ("context", "patch_forecasts", "patch_direction", "patch_quantiles", "patch_uncertainty", "patch_embeddings"):
            df = patch_outputs.get(key)
            if df is not None:
                frames.append(df)
        if not frames:
            raise RuntimeError("No patch outputs available to assemble tree features")
        features_df = pd.concat(frames, axis=1)
        features_df = features_df.fillna(0.0)
        return features_df

    def _scale_tree_features(self, features: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if self.tree_feature_scaler is None or fit:
            scaler = StandardScaler() if self.config.feature_scaler == "standard" else RobustScaler()
            scaled = scaler.fit_transform(features.values)
            self.tree_feature_scaler = scaler
            self.tree_feature_columns = list(features.columns)
        else:
            missing = [col for col in self.tree_feature_columns if col not in features.columns]
            if missing:
                for col in missing:
                    features[col] = 0.0
            features = features[self.tree_feature_columns]
            scaled = self.tree_feature_scaler.transform(features.values)
        return pd.DataFrame(scaled, columns=self.tree_feature_columns, index=features.index)

    def _fit_tree_models(self, features: pd.DataFrame, dataset: PatchTSTSequenceDataset) -> None:
        if features.empty:
            raise ValueError("No features available for tree training")
        if dataset.has_regression:
            reg_targets = dataset.regression_targets.numpy()
            reg_cols = [f"return_h{h}" for h in self.horizons]
            reg_df = pd.DataFrame(reg_targets, columns=reg_cols, index=features.index)
        else:
            reg_df = None
        if dataset.has_classification:
            cls_targets = dataset.classification_targets.numpy()
            cls_cols = [f"direction_prob_h{h}" for h in self.horizons]
            cls_df = pd.DataFrame(cls_targets, columns=cls_cols, index=features.index)
        else:
            cls_df = None
        values = features.values
        if reg_df is not None:
            for idx, horizon in enumerate(self.horizons):
                model = clone(self.base_model)
                model.fit(values, reg_df.iloc[:, idx].values)
                self.tree_models_reg[horizon] = model
        if cls_df is not None:
            template = self.classification_model or self.base_model
            for idx, horizon in enumerate(self.horizons):
                model = clone(template)
                model.fit(values, cls_df.iloc[:, idx].values)
                self.tree_models_cls[horizon] = model
        self.last_tree_features_ = features

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[pd.Series] = None) -> "TreePatchTSTWrapper":
        del y  # Targets are derived from market data directly
        raw_df = self._ensure_dataframe(X)
        augmented_df = self._apply_calendar_features(self._compute_derived_features(raw_df))
        scaled_df = self._transform_channels(augmented_df, fit=True)
        self.context_columns = list(scaled_df.columns)
        self.handcrafted_feature_names = [f"context_{col}" for col in self.context_columns]
        regression_targets, classification_targets = self._compute_targets(augmented_df)
        dataset = self._build_sequence_dataset(
            scaled_df.reindex(columns=self.context_columns),
            regression_targets,
            classification_targets if self.config.task_type in {"classification", "both"} else None,
            require_targets=True,
        )
        if len(dataset) < self.config.min_samples:
            warnings.warn(
                f"Training dataset has only {len(dataset)} samples (< {self.config.min_samples}); model quality may suffer.",
                RuntimeWarning,
            )
        self._initialise_patch_model(input_dim=scaled_df.shape[1])
        train_loader, val_loader, train_indices, _ = self._prepare_dataloaders(dataset)
        logger.info(
            "Training PatchTST encoder on %d samples (train=%d, val=%d)",
            len(dataset),
            len(train_indices),
            len(dataset) - len(train_indices),
        )
        self._train_patch_model(dataset, train_loader, val_loader, train_indices)
        patch_outputs = self._extract_patch_outputs(dataset)
        tree_features = self._assemble_tree_features(patch_outputs)
        scaled_features = self._scale_tree_features(tree_features, fit=True)
        self._fit_tree_models(scaled_features, dataset)
        self.last_patch_outputs_ = patch_outputs
        self.training_metadata = {
            "num_samples": len(dataset),
            "tree_feature_columns": self.tree_feature_columns,
            "horizons": self.horizons,
        }
        self.is_fitted = True
        logger.info("PatchTST + tree pipeline fitted with %d horizons", len(self.horizons))
        return self

    def _transform_for_inference(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        if not self.is_fitted:
            raise RuntimeError("The wrapper must be fitted before calling predict")
        raw_df = self._ensure_dataframe(X)
        augmented_df = self._apply_calendar_features(self._compute_derived_features(raw_df))
        scaled_df = self._transform_channels(augmented_df, fit=False)
        scaled_df = scaled_df.reindex(columns=self.context_columns, fill_value=0.0)
        dataset = self._build_sequence_dataset(
            scaled_df,
            regression_targets=None,
            classification_targets=None,
            require_targets=False,
        )
        patch_outputs = self._extract_patch_outputs(dataset)
        tree_features = self._assemble_tree_features(patch_outputs)
        scaled_features = self._scale_tree_features(tree_features, fit=False)
        return scaled_features, patch_outputs

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        features, patch_outputs = self._transform_for_inference(X)
        predictions: Dict[str, np.ndarray] = {}
        for horizon, model in self.tree_models_reg.items():
            preds = model.predict(features.values)
            predictions[f"return_h{horizon}"] = preds
        result = pd.DataFrame(predictions, index=features.index) if predictions else pd.DataFrame(index=features.index)
        self.last_patch_outputs_ = patch_outputs
        self.last_tree_features_ = features
        return result

    def predict_direction_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if not self.tree_models_cls:
            raise RuntimeError("No classification tree models are available")
        features, patch_outputs = self._transform_for_inference(X)
        predictions: Dict[str, np.ndarray] = {}
        for horizon, model in self.tree_models_cls.items():
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features.values)
                if proba.ndim > 1 and proba.shape[1] > 1:
                    proba = proba[:, -1]
                else:
                    proba = proba.ravel()
            else:
                proba = model.predict(features.values)
            predictions[f"direction_prob_h{horizon}"] = proba
        result = pd.DataFrame(predictions, index=features.index)
        self.last_patch_outputs_ = patch_outputs
        self.last_tree_features_ = features
        return result

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        features, patch_outputs = self._transform_for_inference(X)
        self.last_patch_outputs_ = patch_outputs
        self.last_tree_features_ = features
        return features

    def get_patch_embeddings(self) -> Optional[pd.DataFrame]:
        return self.last_patch_outputs_.get("patch_embeddings") if self.last_patch_outputs_ else None
