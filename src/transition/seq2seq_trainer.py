# src/transition/seq2seq_trainer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import pytorch_lightning as pl
except Exception:  # pragma: no cover
    pl = None  # type: ignore

from src.utils.logger import system_logger


def _to_tensor(x: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype)


class TransitionSeqDataset(Dataset):
    def __init__(self, samples: List[dict[str, Any]], numeric_dim: int, label_index: List[str]):
        self.samples = samples
        self.numeric_dim = numeric_dim
        self.label_index = label_index

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        # Inputs
        X_states_df: pd.DataFrame = s["X_pre_states"]
        hmm_ids = X_states_df["hmm_state_id"].to_numpy(dtype=np.int64)
        # numeric
        X_num = s.get("X_pre_numeric", np.zeros((len(hmm_ids), 0), dtype=float))
        # Targets
        y_ret = s["Y_post_returns"].astype(np.float32)
        y_states_df: pd.DataFrame = s["Y_post_states"]
        y_hmm = y_states_df["hmm_state_id"].to_numpy(dtype=np.int64)
        # Class at t0 (path_class) if present
        path_map = {"continuation":0, "reversal":1, "end_of_trend":2, "beginning_of_trend":3}
        y_path = path_map.get(str(s.get("path_class","end_of_trend")), 2)
        return {
            "hmm_ids": _to_tensor(hmm_ids, torch.long),
            "x_num": _to_tensor(X_num, torch.float32),
            "y_ret": _to_tensor(y_ret, torch.float32),
            "y_hmm": _to_tensor(y_hmm, torch.long),
            "y_path": _to_tensor(np.array(y_path), torch.long),
        }


class SmallTransformer(pl.LightningModule if pl else nn.Module):
    def __init__(self, hmm_vocab: int, num_features: int, post_len: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1, lr: float = 1e-3):
        if pl:
            super().__init__()
        else:
            super().__init__()
        self.save_hyperparameters = getattr(self, "save_hyperparameters", lambda *args, **kwargs: None)
        self.save_hyperparameters()
        self.hmm_emb = nn.Embedding(hmm_vocab, d_model)
        self.num_proj = nn.Linear(num_features, d_model)
        self.enc_ln = nn.LayerNorm(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        # Decoder heads
        self.dec_ret = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.dec_hmm = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, hmm_vocab))
        self.cls_path = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 4))
        self.post_len = post_len
        self.lr = lr
        self.mse = nn.SmoothL1Loss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, hmm_ids: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        # hmm_ids: [B, L_pre] ; x_num: [B, L_pre, F]
        x = self.hmm_emb(hmm_ids) + self.num_proj(x_num)
        x = self.enc_ln(x)
        h = self.encoder(x)  # [B, L_pre, d]
        # CLS token: mean pool last K
        cls = h.mean(dim=1)
        return h, cls

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        hmm_ids = batch["hmm_ids"]
        x_num = batch["x_num"]
        y_ret = batch["y_ret"]
        y_hmm = batch["y_hmm"]
        y_path = batch["y_path"]
        h, cls = self(hmm_ids, x_num)
        # Use last post_len timesteps from the end of encoder sequence as a proxy for decoder targets (simple teacher forcing omission)
        # For strict seq2seq, introduce a decoder; here we keep it compact for speed.
        z = h[:, -self.post_len :, :]
        pred_ret = self.dec_ret(z).squeeze(-1)  # [B, post]
        pred_hmm = self.dec_hmm(z)  # [B, post, vocab]
        pred_path = self.cls_path(cls)  # [B, 4]
        # Losses
        loss_ret = self.mse(pred_ret, y_ret)
        loss_hmm = self.ce(pred_hmm.reshape(-1, pred_hmm.size(-1)), y_hmm.reshape(-1))
        loss_path = self.ce(pred_path, y_path)
        loss = loss_ret * 1.0 + loss_hmm * 0.7 + loss_path * 0.5
        self.log_dict({"train_loss": loss, "loss_ret": loss_ret, "loss_hmm": loss_hmm, "loss_path": loss_path}, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        hmm_ids = batch["hmm_ids"]
        x_num = batch["x_num"]
        y_ret = batch["y_ret"]
        y_hmm = batch["y_hmm"]
        y_path = batch["y_path"]
        h, cls = self(hmm_ids, x_num)
        z = h[:, -self.post_len :, :]
        pred_ret = self.dec_ret(z).squeeze(-1)
        pred_hmm = self.dec_hmm(z)
        pred_path = self.cls_path(cls)
        loss_ret = self.mse(pred_ret, y_ret)
        loss_hmm = self.ce(pred_hmm.reshape(-1, pred_hmm.size(-1)), y_hmm.reshape(-1))
        loss_path = self.ce(pred_path, y_path)
        loss = loss_ret * 1.0 + loss_hmm * 0.7 + loss_path * 0.5
        self.log_dict({"val_loss": loss, "val_loss_ret": loss_ret, "val_loss_hmm": loss_hmm, "val_loss_path": loss_path}, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}


def build_dataloaders(samples: List[dict[str, Any]], numeric_dim: int, label_index: List[str], post_len: int, batch_size: int = 128, seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    # Split by time order: 80/20
    n = len(samples)
    cut = int(n * 0.8)
    train_ds = TransitionSeqDataset(samples[:cut], numeric_dim, label_index)
    val_ds = TransitionSeqDataset(samples[cut:], numeric_dim, label_index)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader


def train_seq2seq(samples: List[dict[str, Any]], label_index: List[str], numeric_feature_names: List[str], post_window: int, hmm_vocab: int = 5, d_model: int = 128, nhead: int = 4, num_layers: int = 2, max_epochs: int = 25, lr: float = 1e-3) -> dict[str, Any]:
    logger = system_logger.getChild("TransitionSeq2SeqTrainer")
    if pl is None:
        logger.warning("PyTorch Lightning not available; skip seq2seq training.")
        return {"trained": False}
    numeric_dim = len(numeric_feature_names)
    train_loader, val_loader = build_dataloaders(samples, numeric_dim, label_index, post_window)
    model = SmallTransformer(hmm_vocab=hmm_vocab, num_features=numeric_dim, post_len=post_window, d_model=d_model, nhead=nhead, num_layers=num_layers, lr=lr)
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="auto", devices="auto", gradient_clip_val=1.0, log_every_n_steps=50)
    trainer.fit(model, train_loader, val_loader)
    return {"trained": True}