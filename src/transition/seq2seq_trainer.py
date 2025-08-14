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

# Hint to speed CPU matmul on Apple Accelerate
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def _to_tensor(x: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype)


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    try:
        n, m = len(a), len(b)
        dtw = np.full((n + 1, m + 1), np.inf, dtype=float)
        dtw[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(a[i - 1] - b[j - 1])
                dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
        return float(dtw[n, m] / (n + m))
    except Exception:
        return float("nan")


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
        # time to pt
        y_ttpt = int(s.get("Y_time_to_pt", -1))
        return {
            "hmm_ids": _to_tensor(hmm_ids, torch.long),
            "x_num": _to_tensor(X_num, torch.float32),
            "y_ret": _to_tensor(y_ret, torch.float32),
            "y_hmm": _to_tensor(y_hmm, torch.long),
            "y_path": _to_tensor(np.array(y_path), torch.long),
            "y_ttpt": _to_tensor(np.array(y_ttpt), torch.float32),
        }


class SmallTransformer(pl.LightningModule if pl else nn.Module):
    def __init__(self, hmm_vocab: int, num_features: int, post_len: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1, lr: float = 1e-3, path_class_weights: dict[str,float] | None = None, focal_gamma: float = 0.0):
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
        # class weights
        if path_class_weights:
            w_map = {"continuation":0, "reversal":1, "end_of_trend":2, "beginning_of_trend":3}
            w = torch.ones(4, dtype=torch.float32)
            for k,v in path_class_weights.items():
                if k in w_map:
                    w[w_map[k]] = float(v)
            self.ce = nn.CrossEntropyLoss(weight=w)
        else:
            self.ce = nn.CrossEntropyLoss()
        self.focal_gamma = float(focal_gamma)

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
        # Use last post_len timesteps
        z = h[:, -self.post_len :, :]
        pred_ret = self.dec_ret(z).squeeze(-1)  # [B, post]
        pred_hmm = self.dec_hmm(z)  # [B, post, vocab]
        pred_path = self.cls_path(cls)  # [B, 4]
        # Losses
        loss_ret = self.mse(pred_ret, y_ret)
        loss_hmm = self.ce(pred_hmm.reshape(-1, pred_hmm.size(-1)), y_hmm.reshape(-1))
        # optional focal loss on path head
        if self.focal_gamma > 0.0:
            ce = self.ce(pred_path, y_path)
            with torch.no_grad():
                p = torch.softmax(pred_path, dim=-1).gather(1, y_path.view(-1,1)).clamp_min(1e-6).squeeze()
            loss_path = ((1 - p) ** self.focal_gamma) * ce
        else:
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
        if self.focal_gamma > 0.0:
            ce = self.ce(pred_path, y_path)
            with torch.no_grad():
                p = torch.softmax(pred_path, dim=-1).gather(1, y_path.view(-1,1)).clamp_min(1e-6).squeeze()
            loss_path = ((1 - p) ** self.focal_gamma) * ce
        else:
            loss_path = self.ce(pred_path, y_path)
        loss = loss_ret * 1.0 + loss_hmm * 0.7 + loss_path * 0.5
        # Metrics: state accuracy, return MSE, DTW (avg over batch subset)
        with torch.no_grad():
            state_pred = pred_hmm.argmax(-1)
            state_acc = (state_pred == y_hmm).float().mean()
            mse = nn.functional.mse_loss(pred_ret, y_ret)
        self.log_dict({"val_loss": loss, "val_state_acc": state_acc, "val_mse": mse}, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}


class SmallTCN(SmallTransformer):
    def __init__(self, hmm_vocab: int, num_features: int, post_len: int, d_model: int = 128, layers: int = 4, dropout: float = 0.1, lr: float = 1e-3, path_class_weights: dict[str,float] | None = None, focal_gamma: float = 0.0):
        super().__init__(hmm_vocab, num_features, post_len, d_model=d_model, nhead=1, num_layers=1, dropout=dropout, lr=lr, path_class_weights=path_class_weights, focal_gamma=focal_gamma)
        # Replace encoder with TCN
        blocks: list[nn.Module] = []
        for i in range(layers):
            dilation = 2 ** i
            blocks.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=dilation, dilation=dilation),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(d_model, d_model, kernel_size=1),
            ))
        self.tcn = nn.ModuleList(blocks)

    def forward(self, hmm_ids: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        x = self.hmm_emb(hmm_ids) + self.num_proj(x_num)  # [B,L,D]
        y = x.transpose(1, 2)  # [B,D,L]
        for block in self.tcn:
            y = y + block(y)
        h = y.transpose(1, 2)  # [B,L,D]
        cls = h.mean(dim=1)
        return h, cls


def build_dataloaders(samples: List[dict[str, Any]], numeric_dim: int, label_index: List[str], post_len: int, batch_size: int = 128, seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    # Split by time order: 80/20
    n = len(samples)
    cut = int(n * 0.8)
    train_ds = TransitionSeqDataset(samples[:cut], numeric_dim, label_index)
    val_ds = TransitionSeqDataset(samples[cut:], numeric_dim, label_index)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader


def evaluate_samples(model: SmallTransformer, dataloader: DataLoader, pt_mult: float, device: str | None = None) -> dict[str, float]:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.eval().to(device)
    mse_list: list[float] = []
    dtw_list: list[float] = []
    acc_list: list[float] = []
    ttpt_mae_list: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            hmm_ids = batch["hmm_ids"].to(device)
            x_num = batch["x_num"].to(device)
            y_ret = batch["y_ret"].to(device)
            y_hmm = batch["y_hmm"].to(device)
            y_ttpt = batch.get("y_ttpt", torch.full((y_ret.size(0),), -1.0)).to(device)
            h, cls = model(hmm_ids, x_num)
            z = h[:, -model.post_len :, :]
            pred_ret = model.dec_ret(z).squeeze(-1)
            pred_hmm = model.dec_hmm(z)
            # Metrics
            mse_list.append(nn.functional.mse_loss(pred_ret, y_ret).item())
            # DTW on first 64 samples to keep it fast
            pr = pred_ret.detach().cpu().numpy()
            yr = y_ret.detach().cpu().numpy()
            for i in range(min(len(pr), 64)):
                dtw_list.append(_dtw_distance(pr[i], yr[i]))
            acc_list.append((pred_hmm.argmax(-1) == y_hmm).float().mean().item())
            # ttpt prediction from returns path
            ttpt_pred = []
            for seq in pr:
                ttp = -1
                for t, r in enumerate(seq, start=1):
                    if r >= pt_mult:
                        ttp = t
                        break
                ttpt_pred.append(ttp)
            mask = (y_ttpt >= 0)
            if mask.any():
                mae = torch.mean(torch.abs(torch.tensor(ttpt_pred, device=device, dtype=torch.float32) - y_ttpt)[mask]).item()
                ttpt_mae_list.append(mae)
    return {
        "mse": float(np.nanmean(mse_list)) if mse_list else float("nan"),
        "dtw": float(np.nanmean(dtw_list)) if dtw_list else float("nan"),
        "state_acc": float(np.nanmean(acc_list)) if acc_list else float("nan"),
        "ttpt_mae": float(np.nanmean(ttpt_mae_list)) if ttpt_mae_list else float("nan"),
    }


def train_seq2seq(samples: List[dict[str, Any]], label_index: List[str], numeric_feature_names: List[str], post_window: int, hmm_vocab: int = 5, d_model: int = 128, nhead: int = 4, num_layers: int = 2, max_epochs: int = 25, lr: float = 1e-3, path_class_weights: dict[str,float] | None = None, focal_gamma: float = 0.0, precision: str = "32", artifact_dir_models: str | None = None, cv_folds: int = 1, pt_mult: float = 0.002, model_type: str = "transformer") -> dict[str, Any]:
    logger = system_logger.getChild("TransitionSeq2SeqTrainer")
    if pl is None:
        logger.warning("PyTorch Lightning not available; skip seq2seq training.")
        return {"trained": False}
    numeric_dim = len(numeric_feature_names)

    def _make_model() -> SmallTransformer:
        if model_type.lower() == "tcn":
            return SmallTCN(hmm_vocab=hmm_vocab, num_features=numeric_dim, post_len=post_window, d_model=d_model, layers=max(2, num_layers), lr=lr, path_class_weights=path_class_weights, focal_gamma=focal_gamma)
        return SmallTransformer(hmm_vocab=hmm_vocab, num_features=numeric_dim, post_len=post_window, d_model=d_model, nhead=nhead, num_layers=num_layers, lr=lr, path_class_weights=path_class_weights, focal_gamma=focal_gamma)

    def _train_one(train_s: List[dict[str, Any]]) -> tuple[SmallTransformer, dict]:
        train_loader, val_loader = build_dataloaders(train_s, numeric_dim, label_index, post_window)
        model = _make_model()
        callbacks = []
        if artifact_dir_models:
            try:
                import os
                os.makedirs(artifact_dir_models, exist_ok=True)
                from pytorch_lightning.callbacks import ModelCheckpoint
                callbacks.append(ModelCheckpoint(dirpath=artifact_dir_models, save_top_k=1, monitor="val_loss", mode="min"))
            except Exception:
                pass
        trainer = pl.Trainer(max_epochs=max_epochs, accelerator="auto", devices="auto", precision=precision, gradient_clip_val=1.0, log_every_n_steps=50, callbacks=callbacks)
        trainer.fit(model, train_loader, val_loader)
        try:
            if artifact_dir_models:
                import os
                os.makedirs(artifact_dir_models, exist_ok=True)
                trainer.save_checkpoint(os.path.join(artifact_dir_models, "last.ckpt"))
        except Exception:
            pass
        metrics = evaluate_samples(model, val_loader, pt_mult=pt_mult)
        return model, metrics

    if cv_folds and cv_folds > 1:
        n = len(samples)
        fold_size = max(1, n // cv_folds)
        all_metrics: list[dict] = []
        best_idx = 0
        best_mse = float("inf")
        best_model: SmallTransformer | None = None
        for k in range(cv_folds):
            end = n - (cv_folds - 1 - k) * fold_size
            start = max(0, end - fold_size)
            fold_samples = samples[:end]
            model, m = _train_one(fold_samples)
            all_metrics.append(m)
            if m.get("mse", float("inf")) < best_mse:
                best_mse = m.get("mse", float("inf"))
                best_idx = k
                best_model = model
        return {"trained": True, "cv_metrics": all_metrics, "best_fold": best_idx, "best_mse": best_mse}
    else:
        model, metrics = _train_one(samples)
        return {"trained": True, "metrics": metrics}