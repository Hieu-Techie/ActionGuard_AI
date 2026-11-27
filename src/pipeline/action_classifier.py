"""
action_classifier.py — Stage 2: Action model loading and sequence classification.

Responsibilities:
  • Define ActionLSTM (Baseline), ActionSTGCNv1 (Minimal GCN Baseline),
    and ActionSTGCN (v2 Proposed Model supporting incremental ablation flags:
    v2.1 Spatial Partitioning + Multi-Scale TCN,
    v2.2 + Temporal Attention & Directional Pyramid Pooling,
    v2.3 + Adaptive Adjacency Matrix,
    v2.4 + Joint/Bone/Motion Kinematic Features).
  • Auto-detect checkpoint version/variant in `load_action_model()` so v1 and
    any v2.1–v2.4 checkpoint loads seamlessly with `weights_only=True`.
  • Run inference on a filled SequenceBuffer and return ActionPrediction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
import numpy as np

import sys
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from utils.config import (
    NUM_JOINTS, NUM_COORDS, WINDOW_SIZE, NUM_CLASSES, CLASS_NAMES,
    LSTM_MODEL_PATH, STGCN_MODEL_PATH, ACTIVE_ACTION_MODEL,
    STGCN_ADJACENCY_MATRIX_PATH,
)
from pipeline.sequence_buffer import SequenceBuffer

log = logging.getLogger(__name__)

ModelType = Literal["lstm", "stgcn"]

# Anatomical parent joint index for each COCO-17 joint (used for Bone vectors in v2.4)
COCO_PARENTS = [0, 0, 0, 1, 2, 0, 0, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14]


# ════════════════════════════════════════════════════════════════════════════
# Output type
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ActionPrediction:
    """
    Result of classify_sequence().

    Attributes
    ----------
    label          : str   — predicted class name from CLASS_NAMES
    label_idx      : int   — corresponding integer index (0-based)
    confidence     : float — softmax probability of the predicted class
    model_type     : str   — which model produced this prediction ("lstm"/"stgcn")
    pose_quality   : str   — skeleton quality tag ("full"/"partial"/"upper_only"/"low"/"none")
    fallback_used  : bool  — True when hybrid LSTM fallback contributed to probabilities
    fallback_model : str   — fallback model identifier when fallback_used is True
    object_context : str   — detected Stage-2B object tag (e.g. "cup:0.78", "no_vessel", "")
    """
    label:          str   = "Unknown"
    label_idx:      int   = -1
    confidence:     float = 0.0
    model_type:     str   = ""
    pose_quality:   str   = "full"
    fallback_used:  bool  = False
    fallback_model: str   = ""
    object_context: str   = ""


# ════════════════════════════════════════════════════════════════════════════
# LSTM Architecture (Baseline)
# ════════════════════════════════════════════════════════════════════════════

class ActionLSTM(nn.Module):
    """
    Bidirectional LSTM action classifier.

    Input shape : (batch, WINDOW_SIZE, NUM_JOINTS * NUM_COORDS)
    Output shape: (batch, NUM_CLASSES)
    """

    def __init__(
        self,
        input_size:  int = NUM_JOINTS * NUM_COORDS,
        hidden_size: int = 128,
        num_layers:  int = 2,
        num_classes: int = NUM_CLASSES,
        dropout:     float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)          # (B, T, 2*hidden)
        out = out[:, -1, :]            # last time step
        out = self.dropout(out)
        return self.fc(out)            # (B, num_classes)


# ════════════════════════════════════════════════════════════════════════════
# ST-GCN v1 Architecture (Minimal Baseline — preserved for v1 comparison)
# ════════════════════════════════════════════════════════════════════════════

def _build_adjacency(num_joints: int = NUM_JOINTS) -> torch.Tensor:
    """Build single symmetric normalized COCO-17 adjacency matrix (v1)."""
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 4),          # face
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
        (5, 11), (6, 12), (11, 12),                # torso
        (11, 13), (13, 15), (12, 14), (14, 16),    # legs
    ]
    A = np.eye(num_joints, dtype=np.float32)
    for a, b in edges:
        A[a, b] = 1.0
        A[b, a] = 1.0
    degree = A.sum(axis=1)
    d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    A_norm = d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
    return torch.from_numpy(A_norm)


class STGCNBlock(nn.Module):
    """Single-adjacency spatial-temporal graph conv block (ST-GCN v1)."""

    A: torch.Tensor

    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor, stride: int = 1) -> None:
        super().__init__()
        self.register_buffer("A", A)
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
        )
        self.residual = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.einsum("bctv,vw->bctw", x, self.A)
        y = self.gcn(y)
        y = self.tcn(y)
        return self.relu(y + self.residual(x))


class ActionSTGCNv1(nn.Module):
    """Minimal ST-GCN v1 architecture (compatible with `models/action_stgcn_v1.pth`)."""

    def __init__(
        self,
        in_channels: int = NUM_COORDS,
        num_classes: int = NUM_CLASSES,
        num_joints:  int = NUM_JOINTS,
        adjacency:   Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if adjacency is not None:
            A = adjacency
        elif STGCN_ADJACENCY_MATRIX_PATH and Path(_PROJECT_ROOT / STGCN_ADJACENCY_MATRIX_PATH).exists():
            arr = np.load(str(_PROJECT_ROOT / STGCN_ADJACENCY_MATRIX_PATH))
            A = torch.from_numpy(arr.astype(np.float32))
        else:
            A = _build_adjacency(num_joints)
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        self.layers = nn.Sequential(
            STGCNBlock(in_channels, 64,  A),
            STGCNBlock(64,          64,  A),
            STGCNBlock(64,          128, A, stride=2),
            STGCNBlock(128,         128, A),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, V = x.shape
        x_bn = x.permute(0, 1, 3, 2).contiguous().view(B, C * V, T)
        x_bn = self.data_bn(x_bn).view(B, C, V, T).permute(0, 1, 3, 2)
        y = self.layers(x_bn)
        y = self.pool(y).view(B, -1)
        y = self.dropout(y)
        return self.fc(y)


# ════════════════════════════════════════════════════════════════════════════
# ST-GCN v2 Architecture (Proposed — Multi-Partition, Multi-Scale TCN,
# Temporal Pyramid/Attention, Adaptive Graph, and Joint+Bone+Motion Streams)
# ════════════════════════════════════════════════════════════════════════════

def build_spatial_partitions(num_joints: int = NUM_JOINTS) -> torch.Tensor:
    """
    Build 3-subset spatial partition adjacency tensor A of shape (3, V, V):
      • A[0]: Self-connections (Identity I)
      • A[1]: Inward (centripetal) edges toward torso core
      • A[2]: Outward (centrifugal) edges away from torso core
    """
    inward_edges = [
        (1, 0), (2, 0), (3, 1), (4, 2),              # face -> nose
        (0, 5), (0, 6),                              # nose -> shoulders
        (7, 5), (9, 7), (8, 6), (10, 8),             # arms -> shoulders
        (5, 11), (6, 12), (5, 6), (11, 12),          # torso ring
        (13, 11), (15, 13), (14, 12), (16, 14),      # legs -> hips
    ]
    A_self = np.eye(num_joints, dtype=np.float32)
    A_in   = np.zeros((num_joints, num_joints), dtype=np.float32)
    A_out  = np.zeros((num_joints, num_joints), dtype=np.float32)

    for child, parent in inward_edges:
        if child < num_joints and parent < num_joints:
            A_in[child, parent] = 1.0
            A_out[parent, child] = 1.0

    def _norm_digraph(M: np.ndarray) -> np.ndarray:
        deg = M.sum(axis=1, keepdims=True)
        return np.where(deg > 0, M / np.maximum(deg, 1e-6), 0.0).astype(np.float32)

    return torch.from_numpy(np.stack([A_self, _norm_digraph(A_in), _norm_digraph(A_out)], axis=0))


class STGCNv2Block(nn.Module):
    """
    Spatial-Temporal Graph Convolution Block v2:
      • 3-subset partitioned spatial graph convolution + edge importance
      • Optional learnable adaptive adjacency matrix A_adap (v2.3+)
      • Multi-scale temporal convolution (branches k=3 and k=5) + residual
    """

    A: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        stride: int = 1,
        use_adaptive_adj: bool = True,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.K = int(A.shape[0])
        self.register_buffer("A", A)
        self.use_adaptive_adj = use_adaptive_adj
        self.edge_importance = nn.Parameter(torch.ones_like(A))
        if use_adaptive_adj:
            self.A_adap = nn.Parameter(torch.zeros_like(A))
        else:
            self.register_parameter("A_adap", None)

        self.gcn = nn.Conv2d(in_channels, out_channels * self.K, kernel_size=1)
        self.bn_s = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

        mid_c = out_channels // 2
        self.tcn_b1 = nn.Conv2d(out_channels, mid_c, kernel_size=(3, 1), padding=(1, 0), stride=(stride, 1))
        self.tcn_b2 = nn.Conv2d(out_channels, out_channels - mid_c, kernel_size=(5, 1), padding=(2, 0), stride=(stride, 1))
        self.bn_t = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout)

        self.residual = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, V = x.shape
        A_eff = self.A * self.edge_importance
        if self.use_adaptive_adj and self.A_adap is not None:
            A_eff = A_eff + self.A_adap

        z = self.gcn(x).view(B, self.K, -1, T, V)
        y = torch.einsum("bkctv,kvw->bctw", z, A_eff)
        y = self.act(self.bn_s(y))

        y_t = torch.cat([self.tcn_b1(y), self.tcn_b2(y)], dim=1)
        y_t = self.drop(self.bn_t(y_t))
        return self.act(y_t + self.residual(x))


class ActionSTGCN(nn.Module):
    """
    Proposed ST-GCN (v2.1–v2.4 & v3.1-A/B/C/D) for skeleton-based action recognition.

    Supports all incremental ablation stages via constructor flags:
      • v2.1: `use_temporal_pyr=False, use_adaptive_adj=False, use_kinematic_feats=False`
      • v2.2: `use_temporal_pyr=True,  use_adaptive_adj=False, use_kinematic_feats=False`
      • v2.3: `use_temporal_pyr=True,  use_adaptive_adj=True,  use_kinematic_feats=False`
      • v2.4 / v3.0-clean: `use_temporal_pyr=True, use_adaptive_adj=True, use_kinematic_feats=True`
      • v3.1-A (+ Gravity-Aware Features): `use_gravity_feats=True`
      • v3.1-B (+ Wrist-to-Head Features): `use_wrist_head_feats=True`
      • v3.1-C (+ Lower-Body Kinematics):  `use_lower_body_feats=True`
      • v3.1-D (+ Feature Fusion):         combination of `use_gravity_feats`, `use_wrist_head_feats`, `use_lower_body_feats`

    Input shape : (batch, NUM_COORDS, WINDOW_SIZE, NUM_JOINTS) — i.e. (N, 3, 16, 17)
    Output shape: (batch, NUM_CLASSES)
    """

    parents: torch.Tensor

    def __init__(
        self,
        in_channels: int = NUM_COORDS,
        num_classes: int = NUM_CLASSES,
        num_joints:  int = NUM_JOINTS,
        use_temporal_pyr:     bool = True,
        use_adaptive_adj:     bool = True,
        use_kinematic_feats:  bool = True,
        use_gravity_feats:    bool = False,
        use_wrist_head_feats: bool = False,
        use_lower_body_feats: bool = False,
    ) -> None:
        super().__init__()
        self.use_temporal_pyr     = use_temporal_pyr
        self.use_adaptive_adj     = use_adaptive_adj
        self.use_kinematic_feats  = use_kinematic_feats
        self.use_gravity_feats    = use_gravity_feats
        self.use_wrist_head_feats = use_wrist_head_feats
        self.use_lower_body_feats = use_lower_body_feats

        parent_list = COCO_PARENTS[:num_joints] if num_joints <= len(COCO_PARENTS) else list(range(num_joints))
        self.register_buffer("parents", torch.tensor(parent_list, dtype=torch.long))

        eff_in = in_channels * 3 if use_kinematic_feats else in_channels
        if use_gravity_feats:
            eff_in += 2   # [signed_acc_y, height_above_ground_anchor]
        if use_wrist_head_feats:
            eff_in += 2   # [dx_to_nose, dy_to_nose]
        if use_lower_body_feats:
            eff_in += 2   # [dy_to_ankle_mid, vy_rel_to_ankle_mid]

        A = build_spatial_partitions(num_joints)

        self.data_bn = nn.BatchNorm1d(eff_in * num_joints)
        self.b1 = STGCNv2Block(eff_in, 64,  A, stride=1, use_adaptive_adj=use_adaptive_adj, dropout=0.10)
        self.b2 = STGCNv2Block(64,     64,  A, stride=1, use_adaptive_adj=use_adaptive_adj, dropout=0.10)
        self.b3 = STGCNv2Block(64,     128, A, stride=2, use_adaptive_adj=use_adaptive_adj, dropout=0.20)
        self.b4 = STGCNv2Block(128,    128, A, stride=1, use_adaptive_adj=use_adaptive_adj, dropout=0.20)

        if self.use_temporal_pyr:
            self.temp_att = nn.Sequential(
                nn.Linear(128, 32),
                nn.Tanh(),
                nn.Linear(32, 1),
            )
            head_in = 128 * 3   # [att_pool (128), temporal_diff (128), spatial_temporal_max (128)]
        else:
            head_in = 128

        # Optional v3.1 domain-specific temporal-geometry summary branches (24 -> 32 dims each)
        if self.use_gravity_feats:
            self.grav_mlp = nn.Sequential(
                nn.LayerNorm(24),
                nn.Linear(24, 32),
                nn.SiLU(),
            )
            head_in += 32
        if self.use_wrist_head_feats:
            self.wrist_mlp = nn.Sequential(
                nn.LayerNorm(24),
                nn.Linear(24, 32),
                nn.SiLU(),
            )
            head_in += 32
        if self.use_lower_body_feats:
            self.lower_mlp = nn.Sequential(
                nn.LayerNorm(24),
                nn.Linear(24, 32),
                nn.SiLU(),
            )
            head_in += 32

        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Dropout(0.3),
            nn.Linear(head_in, num_classes),
        )

    @staticmethod
    def _summarize_temporal_signals(sig: torch.Tensor) -> torch.Tensor:
        """
        Summarize a frame-level signal tensor `sig` of shape (B, K, T) into
        a 4*K dimensional descriptor: [mean, std, late_half - early_half, mean_abs_diff].
        """
        mid = max(1, sig.shape[2] // 2)
        s_mean = sig.mean(dim=2)
        s_std  = torch.sqrt(sig.var(dim=2, unbiased=False) + 1e-6)
        s_diff = sig[:, :, mid:].mean(dim=2) - sig[:, :, :mid].mean(dim=2)
        if sig.shape[2] > 1:
            s_osc = (sig[:, :, 1:] - sig[:, :, :-1]).abs().mean(dim=2)
        else:
            s_osc = torch.zeros_like(s_mean)
        return torch.cat([s_mean, s_std, s_diff, s_osc], dim=1)

    @staticmethod
    def _joint_angle_cos(a_xy: torch.Tensor, b_xy: torch.Tensor, c_xy: torch.Tensor) -> torch.Tensor:
        """Cosine of angle A-B-C at vertex B; inputs shaped (B, 2, T), returns (B, 1, T)."""
        u = a_xy - b_xy
        v = c_xy - b_xy
        dot = (u * v).sum(dim=1, keepdim=True)
        norm = torch.sqrt((u ** 2).sum(dim=1, keepdim=True) + 1e-6) * torch.sqrt((v ** 2).sum(dim=1, keepdim=True) + 1e-6)
        return torch.clamp(dot / (norm + 1e-6), -1.0, 1.0)

    def _compute_v31_branches(
        self, raw_x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        From raw input `raw_x` of shape (B, 3, T, V) [x_norm, y_norm, conf],
        compute optional v3.1 per-node feature tensors and global branch embeddings.
        """
        x_c = raw_x[:, 0, :, :]   # (B, T, V)
        y_c = raw_x[:, 1, :, :]   # (B, T, V)
        vis = (raw_x[:, 2, :, :] >= 0.20).float()  # (B, T, V)
        B, T, V = x_c.shape

        node_extras: list[torch.Tensor] = []
        head_extras: list[torch.Tensor] = []

        # Common anchor points
        # Lowest visible lower-body joint (knees 13,14 & ankles 15,16) as virtual ground anchor
        low_y = y_c[:, :, 13:17]
        low_vis = vis[:, :, 13:17]
        any_low = (low_vis.sum(dim=-1, keepdim=True) > 0).float()
        ground_y = torch.where(low_vis > 0, low_y, torch.full_like(low_y, 10.0)).amin(dim=-1, keepdim=True) * any_low  # (B, T, 1)

        if self.use_gravity_feats:
            # 1) Per-node signed vertical acceleration a_y and height above ground anchor
            vy = torch.zeros_like(y_c)
            vis_t1 = vis[:, 1:, :] * vis[:, :-1, :]
            vy[:, 1:, :] = (y_c[:, 1:, :] - y_c[:, :-1, :]) * vis_t1
            ay = torch.zeros_like(y_c)
            vis_t2 = vis_t1[:, 1:, :] * vis_t1[:, :-1, :]
            ay[:, 2:, :] = (vy[:, 2:, :] - vy[:, 1:-1, :]) * vis_t2
            rel_ground = (y_c - ground_y) * vis
            node_extras.append(torch.stack([ay, rel_ground], dim=1))  # (B, 2, T, V)

            # 2) Global gravity-aware signals (B, 6, T)
            sh_x = 0.5 * (x_c[:, :, 5] + x_c[:, :, 6])
            sh_y = 0.5 * (y_c[:, :, 5] + y_c[:, :, 6])
            sh_norm = torch.sqrt(sh_x ** 2 + sh_y ** 2 + 1e-6)
            cos_theta = sh_y / sh_norm
            abs_sin_theta = sh_x.abs() / sh_norm

            any_vis = (vis.sum(dim=-1, keepdim=True) > 0).float()
            y_max = torch.where(vis > 0, y_c, torch.full_like(y_c, -10.0)).amax(dim=-1, keepdim=True) * any_vis
            y_min = torch.where(vis > 0, y_c, torch.full_like(y_c, +10.0)).amin(dim=-1, keepdim=True) * any_vis
            x_max = torch.where(vis > 0, x_c, torch.full_like(x_c, -10.0)).amax(dim=-1, keepdim=True) * any_vis
            x_min = torch.where(vis > 0, x_c, torch.full_like(x_c, +10.0)).amin(dim=-1, keepdim=True) * any_vis
            H_box = torch.clamp(y_max - y_min, min=0.0).squeeze(-1)
            W_box = torch.clamp(x_max - x_min, min=0.0).squeeze(-1)
            aspect_ar = (H_box - W_box) / (H_box + W_box + 1e-6)

            h_hip = -ground_y.squeeze(-1)  # virtual hip height above ground contact (B, T)
            vy_hip = torch.zeros_like(h_hip)
            vy_hip[:, 1:] = h_hip[:, 1:] - h_hip[:, :-1]
            ay_hip = torch.zeros_like(h_hip)
            ay_hip[:, 2:] = vy_hip[:, 2:] - vy_hip[:, 1:-1]

            grav_sig = torch.stack([cos_theta, abs_sin_theta, aspect_ar, h_hip, vy_hip, ay_hip], dim=1)  # (B, 6, T)
            head_extras.append(self.grav_mlp(self._summarize_temporal_signals(grav_sig)))

        if self.use_wrist_head_feats:
            # 1) Per-node relative vector to nose (joint 0)
            nose_x = x_c[:, :, 0:1]
            nose_y = y_c[:, :, 0:1]
            nose_vis = vis[:, :, 0:1]
            dx_nose = (x_c - nose_x) * vis * nose_vis
            dy_nose = (y_c - nose_y) * vis * nose_vis
            node_extras.append(torch.stack([dx_nose, dy_nose], dim=1))  # (B, 2, T, V)

            # 2) Global wrist-to-head signals (B, 6, T)
            d_lw = torch.sqrt((x_c[:, :, 9] - x_c[:, :, 0]) ** 2 + (y_c[:, :, 9] - y_c[:, :, 0]) ** 2 + 1e-6)
            d_rw = torch.sqrt((x_c[:, :, 10] - x_c[:, :, 0]) ** 2 + (y_c[:, :, 10] - y_c[:, :, 0]) ** 2 + 1e-6)
            d_min = torch.minimum(d_lw, d_rw)
            dy_w_max = torch.maximum(y_c[:, :, 9] - y_c[:, :, 0], y_c[:, :, 10] - y_c[:, :, 0])

            p_xy = raw_x[:, :2, :, :]
            cos_el_l = self._joint_angle_cos(p_xy[:, :, :, 5], p_xy[:, :, :, 7], p_xy[:, :, :, 9]).squeeze(1)
            cos_el_r = self._joint_angle_cos(p_xy[:, :, :, 6], p_xy[:, :, :, 8], p_xy[:, :, :, 10]).squeeze(1)

            wrist_sig = torch.stack([d_lw, d_rw, d_min, dy_w_max, cos_el_l, cos_el_r], dim=1)  # (B, 6, T)
            head_extras.append(self.wrist_mlp(self._summarize_temporal_signals(wrist_sig)))

        if self.use_lower_body_feats:
            # 1) Per-node height & vertical velocity relative to ankle midpoint (joints 15, 16)
            ank_y = 0.5 * (y_c[:, :, 15:16] + y_c[:, :, 16:17])
            dy_ank = (y_c - ank_y) * vis
            vy_ank = torch.zeros_like(dy_ank)
            vy_ank[:, 1:, :] = (dy_ank[:, 1:, :] - dy_ank[:, :-1, :]) * vis[:, 1:, :] * vis[:, :-1, :]
            node_extras.append(torch.stack([dy_ank, vy_ank], dim=1))  # (B, 2, T, V)

            # 2) Global lower-body flexion & geometry signals (B, 6, T)
            p_xy = raw_x[:, :2, :, :]
            cos_kn_l = self._joint_angle_cos(p_xy[:, :, :, 11], p_xy[:, :, :, 13], p_xy[:, :, :, 15]).squeeze(1)
            cos_kn_r = self._joint_angle_cos(p_xy[:, :, :, 12], p_xy[:, :, :, 14], p_xy[:, :, :, 16]).squeeze(1)
            kn_y_mid = 0.5 * (y_c[:, :, 13] + y_c[:, :, 14])
            ank_y_mid = 0.5 * (y_c[:, :, 15] + y_c[:, :, 16])
            hip_to_knee = -kn_y_mid
            knee_to_ank = kn_y_mid - ank_y_mid
            hip_to_ank  = -ank_y_mid
            stance_w    = (x_c[:, :, 15] - x_c[:, :, 16]).abs()

            lower_sig = torch.stack([cos_kn_l, cos_kn_r, hip_to_knee, knee_to_ank, hip_to_ank, stance_w], dim=1)
            head_extras.append(self.lower_mlp(self._summarize_temporal_signals(lower_sig)))

        return node_extras, head_extras

    def _compute_kinematics(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, T, V) -> [x_norm, y_norm, conf]
        x_parent = x[:, :, :, self.parents]
        both_vis = ((x[:, 2:3, :, :] > 0) & (x_parent[:, 2:3, :, :] > 0)).float()
        bone_xy  = (x[:, :2, :, :] - x_parent[:, :2, :, :]) * both_vis
        bone_len = torch.sqrt((bone_xy ** 2).sum(dim=1, keepdim=True) + 1e-8) * both_vis
        bone     = torch.cat([bone_xy, bone_len], dim=1)  # (B, 3, T, V)

        vel_xy = torch.zeros_like(x[:, :2, :, :])
        both_t_vis = ((x[:, 2:3, 1:, :] > 0) & (x[:, 2:3, :-1, :] > 0)).float()
        vel_xy[:, :, 1:, :] = (x[:, :2, 1:, :] - x[:, :2, :-1, :]) * both_t_vis
        vel_mag = torch.sqrt((vel_xy ** 2).sum(dim=1, keepdim=True) + 1e-8)
        motion  = torch.cat([vel_xy, vel_mag], dim=1)     # (B, 3, T, V)

        return torch.cat([x, bone, motion], dim=1)        # (B, 9, T, V)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_x = x
        node_extras, head_extras = self._compute_v31_branches(raw_x)
        if self.use_kinematic_feats:
            x = self._compute_kinematics(raw_x)
        if node_extras:
            x = torch.cat([x] + node_extras, dim=1)

        B, C, T, V = x.shape
        x_bn = x.permute(0, 1, 3, 2).contiguous().view(B, C * V, T)
        x_bn = self.data_bn(x_bn).view(B, C, V, T).permute(0, 1, 3, 2)

        h = self.b1(x_bn)
        h = self.b2(h)
        h = self.b3(h)
        h = self.b4(h)  # (B, 128, T', V)

        h_v = h.mean(dim=-1)  # (B, 128, T')

        if self.use_temporal_pyr:
            h_perm = h_v.permute(0, 2, 1)                       # (B, T', 128)
            att_w  = torch.softmax(self.temp_att(h_perm), dim=1)
            h_att  = (h_perm * att_w).sum(dim=1)                # (B, 128)
            mid    = max(1, h_v.shape[2] // 2)
            h_diff = h_v[:, :, mid:].mean(dim=2) - h_v[:, :, :mid].mean(dim=2)
            h_max  = h.amax(dim=(2, 3))                         # (B, 128)
            feat   = torch.cat([h_att, h_diff, h_max], dim=1)   # (B, 384)
        else:
            feat = h_v.mean(dim=-1)                             # (B, 128)

        if head_extras:
            feat = torch.cat([feat] + head_extras, dim=1)

        return self.head(feat)


# ════════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════════

def _instantiate_stgcn_from_state_dict(state: dict) -> nn.Module:
    """
    Auto-detect whether `state` belongs to ActionSTGCNv1 or ActionSTGCN
    (v2.1–v2.4 / v3.0-clean / v3.1-A/B/C/D) by inspecting parameter keys and shapes.
    """
    if "layers.0.gcn.weight" in state:
        num_cls = int(state["fc.weight"].shape[0])
        return ActionSTGCNv1(num_classes=num_cls)

    # ActionSTGCN (v2.1 .. v3.1-D)
    in_c_b1 = int(state["b1.gcn.weight"].shape[1])
    use_grav  = ("grav_mlp.1.weight" in state)
    use_wrist = ("wrist_mlp.1.weight" in state)
    use_lower = ("lower_mlp.1.weight" in state)
    extra_c = (2 if use_grav else 0) + (2 if use_wrist else 0) + (2 if use_lower else 0)
    use_kin = ((in_c_b1 - extra_c) == NUM_COORDS * 3)
    use_pyr = ("temp_att.0.weight" in state)
    use_adap = ("b1.A_adap" in state)
    num_cls = int(state["head.2.weight"].shape[0])
    return ActionSTGCN(
        in_channels=NUM_COORDS,
        num_classes=num_cls,
        num_joints=NUM_JOINTS,
        use_temporal_pyr=use_pyr,
        use_adaptive_adj=use_adap,
        use_kinematic_feats=use_kin,
        use_gravity_feats=use_grav,
        use_wrist_head_feats=use_wrist,
        use_lower_body_feats=use_lower,
    )


def load_action_model(
    path: str,
    model_type: ModelType,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Load a saved action model checkpoint from `path`.

    Automatically detects ST-GCN v1 (`ActionSTGCNv1`) vs. ST-GCN v2.1–v2.4
    (`ActionSTGCN`) from state_dict keys/shapes.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not Path(path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    raw_ckpt = torch.load(path, map_location=device, weights_only=True)
    if isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt:
        state = raw_ckpt["state_dict"]
    else:
        state = raw_ckpt

    if model_type == "lstm":
        num_cls = int(state["fc.weight"].shape[0])
        hid_sz  = int(state["fc.weight"].shape[1]) // 2
        model: nn.Module = ActionLSTM(hidden_size=hid_sz, num_classes=num_cls)
    elif model_type == "stgcn":
        model = _instantiate_stgcn_from_state_dict(state)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Must be 'lstm' or 'stgcn'.")

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    log.info("Loaded %s model (%s) from %s", model_type, type(model).__name__, path)
    return model


def classify_sequence(
    buffer: SequenceBuffer,
    models: dict,
    active_model: ModelType = ACTIVE_ACTION_MODEL,
    device: Optional[torch.device] = None,
    label_map: Optional[dict[int, str]] = None,
) -> ActionPrediction:
    """
    Run action classification on a filled SequenceBuffer.
    """
    if not buffer.is_ready():
        log.debug("Buffer not ready — returning default prediction.")
        return ActionPrediction()

    if not buffer.has_valid_subject():
        log.debug("Prolonged no-person in buffer — skipping classification.")
        return ActionPrediction(
            label="No_Person",
            label_idx=-1,
            confidence=0.0,
            model_type=active_model,
        )

    if active_model not in models:
        log.warning("Model '%s' not in models dict — returning default.", active_model)
        return ActionPrediction()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models[active_model]
    model.eval()

    with torch.no_grad():
        if active_model == "lstm":
            tensor = buffer.as_lstm_tensor().to(device)   # (1, T, J*C)
        else:
            tensor = buffer.as_stgcn_tensor().to(device)  # (1, C, T, J)

        logits = model(tensor)                             # (1, num_classes)
        probs  = torch.softmax(logits, dim=-1)             # (1, num_classes)
        idx    = int(torch.argmax(probs, dim=-1).item())
        conf   = float(probs[0, idx].item())

    if label_map is not None:
        label = label_map.get(idx, "Unknown")
    else:
        label = CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else "Unknown"

    return ActionPrediction(
        label=label,
        label_idx=idx,
        confidence=conf,
        model_type=active_model,
    )


# ════════════════════════════════════════════════════════════════════════════
# Probability-level API (used by the inference stabilizer + ensemble mode)
# ════════════════════════════════════════════════════════════════════════════

ENSEMBLE_KEY = "ensemble"
STGCN_WRIST_KEY = "stgcn_wrist"
_DUAL_SPEC_W_C = np.array(
    [0.50, 0.70, 0.30, 0.60, 0.25, 0.45, 0.60, 0.40],
    dtype=np.float64,
)


def predict_probs(
    buffer: SequenceBuffer,
    models: dict,
    active_model: str = ACTIVE_ACTION_MODEL,
    device: Optional[torch.device] = None,
) -> Optional[np.ndarray]:
    """
    Return the softmax probability vector (np.ndarray, shape (num_classes,))
    for the current window, or None when the buffer is not ready / has no
    valid subject / requested model is unavailable.

    When `models` includes `"stgcn_wrist"`, ST-GCN inference automatically
    applies the V4 Dual-Specialist class-weighted fusion (`v4-C` Lower-Body +
    `v4-B` Wrist-to-Head + max Fall safety gate). `active_model="ensemble"`
    averages that ST-GCN distribution with the LSTM baseline.
    """
    if not buffer.is_ready() or not buffer.has_valid_subject():
        return None

    keys = ["stgcn", "lstm"] if active_model == ENSEMBLE_KEY else [active_model]
    if any(k not in models for k in keys):
        return None

    dev = device if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    prob_list: list[np.ndarray] = []
    with torch.no_grad():
        for k in keys:
            model = models[k]
            model.eval()
            if k == "lstm":
                tensor = buffer.as_lstm_tensor().to(dev)
                probs = (
                    torch.softmax(model(tensor), dim=-1)[0]
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
            else:
                tensor = buffer.as_stgcn_tensor().to(dev)
                probs = (
                    torch.softmax(model(tensor), dim=-1)[0]
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                if k == "stgcn":
                    wrist_model = models.get(STGCN_WRIST_KEY)
                    if wrist_model is not None:
                        wrist_model.eval()
                        probs_b = (
                            torch.softmax(wrist_model(tensor), dim=-1)[0]
                            .cpu()
                            .numpy()
                            .astype(np.float64)
                        )
                        if probs.shape == probs_b.shape == _DUAL_SPEC_W_C.shape:
                            fused = (
                                probs * _DUAL_SPEC_W_C
                                + probs_b * (1.0 - _DUAL_SPEC_W_C)
                            )
                            fused[3] = max(
                                float(fused[3]), float(probs[3]), float(probs_b[3])
                            )
                            total = float(fused.sum())
                            probs = fused / total if total > 0.0 else probs
            prob_list.append(probs)
    return np.mean(np.stack(prob_list, axis=0), axis=0)
