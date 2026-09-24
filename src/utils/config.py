"""Central configuration placeholders for the fall detection prototype."""

# ── Temporal ────────────────────────────────────────────────────────────────
WINDOW_SIZE = 16          # frames per classification window (finalised in Phase 3)
TARGET_FPS = 10           # target inference rate

# ── Detection thresholds ────────────────────────────────────────────────────
CONF_THRESHOLD = 0.50         # minimum YOLO person/keypoint confidence to accept a detection
WARNING_THRESHOLD = 0.50      # action model confidence → WARNING alert
CRITICAL_THRESHOLD = 0.75     # action model confidence → CRITICAL alert
INSTANT_RED_THRESHOLD = 0.90  # single-frame confidence override to RED

# ── Alert state-machine frame counters ──────────────────────────────────────
N_WARN_FRAMES = 3        # consecutive WARNING predictions before YELLOW
N_CRITICAL_FRAMES = 3    # consecutive CRITICAL predictions before RED
N_RECOVERY_FRAMES = 5    # consecutive NORMAL predictions before downgrading

# ── Skeleton / joint schema ─────────────────────────────────────────────────
# Number of joints tracked per person (placeholder — finalise in Phase 1.2 / Phase 2).
# YOLO-Pose (COCO) exposes 17 keypoints; MediaPipe Pose exposes 33.
NUM_JOINTS = 17

# Coordinates stored per joint: (x, y, confidence) for YOLO-Pose COCO keypoints.
# Set to 2 if only (x, y) are used after normalization, or 3 to keep confidence.
NUM_COORDS = 3

# Canonical joint order (COCO 17-keypoint convention, used by YOLO-Pose).
# This list is the single source of truth for index → body-part mapping.
# Any alternative extraction model (MediaPipe) must remap to this order before saving.
JOINT_NAMES = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]

# ── Action model selection ───────────────────────────────────────────────────
# "lstm"  → use LSTM-based action classifier (lighter, lower latency)
# "stgcn" → use ST-GCN action classifier   (graph-based, potentially higher accuracy)
# Finalised after Phase 3 comparison; both remain selectable at runtime.
ACTIVE_ACTION_MODEL = "lstm"  # default; override via sidebar at runtime

# ── ST-GCN adjacency matrix ─────────────────────────────────────────────────
# Path to a pre-serialised adjacency matrix (.npy) for the ST-GCN model.
# The matrix must use the same joint index ordering as JOINT_NAMES above.
# Leave as None until the matrix is generated in Phase 3.
STGCN_ADJACENCY_MATRIX_PATH = None  # e.g., "models/stgcn_adjacency.npy"

# ── Model file paths ─────────────────────────────────────────────────────────
YOLO_POSE_MODEL_PATH = "models/yolo_pose_pretrained.pt"   # Stage 1 pose detector
LSTM_MODEL_PATH = "models/action_lstm.pth"                # Stage 2 action — LSTM
STGCN_MODEL_PATH = "models/action_stgcn.pth"             # Stage 2 action — ST-GCN