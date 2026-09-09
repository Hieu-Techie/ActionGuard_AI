"""Central configuration placeholders for the fall detection prototype."""

WINDOW_SIZE = 16
TARGET_FPS = 10
CONF_THRESHOLD = 0.50
WARNING_THRESHOLD = 0.50
CRITICAL_THRESHOLD = 0.75
INSTANT_RED_THRESHOLD = 0.90
N_WARN_FRAMES = 3
N_CRITICAL_FRAMES = 3
N_RECOVERY_FRAMES = 5

YOLO_MODEL_PATH = "models/yolo_person_detector.pt"
ACTION_MODEL_PATH = "models/action_lstm.pth"