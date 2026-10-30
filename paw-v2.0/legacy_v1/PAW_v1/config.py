"""PAW v1 runtime settings. Edit values here; no command-line flags needed."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STORE_DIR = ROOT / "PAW_store"
SEED = 42
EPOCHS_PER_TRAIN = 160
BATCH_SIZE = 32
LEARNING_RATE = 0.008
VALIDATION_FRACTION = 0.20
GRADIENT_CLIP_NORM = 5.0
SATURATION_ABS = 0.95
DEAD_VARIANCE = 1e-7
LOCK_TIMEOUT_SECONDS = 10.0
MAX_CSV_ROWS = 1_000_000
MAX_TASK_NAME_LENGTH = 100
LOG_EVERY_EPOCHS = 20
