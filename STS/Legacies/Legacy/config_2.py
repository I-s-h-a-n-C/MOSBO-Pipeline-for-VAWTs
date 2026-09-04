import os

# =============================================================================
# 1. DIRECTORY PATHS
# =============================================================================
# Assuming data is in the same directory or a specific data folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR  # Update this if your CSVs are in a subfolder like './data'

CP_FILES = ["Cp_0015.csv", "Cp_0018.csv"]
AZIM_FOLDERS = ["Azim_0015", "Azim_0018"]

# =============================================================================
# 2. PHYSICAL BOUNDARIES & RULES (Stage 2)
# =============================================================================
# Impossible Cp values (Betz limit is ~0.59, giving some buffer)
CP_MIN = -0.2
CP_MAX = 0.65

# Impossible TRF values (Torque ripple cannot be negative or zero)
TRF_MIN = 0.0

# Z-score threshold for outlier detection (values > 3 standard deviations are dropped)
Z_SCORE_THRESHOLD = 3.0

# =============================================================================
# 3. MACHINE LEARNING HYPERPARAMETERS
# =============================================================================
RANDOM_STATE = 42
TRAIN_TEST_SPLIT = 0.2  # 80% Train, 20% Test
CV_FOLDS = 5            # 5-Fold Cross Validation

# =============================================================================
# 4. OPTIMIZATION SETTINGS
# =============================================================================
# Risk penalty multiplier (λ) for UCB/LCB in optimization
# Objective = Mean +/- (LAMBDA_PENALTY * StdDev)
LAMBDA_PENALTY = 1.0  

POPULATION_SIZE = 200
OFFSPRING_SIZE = 100
GENERATIONS = 100