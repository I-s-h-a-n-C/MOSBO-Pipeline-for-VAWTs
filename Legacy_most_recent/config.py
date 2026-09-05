import os

# =============================================================================
# 1. DIRECTORY PATHS
# =============================================================================
# Assuming data is in the same directory or a specific data folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR  # Update this if your CSVs are in a subfolder like './data'

# Updated to reflect folder structures for all 8 NACA profiles
CP_FOLDERS = [
    "Cp_0013", 
    "Cp_0015", 
    "Cp_0018", 
    "Cp_0021"
]

AZIM_FOLDERS = [
    "Azim_0013", 
    "Azim_0015", 
    "Azim_0018", 
    "Azim_0021"
]

# =============================================================================
# 2. PHYSICAL BOUNDARIES & RULES (Stage 2)
# =============================================================================
# Impossible Cp values (Betz limit is ~0.59, giving some buffer)
CP_MIN = -0.2
CP_MAX = 0.65

# Impossible TRF values (Torque ripple cannot be negative or zero)
TRF_MIN = 0.0
TRF_MAX = 5.0

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
LAMBDA_PENALTY = 0.25  

POPULATION_SIZE = 500
OFFSPRING_SIZE = 250
GENERATIONS = 250

# Hard TRF ceiling used consistently across the pipeline:
#  - NSGA-II inequality constraint (RiskAwareVAWTProblem)
#  - SLSQP local-refinement constraint (trf_constraint)
#  - Post-optimization hard filter on the final Pareto export
# FIX: Previously NSGA-II/SLSQP allowed TRF <= 2.0 while the final export
# filtered to TRF <= 1.5, silently dropping a large chunk of the discovered
# front (including much of the high-Cp end) before it ever reached the
# comparison plot or the CSV export. Keeping this in one place prevents
# that mismatch from reappearing.
TRF_PARETO_LIMIT = 1.5