ENABLE_LIVENESS_CHECK = False

LIVENESS_THRESHOLD = 0.5

LIVENESS_MODEL_PATH = 'models/2.7_80x80_MiniFASNetV2.pth'

# Tracking and Voting Configuration
ENABLE_LIVENESS_TRACKING = True

# Voting method: 'majority', 'weighted', or 'confidence_threshold'
LIVENESS_VOTING_METHOD = 'weighted'

# Number of predictions to keep for voting
LIVENESS_MAX_HISTORY = 10

# Minimum samples needed for reliable voting
LIVENESS_MIN_SAMPLES = 3

# Track timeout in seconds
LIVENESS_TRACK_TIMEOUT = 2.0

# Minimum stability score to trust the result (0-1)
LIVENESS_MIN_STABILITY = 0.6
