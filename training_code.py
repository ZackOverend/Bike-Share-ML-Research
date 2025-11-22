# --- MODEL TRAINING SECTION ---
from pycaret.regression import *

# 1. Setup the Experiment
# We drop 'casual' and 'registered' to prevent leakage
# We drop 'instant' and 'dteday' as they are not useful features
# CRITICAL UPDATE: We force 'hr' and 'mnth' to be categorical to capture non-linear cycles
categorical_cols = ['hr', 'mnth', 'weekday', 'season', 'weathersit', 'holiday', 'workingday', 'yr']

s = setup(data=df, 
          target='cnt', 
          ignore_features=['casual', 'registered', 'instant', 'dteday'],
          categorical_features=categorical_cols,
          session_id=123) 

# 2. Compare Models
# This will train ~20 models and rank them
best_model = compare_models()

# 3. Analyze the Best Model
# Feature Importance (Support your thesis!)
plot_model(best_model, plot='feature')

# Residuals (Where is it wrong?)
plot_model(best_model, plot='residuals')

# Prediction Error
plot_model(best_model, plot='error')

# 4. Finalize
# Create the final model trained on ALL data (not just training set)
final_best = finalize_model(best_model)
print(best_model)
