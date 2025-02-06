import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load the model from the file
loaded_model = load_model('linear_model.h5')

# Load scalers
with open('scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)  # Changed from dump to load
    loaded_scaler_X = scalers['scaler_X']
    loaded_scaler_y = scalers['scaler_y']

# Make predictions
new_X = np.array([[30], [34], [67], [345], [45]])
new_X_scaled = loaded_scaler_X.transform(new_X)
predictions_scaled = loaded_model.predict(new_X_scaled)
predictions = loaded_scaler_y.inverse_transform(predictions_scaled)

# Print predictions
for i, x in enumerate(new_X):
    print(f"Input: {x[0]}, Predicted Output: {predictions[i][0]:.2f}, Expected: {2*x[0] + 1}")