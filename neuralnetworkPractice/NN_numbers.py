import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential,save_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import pickle

# Generate more training data
X = np.array([[i] for i in range(20)])  # More training points
y = 2 * X + 1

# Normalize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Build a deeper model
model = Sequential([
    Dense(16, activation='relu', input_dim=1),
    Dense(8, activation='relu'),
    Dense(1)
])

# Compile and train
model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
history = model.fit(X_scaled, y_scaled, epochs=500, verbose=0)

# Plot loss
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
# Save the model and scalers
save_model(model, 'linear_model.h5')

# Save scalers
with open('scalers.pkl', 'wb') as f:
    pickle.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)


# Make predictions
new_X = np.array([[10], [11], [12], [13], [14]])
new_X_scaled = scaler_X.transform(new_X)
predictions_scaled = model.predict(new_X_scaled)
predictions = scaler_y.inverse_transform(predictions_scaled)

# Print predictions
for i, x in enumerate(new_X):
    print(f"Input: {x[0]}, Predicted Output: {predictions[i][0]:.2f}, Expected: {2*x[0] + 1}")