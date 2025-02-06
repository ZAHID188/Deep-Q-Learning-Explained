


## Neural Network Explained
In this example, we will create a simple neural network to learn a linear relationship defined by the equation:

$$𝑦=2𝑥+1$$

### Step 1: Understanding the Data

- **Input (`x`)**: This is the data we will feed into our neural network.
  - Example: `x = [1, 2, 3]`
  
- **Output (`y`)**: This is the expected result we want the network to learn.
  - According to our equation, the output will be:
    - For \( x = 1 \): \( y = 2(1) + 1 = 3 \)
    - For \( x = 2 \): \( y = 2(2) + 1 = 5 \)
    - For \( x = 3 \): \( y = 2(3) + 1 = 7 \)
  - Therefore, `y = [3, 5, 7]`

### Step 2: Create the Training Script

You will first run a training script named `model_train.py` to train the neural network.

#### `model_train.py`

1. **Importing Libraries**:
   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   from tensorflow.keras.models import Sequential, save_model
   from tensorflow.keras.layers import Dense
   from tensorflow.keras.optimizers import Adam
   from sklearn.preprocessing import StandardScaler
   import pickle
   ```
   - **NumPy**: For numerical operations.
   - **Matplotlib**: For plotting the training loss.
   - **TensorFlow/Keras**: For building and training the neural network.
   - **Scikit-learn**: For data normalization.
   - **Pickle**: For saving objects to files.
2. **Generating Training Data**:
   ```python
   # Generate more training data
   X = np.array([[i] for i in range(20)])  # More training points
   y = 2 * X + 1
   ```
   - Generates input values \( X \) from 0 to 19.
   - Calculates corresponding output values \( y \) using the equation \( y = 2x + 1 \).

3. **Normalizing the Data**:
   ```python
   # Normalize the data
   scaler_X = StandardScaler()
   scaler_y = StandardScaler()
   X_scaled = scaler_X.fit_transform(X)
   y_scaled = scaler_y.fit_transform(y)
   ```
   - **StandardScaler**: Normalizes the data to have a mean of 0 and a standard deviation of 1.
   - The input data \( X \) and output data \( y \) are both scaled.

4. **Building the Model**:
   ```python
   # Build a deeper model
   model = Sequential([
       Dense(16, activation='relu', input_dim=1),
       Dense(8, activation='relu'),
       Dense(1)     
       ])
   ```
   - A **Sequential** model is created with:
     - An input layer with 16 neurons and ReLU activation.
     - A hidden layer with 8 neurons and ReLU activation.
     - An output layer with 1 neuron.

5. **Compiling and Training the Model**:
   ```python
   # Compile and train
   model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
   history = model.fit(X_scaled, y_scaled, epochs=500, verbose=0)
   ```
   - The model is compiled using the Adam optimizer and mean squared error as the loss function.
   - The model is trained for 500 epochs, with training output suppressed (`verbose=0`).

6. **Plotting the Training Loss**:
   ```python
   # Plot loss
   plt.plot(history.history['loss'])
   plt.title('Model Loss')
   plt.ylabel('Loss')
   plt.xlabel('Epoch')
   plt.show()
   ```
   - Displays a plot of the loss over epochs, allowing you to visualize the training process.

7. **Saving the Model and Scalers**:
   ```python
   # Save the model and scalers
   save_model(model, 'linear_model.h5')

   # Save scalers
   with open('scalers.pkl', 'wb') as f:
       pickle.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)
   ```
   - The trained model is saved to a file named `linear_model.h5`.
   - The scalers used for normalization are saved to a file named `scalers.pkl` using `pickle`.

8. **Making Predictions**:
   ```python
   # Make predictions
   new_X = np.array([[10], [11], [12], [13], [14]])
   new_X_scaled = scaler_X.transform(new_X)
   predictions_scaled = model.predict(new_X_scaled)
   predictions = scaler_y.inverse_transform(predictions_scaled)

   # Print predictions
   for i, x in enumerate(new_X):
       print(f"Input: {x[0]}, Predicted Output: {predictions[i][0]:.2f}, Expected: {2*x[0] + 1}")
   ```
   - New input data is created for predictions.
   - The input data is scaled using the previously fitted scaler.
   - The model predicts the scaled outputs, which are then inverted back to the original scale.
   - Finally, the predicted output and the expected output are printed for comparison.

 ### Create the Prediction Script

After training, you will run a second script named `run_model.py` to make predictions using the trained model.

#### `run_model.py`

```python
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
```