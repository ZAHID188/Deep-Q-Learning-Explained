import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

x=[1,4,3,5,6,83,2,8,9]
print(np.amax(x))

def model(self):
    model=Sequential() 
    model.add(Dense(24,input_dim=2,activation='relu')) 
    model.add(Dense(24, activation='relu'))
    model.add(Dense(4,activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model