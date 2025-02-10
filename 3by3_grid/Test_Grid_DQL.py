import numpy as np
from tensorflow.keras.models import load_model

class GridWorld:
    def __init__(self):
        self.grid = np.zeros((3, 3))
        self.goal = (2, 2)
        self.state = (2, 1)

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:
            x = max(x - 1, 0)  # up
        elif action == 1:
            x = min(x + 1, 2)  # down
        elif action == 2:
            y = max(y - 1, 0)  # left
        elif action == 3:
            y = min(y + 1, 2)  # right

        self.state = (x, y)
        if self.state == self.goal:
            reward = 10
            Done = True
        else:
            reward = -1
            Done = False
        print(x, y)
        return self.state, reward, Done

def test_model():
    # Load the trained model
    model = load_model("dql_model_2_test.h5")
    env = GridWorld()
    
    # Test the trained agent
    state = env.reset()
    state = np.array(state)
    done = False
    total_reward = 0

    while not done:
        state_reshaped = np.reshape(state, [1, 2])
        q_values = model.predict(state_reshaped, verbose=0)
        action = np.argmax(q_values[0])
        
        next_state, reward, done = env.step(action)
        next_state = np.array(next_state)
        state = next_state
        total_reward += reward
        print(f"State: {state}, Action: {action}, Reward: {reward}")

    print(f"Total Reward: {total_reward}")

if __name__ == "__main__":
    test_model()