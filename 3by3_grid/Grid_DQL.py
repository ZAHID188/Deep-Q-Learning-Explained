import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random

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


class DQLAgent:
    def __init__(self, state_size_input, action_size_input):
        self.state_size = state_size_input
        self.action_size = action_size_input
        self.memory = deque(maxlen=2000)  # Memory for experience replay
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()  # simple way to build neural networks layer by layer
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))  # input Layer
        model.add(Dense(24, activation='relu'))  # hidden layer
        model.add(Dense(self.action_size, activation='linear'))  # output layer
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            print("random-", np.random.rand(), self.epsilon, random.randrange(self.action_size))
            return random.randrange(self.action_size)  # explore random move up, down, left, right
        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])  # Exploit: choose from what models knows

    def replay(self, batch_size):
        print(self.memory)
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = np.reshape(next_state, [1, self.state_size])
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
                print(" reply function targt:",target)
            state = np.reshape(state, [1, self.state_size])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Initialize environment and agent
env = GridWorld()
state_size = 2  # (x, y) position
action_size = 4  # Up, Down, Left, Right
agent = DQLAgent(state_size, action_size)

# Training parameters
episodes = 100
batch_size = 6

# Training loop
for episode in range(episodes):
    state = env.reset()
    state = np.array(state)
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        next_state = np.array(next_state)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

# Save the trained model
agent.model.save("dql_model_2.h5")
print("Model saved as dql_model.h5")

# Test the trained agent
state = env.reset()
state = np.array(state)
done = False
total_reward = 0

while not done:
    action = agent.act(state)
    next_state, reward, done = env.step(action)
    next_state = np.array(next_state)
    state = next_state
    total_reward += reward
    print(f"State: {state}, Action: {action}, Reward: {reward}")

print(f"Total Reward: {total_reward}")






'''import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random

class GridWorld:
    def __init__(self):
        self.grid=np.zeros((3,3))
        self.goal=(2,2)
        self.state=(2,1)
    def reset(self):
        self.state=(0,0)
        return self.state
    def step(self,action):
        x,y=self.state
        if action==0:
            x=max(x-1,0) #up
        elif action==1:
            x=min(x+1,2) #down
        elif action==2:
            y=max(y-1,0) #left
        elif action==3:
            y=min(y+1,2) #right

        self.state=(x,y)
        if self.state ==self.goal:
            reward=10
            Done=True
        else:
            reward=-1
            Done=False
        print(x,y)
        return self.state, reward, Done

class DQLAgent:
    def __init__(self,state_size_input,action_size_input):
        self.state_size=state_size_input
        self.action_size=action_size_input
        self.memory=deque(maxlen=2000)  # Memory for experience replay
        self.gamma=0.95 #discount factor
        self.epsilon=1.0 #experience rate
        self.epsilon_min=0.01
        self.epsilon_decay=0.995
        self.learning_rate=0.001
        self.model=self._build_model()
    def _build_model(self):
        model=Sequential() #simple way to build neural networks layer by layer
        model.add(Dense(24,input_dim=self.state_size,activation='relu')) #input Layer
        model.add(Dense(24, activation='relu')) #hidden layer
        model.add(Dense(self.action_size,activation='linear')) #output layer
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
    
    def act(self,state):
        if np.random.rand()<=self.epsilon:
            print("random-",np.random.rand(),self.epsilon, random.randrange(self.action_size))
            return random.randrange(self.action_size)  # explore random move up, down, left, right
        state=np.reshape(state,[1,self.state_size])
        q_values=self.model.predict(state)
        return np.argmax(q_values[0]) # Exploit : choose from what models knows
    
    def replay(self,batch_size):
        if len(self.memory)<batch_size:
            return
        minibatch=random.sample(self.memory,batch_size)  # minibatch = [state2, action2, reward2, next_state2, done2),(state4, action4, reward4, next_state4, done4),]
        for state,action,reward,next_state,done in minibatch:
            target=reward
            if not done:
                next_state=np.reshape(next_state,[1,self.state_size])
                target=reward +self.gamma * np.amax(self.model.predict(next_state)[0])
            state=np.reshape(state,[1,self.state_size])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon*=self.epsilon_decay




# Initialize environment and agent
env = GridWorld()
state_size = 2  # (x, y) position
action_size = 4  # Up, Down, Left, Right
agent = DQLAgent(state_size, action_size)   

# Training parameters
episodes = 1000
batch_size = 32

# Training loop
for episode in range(episodes):
    state = env.reset()
    state = np.array(state)
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        next_state = np.array(next_state)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)



# Test the trained agent
state = env.reset()
state = np.array(state)
done = False
total_reward = 0

while not done:
    action = agent.act(state)
    next_state, reward, done = env.step(action)
    next_state = np.array(next_state)
    state = next_state
    total_reward += reward
    print(f"State: {state}, Action: {action}, Reward: {reward}")

print(f"Total Reward: {total_reward}")


x=GridWorld()
agent=DQLAgent(state_size_input=2,action_size_input=4)

print(agent.act((0,0)))'''