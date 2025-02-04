## Deep-Q-Learning 
Environment: A 3x3 grid world.

- The agent starts at the top-left corner.

- The goal is at the bottom-right corner.

- The agent can move up, down, left, or right.

- If the agent reaches the goal, it gets a reward of +10. If it hits a wall, it gets a reward of -1.

Objective: The agent learns to navigate the grid to reach the goal in the fewest steps.

### 3x3 grid 
![3x3 grid ](temp/grid.png)

### Define the Environment
**GridWorld class**

<details>
<summary>Init function</summary>

- Defining Grid 3*3 using `numpy`
- Defining Goal and state

</details>

<details>
<summary>Reset</summary>

- Reset the state 
- start from the begining `(0,0)`

</details>

<details>
<summary>Step</summary>

- To do the Action
- 0 for go up, 1 for down, 2 left, 3 down
- After move , check the state reached the goal or not
- if Target reached get `reward` and `Done`

</details>