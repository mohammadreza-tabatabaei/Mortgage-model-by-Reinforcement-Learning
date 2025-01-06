from model import MortgageModel
from policy import DQNAgent
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Initialize the environment (MortgageModel)
    env = MortgageModel(ltv=0.8, income=50000, credit_score=700, trend=0)

    # Set up the RL agent
    state_size = len(env.state)  # Number of state variables (ltv, income, credit_score, trend)
    action_size = len(env.actions)  # Number of actions
    agent = DQNAgent(state_size, action_size)

    # Visualization data collectors
    all_rewards = []
    all_actions = []
    simulated_rewards = []

    # Training parameters
    episodes = 1000
    batch_size = 32

    # Training phase
    for e in range(episodes):
        state = np.array(list(env.state.values())).reshape(1, -1)  # Initial state from the environment
        total_reward = 0
        episode_actions = []

        for time in range(200):  # A maximum of 200 time steps per episode
            # Select action using the agent's policy (epsilon-greedy)
            action_idx = agent.act(state)
            episode_actions.append(action_idx)
            action = env.actions[action_idx]  # Map index to actual action (approve, rate)

            # Simulate exogenous factors (house value change, credit change, income change)
            exogenous_info = (
                np.random.uniform(-0.02, 0.02),
                np.random.randint(-10, 10),
                np.random.randint(-2000, 2000)
            )

            # Transition the environment with the action and exogenous info
            next_state = env.transition(action, exogenous_info)

            # Calculate the reward based on the action taken
            reward = env.reward(action)
            total_reward += reward

            # Check if the episode is done (e.g., after 200 steps or you can define another condition)
            done = time == 199

            # Prepare the next state for the agent (reshape it to match the input format)
            next_state = np.array(list(next_state.values())).reshape(1, -1)

            # Remember the experience in the agent's memory
            agent.remember(state, action_idx, reward, next_state, done)

            # Update the state
            state = next_state

            # If the episode is done, break the loop
            if done:
                print(f"Episode {e + 1}/{episodes}, Total Reward: {total_reward}")
                break

        # Record rewards and actions
        all_rewards.append(total_reward)
        all_actions.extend(episode_actions)

        # Train the agent using a batch of experiences from memory
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    # Optionally, after training, simulate the performance using the trained policy
    simulated_rewards = simulate_policy(env, agent, 100)
    plot_rewards(all_rewards)
    plot_action_distribution(all_actions, ['Deny', 'Approve Low Rate', 'Approve High Rate'])
    plot_simulated_rewards(simulated_rewards)

def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward Progression Over Episodes')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_action_distribution(actions, labels):
    counts = [actions.count(i) for i in range(len(labels))]
    plt.figure(figsize=(7, 7))
    plt.pie(counts, labels=labels, autopct='%1.1f%%')
    plt.title('Action Distribution')
    plt.show()

def plot_simulated_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Simulated Reward')
    plt.xlabel('Simulation Step')
    plt.ylabel('Reward')
    plt.title('Simulated Rewards Over Steps')
    plt.legend()
    plt.grid(True)
    plt.show()

def simulate_policy(env, agent, steps=100):
    """
    Function to simulate the trained policy after training.
    """
    total_rewards = []
    for _ in range(steps):
        state = np.array(list(env.state.values())).reshape(1, -1)
        action_idx = agent.act(state)
        action = env.actions[action_idx]

        # Simulate exogenous factors
        exogenous_info = (
            np.random.uniform(-0.02, 0.02),
            np.random.randint(-10, 10),
            np.random.randint(-2000, 2000)
        )

        # Transition the environment and calculate the reward
        next_state = env.transition(action, exogenous_info)
        reward = env.reward(action)
        total_rewards.append(reward)

    print(f"Simulated Total Reward: {sum(total_rewards)}")
    return total_rewards

if __name__ == "__main__":
    main()
