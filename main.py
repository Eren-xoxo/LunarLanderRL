import gymnasium as gym
import numpy as np
import torch
import os
from dqn_agent import DQNAgent
from utils import plot_rewards

# === Konfiguration ===
episodes = 10  # Gesamtzahl Trainingsrunden
save_at = [10]  # Liste von Episoden, die gespeichert werden sollen
#save_at = "all"      # Oder: "all", um alle Episoden zu speichern

save_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(save_dir, exist_ok=True)

# === Setup ===
env = gym.make("LunarLander-v3")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

rewards = []

for ep in range(1, episodes + 1):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, truncated, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        state = next_state
        total_reward += reward

    rewards.append(total_reward)
    print(f"Episode {ep}: Reward = {total_reward:.2f} | Epsilon = {agent.epsilon:.4f}")

    # Modell speichern je nach Einstellung
    if save_at == "all" or ep in save_at:
        model_path = os.path.join(save_dir, f"lunarlander_ep{ep}.pth")
        torch.save(agent.qnetwork.state_dict(), model_path)
        print(f" Modell gespeichert: {model_path}")

# Plot anzeigen
plot_rewards(rewards)
env.close()
