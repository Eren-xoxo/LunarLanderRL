import torch
import gymnasium as gym
from dqn_agent import QNetwork
import time
import os

# Modellpfad
model_path = r"C:\Users\Lenovo\Desktop\schule\OneDrive - HTL Anichstrasse\WPG_KISY\4 Klasse\Korber\4 Klasse\4. Neuronal Networks\Deep_RL_Agent_Gymnasium\LunarLanderRl\models\lunarlander_ep10.pth"

# Überprüfen, ob Modell existiert
if not os.path.exists(model_path):
    print(f"Modell nicht gefunden unter:\n{model_path}")
    exit()

# Umgebung vorbereiten
env = gym.make("LunarLander-v3", render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Modell laden
model = QNetwork(state_size, action_size)
model.load_state_dict(torch.load(model_path))
model.eval()

# Eine Episode abspielen
state, _ = env.reset()
done = False
total_reward = 0
print(f"\nReplay gestartet mit Modell:\n{model_path}\n")

while not done:
    time.sleep(0.03)
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state_tensor)
    action = torch.argmax(q_values).item()
    state, reward, done, truncated, _ = env.step(action)
    total_reward += reward

print(f"\nReplay abgeschlossen – Gesamt-Reward: {total_reward:.2f}")
env.close()
