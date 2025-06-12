import torch
import gymnasium as gym
from dqn_agent import QNetwork
import time

# Modell laden
model_path = r"C:\Users\Lenovo\Desktop\schule\OneDrive - HTL Anichstrasse\WPG_KISY\4 Klasse\Korber\4 Klasse\4. Neuronal Networks\Deep_RL_Agent_Gymnasium\LunarLanderRl\models\lunarlander_model.pth"  # Pfad anpassen!
env = gym.make("LunarLander-v3")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

model = QNetwork(state_size, action_size)
model.load_state_dict(torch.load(model_path))
model.eval()

# Beispielzustand
state = [0.05, 1.0, -0.1, -0.8, 0.05, 0.2, 0.0, 0.0]
state_tensor = torch.FloatTensor(state).unsqueeze(0)
with torch.no_grad():
    q_values = model(state_tensor)

print("Beispielzustand:", state)
print("Q-Werte:", q_values)
print("Ausgewählte Aktion:", torch.argmax(q_values).item())
