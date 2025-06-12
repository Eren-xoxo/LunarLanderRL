import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")

# Zeige Info über den Beobachtungsraum
print("Observation space:", env.observation_space)

# Starte das Spiel und gib den Anfangszustand aus
state, _ = env.reset()
print("Beispielzustand:", state)

# Spiel läuft kurz für GUI-Screenshot 
import time
for _ in range(100):
    env.step(env.action_space.sample())
    env.render()
    time.sleep(0.05)

env.close()
