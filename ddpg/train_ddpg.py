import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from carenv import CarEnv

# TensorBoard callback for custom logging
class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        self.logger.record("train/reward", self.locals["rewards"][0])
        return True

# Initialize environment
env = CarEnv()

# Initialize DDPG model
model = DDPG("CnnPolicy", env, verbose=1, tensorboard_log="./ddpg_tensorboard/")

# Train the model
callback = TensorboardCallback()
model.learn(total_timesteps=100000, log_interval=10, callback=callback)

# Save the model
model.save("ddpg_carla_model")
print("Model saved successfully.")
