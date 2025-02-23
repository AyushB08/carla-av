from stable_baselines3 import PPO #PPO
from typing import Callable
import os
from PPO.carenv import CarEnv
import time



print('This is the start of training script')

print('setting folders for logs and models')
models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

print('connecting to env..')

env = CarEnv()

env.reset(seed=None)
print('Env has been reset as part of launch')
model = PPO('MlpPolicy', env, verbose=1,
           learning_rate=0.00005, 
           n_steps=2048,
           batch_size=256,  
           gamma=0.995,  
           gae_lambda=0.98, 
           clip_range=0.1,  
           ent_coef=0.005, 
           vf_coef=1.0, 
           max_grad_norm=0.3, 
           tensorboard_log=logdir)

TIMESTEPS = 1_500_000  
iters = 0
while iters < 6: 
    iters += 1
    print('Iteration ', iters,' is to commence...')
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
    print('Iteration ', iters,' has been trained')
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
