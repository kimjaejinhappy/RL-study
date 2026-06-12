import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=False)

# Q-table 초기화
q_table = np.zeros([env.observation_space.n, env.action_space.n])

print("환경 상태 수:", env.observation_space.n)
print("행동 수:", env.action_space.n)
print("Q-table shape:", q_table.shape)
print("세팅 완료!")