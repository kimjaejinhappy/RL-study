import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 이 줄 추가
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v1", is_slippery=False)

# Q-table 초기화 (16개 상태 x 4개 행동)
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# 하이퍼파라미터
learning_rate = 0.1
discount_factor = 0.99   # gamma — 미래 보상 중요도
epsilon = 1.0            # 탐색 확률 (처음엔 100% 랜덤)
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 2000

rewards_per_episode = []

for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    for step in range(100):
        # epsilon-greedy: 랜덤 탐색 vs Q-table 활용
        if np.random.random() < epsilon:
            action = env.action_space.sample()  # 랜덤 행동
        else:
            action = np.argmax(q_table[state])  # Q값 최대 행동

        next_state, reward, terminated, truncated, _ = env.step(action)

        # Q-table 업데이트 (핵심 수식)
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + discount_factor * np.max(q_table[next_state]) - q_table[state, action]
        )

        state = next_state
        total_reward += reward

        if terminated or truncated:
            break

    # epsilon 줄이기 (점점 덜 탐색)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards_per_episode.append(total_reward)

# 결과 출력
print("학습 완료!")
print(f"마지막 100 에피소드 성공률: {sum(rewards_per_episode[-100:]):.0f}%")
print("\nQ-table:")
print(q_table.reshape(4, 4, 4).round(2))

# 학습 곡선
plt.plot(np.convolve(rewards_per_episode, np.ones(100)/100, mode='valid'))
plt.xlabel("Episode")
plt.ylabel("Success Rate (100-ep average)")
plt.title("FrozenLake Q-Learning")
plt.savefig("week1_rl_basics/result.png")
print("그래프 저장 완료!")