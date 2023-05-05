
#import pybullet
import gym
import pybullet_envs
import numpy as np
from sac_tf2 import Agent
from utils import plot_learning_curve

if __name__== '__main__':
    env = gym.make('InvertedPendulumPyBulletEnv-v0')
    agent = Agent(input_dims=env.observation_space.shape, env=env,
                  n_actions=env.action_space.shape[0])
    n_games = 250

    filename = 'inverted_pendulum.png'
    figure_file = 'plots/'+filename


    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    print(best_score)
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.high)

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        
        print(f'episode {i}, score {score}, avg_score {avg_score}')

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)