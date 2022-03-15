import gym    
from matplotlib import pyplot as plt
from deepQ import Agent
import numpy as np
import sys


save_model = False
training = False

env = gym.make("Breakout-ram-v0")
observation = env.reset()/256
episodes = 100

agent = Agent(state_dim=128, n_actions=4, lr=1e-4, hidden=128, batch_size=32, load=True)
scores = []
epsilons = []


for episode in range(episodes):
    done = False
    score = 0
    
    while not done:
        #env.render()
        action = agent.choose_action(observation)
        #action = env.action_space.sample()
        new_observation, reward, done, info = env.step(action)
        new_observation = new_observation/256
        if training:
            agent.store_in_memory(observation, new_observation, reward, action)
            agent.learn()
        observation = new_observation
        score += reward
        
    if save_model and score >= 40:
        agent.save_model('Breakout5.pt')
        
    scores.append(score)
    epsilons.append(agent.epsilon)
    print('Episode: %d Score: %d Epsilon: %f Loss: %f' % (episode, score, agent.epsilon, agent.loss))
    observation = env.reset()/256


env.close()


plt.hist(scores)
plt.show()

print(np.mean(scores), np.std(scores), np.max(scores))

sys.exit()
