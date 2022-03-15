import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy


class DeepQNetwork(nn.Module):
    def __init__(self, state_dim, hidden, n_actions, lr):
        super(DeepQNetwork, self).__init__()
        self.state_dim = state_dim
        self.hidden = hidden
        self.n_actions = n_actions
        
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, n_actions)
        self.relu = nn.ReLU()
        
        self.optimiser = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        
        
    def forward(self, state):
        out = self.relu(self.fc1(state))
        out = self.fc2(out)
        return out



class Agent():
    def __init__(self, state_dim, n_actions, lr, hidden, batch_size, load=False):
        self.epsilon = 0
        self.epsilon_decay = 1e-6
        self.min_epsilon = 0.1
        self.gamma = 0.99
        self.target_network_update = 100
        self.loss = torch.tensor(0)
        
        self.state_dim = state_dim
        self.n_actions = n_actions
        
        if load:
            self.model = torch.load('Breakout40.pt')
            self.model.eval()
        else:
            self.model = DeepQNetwork(state_dim=state_dim, hidden=hidden, n_actions=n_actions, lr=lr).cuda()
            

        # initialise memory
        self.memory_size = 50000
        self.batch_size = batch_size
        self.step_counter = 0
        
        
        if self.step_counter % self.target_network_update == 0:
            self.target_model = copy.deepcopy(self.model)
            for param in self.target_model.parameters():
                param.requires_grad = False
        
        
        self.state_mem = np.zeros((self.memory_size, self.state_dim))
        self.new_state_mem = np.zeros((self.memory_size, self.state_dim))
        self.reward_mem = np.zeros(self.memory_size)
        self.action_mem = np.zeros(self.memory_size)
        
        
    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).cuda()
        if np.random.random() > self.epsilon:
            q_values = self.model.forward(state)
            action = torch.argmax(q_values)
            return action.item()
        else:
            action = np.random.randint(self.n_actions)
            return action
    
    
    def learn(self): # update network
        if self.step_counter < self.memory_size:
            return
        
        # get random batch of memories
        batch_index = np.random.randint(0,high=self.memory_size, size=self.batch_size)
        batches = np.arange(self.batch_size)
        
        state = torch.tensor(self.state_mem[batch_index], dtype=torch.float32).cuda()
        new_state = torch.tensor(self.new_state_mem[batch_index], dtype=torch.float32).cuda()
        reward = torch.tensor(self.reward_mem[batch_index], dtype=torch.float32).cuda()
        actions = self.action_mem[batch_index]
        
        
        # q prediction
        q_pred = self.model.forward(state)[batches, actions]
        
        
        if self.step_counter % self.target_network_update == 0:
            self.target_model = copy.deepcopy(self.model)
            for param in self.target_model.parameters():
                param.requires_grad = False
        
        # q target
        q_target = reward + self.gamma*torch.max(self.target_model.forward(new_state), dim=1)[0]

        
        # loss fucntion and training loop
        self.loss = self.model.loss(q_pred, q_target)
        self.loss.backward()
        self.model.optimiser.step()
        self.model.optimiser.zero_grad()
        
        # decay epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = self.min_epsilon
        
              
    def store_in_memory(self, state, new_state, reward, action):
        index = self.step_counter % self.memory_size
        self.state_mem[index] = state
        self.new_state_mem[index] = new_state
        self.reward_mem[index] = reward
        self.action_mem[index] = action
        self.step_counter += 1
        
        
    def save_model(self, name):
        torch.save(self.model, name)
        






















