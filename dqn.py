import os
import random
import numpy as np
import torch
from config import LEARNING_RATE
from torch.utils.tensorboard import SummaryWriter

checkpoint_dir = 'models'
GAMMA = 0.95
use_cuda = torch.cuda.is_available()

class DQN:
    def __init__(self, model_name, replay_memory, target=False):
        self.online_network = Network()
        self.target_network = Network()
        if use_cuda:
            self.online_network.cuda()
            self.target_network.cuda()
        self.optimizer = torch.optim.RMSprop(
                self.online_network.parameters(), lr=LEARNING_RATE)
        self.steps = 0
        self.episodes = 0
        self.replay_memory = replay_memory
        self.writer = SummaryWriter()
        

    def update_target(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def load_checkpoint(self):
        self.online_network.load_state_dict(torch.load(
                os.path.join(checkpoint_dir, 'online.ckpt'),
                map_location=lambda storage, loc: storage))
        self.target_network.load_state_dict(torch.load(
                os.path.join(checkpoint_dir, 'target.ckpt'),
                map_location=lambda storage, loc: storage))
        

    def save_checkpoint(self, current_iteration):
        torch.save(
                self.online_network.state_dict(),
                os.path.join(checkpoint_dir, 'online.ckpt'))
        torch.save(
                self.target_network.state_dict(),
                os.path.join(checkpoint_dir, 'target.ckpt'))

    def get_q_values(self, states, actions, target=False):
        if use_cuda:
            states = states.cuda()
            actions = actions.cuda()
        if target:
            return self.target_network(states, actions)
        else:
            return self.online_network(states, actions)

    def optimize(self, memory, batch_size=128):

        batch = random.sample(memory, batch_size)
        if use_cuda:
            batch = (torch.cat(b, 0).cuda() for b in list(zip(*batch)))
        else:
            batch = (torch.cat(b, 0) for b in list(zip(*batch)))
        states, next_states, cur_action, reward, done, actions, next_actions = batch
        cur_action = cur_action.unsqueeze(-1)

        online_q = self.get_q_values(states, actions).gather(-1, cur_action)

        with torch.no_grad():
            target_q = self.get_q_values(next_states, next_actions, target=True)
            target_q = reward + (1.0 - done.float()) * GAMMA * target_q.max(1)[0]

        loss = torch.nn.functional.mse_loss(online_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_count_states(self):
        return self.steps

    def increase_count_states(self):
        self.steps += 1
        return self.steps
    
    def get_count_episodes(self):
        return self.episodes

    def increase_count_episodes(self):
        self.episodes += 1
        return self.episodes
    
    def close(self):
        pass
    
    def log_average_speed(self, speed):
        self.writer.add_scalar('avg_speed', speed, self.get_count_episodes())

    def log_testing_speed(self, speed):
        self.writer.add_scalar('test_speed', speed, self.get_count_episodes())

    def log_training_loss(self, loss):
        self.writer.add_scalar('loss', loss, self.get_count_episodes())

    def log_total_frame(self, frame):
        self.writer.add_scalar('frame', frame, self.get_count_episodes())

    def log_terminated(self, terminated):
        pass

    def log_reward(self, reward):
        self.writer.add_scalar('reward', reward, self.get_count_episodes())

    def log_average_test_speed_20(self, test_speed):
        self.writer.add_scalar('test_average_speed_20', test_speed, 0)
        self.writer.add_scalar('test_average_speed_20', test_speed, 1)

    def log_average_test_speed_40(self, test_speed):
        self.writer.add_scalar('test_average_speed_40', test_speed, 0)
        self.writer.add_scalar('test_average_speed_40', test_speed, 1)

    def log_average_test_speed_60(self, test_speed):
        self.writer.add_scalar('test_average_speed_60', test_speed, 0)
        self.writer.add_scalar('test_average_speed_60', test_speed, 1)

    def log_target_network_update(self):
        pass

    def log_q_values(self, q_values):
        self.writer.add_scalar('sum_q_values', q_values, self.get_count_episodes())

    def log_hard_brake_count(self, count):
        self.writer.add_scalar('hard_brake_count', count, self.get_count_episodes())

    def log_action_frequency(self, stats):
        s = float(np.sum(stats)).tolist()
        for i, v in enumerate(s):
            self.writer.add_scalar('action_freq', v, i)

    def log_histogram(self, tag, values, step, bins=1000):
        pass

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_head = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.action_head = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 4),
            torch.nn.ReLU(),
        )
        self.q_value = torch.nn.Sequential(
            torch.nn.Linear(288 + 4, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 5),
        )

    def forward(self, x, action):
        # BHWC -> BCHW
        x = x.permute(0, 3, 1, 2)
        x = self.conv_head(x)
        x = x.reshape(x.size(0), -1)
        action = self.action_head(action)
        x = torch.cat([x, action], -1)
        q_value = self.q_value(x)
        return q_value

