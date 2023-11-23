#%pylab inline
# -*- coding:UTF-8 -*-
import gym
import math
import random
import numpy as np
import pandas as pd
from collections import namedtuple  #
from itertools import count
from copy import deepcopy
from PIL import Image
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


# lot real-time data dynamic display
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# Load neural network related packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T  #图像转换
from torch import FloatTensor, LongTensor, ByteTensor
Tensor = FloatTensor

# load gym environment
env = gym.make('CartPole-v0').unwrapped  # unwrapped is the original game with unlimited steps

# Each record in the relay buffer
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))

#relay buffer
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, *args):
        """Save an interaction
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)  #从buffer中随机抽样batch个样本
    
    def __len__(self):
        return len(self.memory)

#Q网络
class DQN(nn.Module):
    '''
    First: Pay attention to the consistency of the output channel and the input channel. You can't have the first convolutional layer output 4 channels and the second input 6 channels, and you'll get an error.
    Second, it is a little different from the regular python class. You define a Net instance first and then pass in the parameters
    '''
    def __init__(self):
        # For example, the first layer, which we call conv1, is defined as a convolution layer of 3 input channels, 16 output channels, and 5*5 convolution kernel. Same with conv2
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)  #Two-dimensional convolution layer
        self.bn1 = nn.BatchNorm2d(16)  #Add the batch_normalization layer to the list of layers and maintain the mean variance of all mini-batch data
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # A layer of convolution, corrected linear rectification function, activation function
        x = F.relu(self.bn2(self.conv2(x)))  
        x = F.relu(self.bn3(self.conv3(x)))  
        return self.head(x.view(x.size(0), -1)) 

resize = T.Compose([
    T.ToPILImage(),  # transform to PILImage
    T.Scale(40, interpolation=Image.CUBIC),  # reduce or enlarge
    T.ToTensor()  # transform to tensor, (H X W X C) in range(255)=> (C X H X W) in range(1.0)
])

# define the environment
'''

 In the world of the car, on an X-axis, the variable env.x_threshold stores the maximum value of the car coordinate (=2.4), beyond which the world ends. 
Each step() is rewarded with 1 until the last done is True. 
 
env = gym.make() Each env has its own draw window 
Environment needs to be initialized env.reset() 
env.render() opens a drawing window, drawing the current state 
env.step() updates the status each time 
When done, you need to call env.close() to close the drawing window
'''
screen_width = 600

def get_cart_location():
    #
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  #state:(位置x，x加速度, 偏移角度theta, 角加速度)

def get_screen():
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))  # 转化成torch序列（CHW）
    screen = screen[:, 169:320]
    view_width = 320
    cart_location = get_cart_location()
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    screen = screen[:, :, slice_range]
    # 将图像转化成为tensor
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # 调整尺寸
    return resize(screen).unsqueeze(0).type(Tensor)

# 显示其中一次案例
env.reset()
# plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(), 
#     interpolation='none')
# plt.title('一次游戏截取案例');

#行动决策采用 epsilon greedy policy，就是有一定的比例，选择随机行为（否则按照网络预测的最佳行为行事）。这个比例从0.9逐渐降到0.05，按EXP曲线递减：
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

model = DQN()  # 初始化对象
optimizer = optim.RMSprop(model.parameters())  # 设置优化器
memory = ReplayMemory(10000)

steps_done = 0
def select_action(state):
    """选择动作
    """
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold: # 刚开始不采用DQN进行更新，采用随机探索
        #第一块网络：输入(3,40,84)：当前state，输出（2维）：当前state各action的Q function Q^pi(s,a)，选Q值最大的action
        return model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)  # 返回最优的动作
    else:
        return LongTensor([[random.randrange(2)]])

episode_durations = []  # 维持时间长度
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('训练中。。。')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # 平均每100次迭代画出一幅图
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)  #unfold窗口(100*1)，mean(1)对列(100个数)求平均，view(-1)组合拼接成一维数组展示
        means = torch.cat((torch.zeros(99), means))  # 拼接数据
        plt.plot(means.numpy())
    
    plt.pause(0.001)
    if is_ipython:
        display.clear_output(wait=True)  #实时动态展示
        display.display(plt.gcf())


last_sync = 0

def optimize_model():
    """训练函数
    """
    global last_sync
    if len(memory) < BATCH_SIZE: # 如果样本数小于最低批次大小返回 
        return
    # 转化batch
    transitions = memory.sample(BATCH_SIZE)  # 抽样
    batch = Transition(*zip(*transitions))  # 转换成为一批次
    
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, 
                                          batch.next_state)))
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state 
                                     if s is not None]), volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))
    
    # 计算Q(s_t, a)，选择动作
    state_action_values = model(state_batch).gather(1, action_batch)
    
    # 计算下一步的所有动作的价值V（s_{t+1}）
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    
    next_state_values.volatile = False
    # 计算预期的Q值
    expected_state_action_values = (next_state_values * GAMMA) + \
                                    reward_batch
    # 计算Huber loss，损失函数采用smooth_ll_loss
    loss = F.smooth_l1_loss(state_action_values, 
                            expected_state_action_values)
    
    # 优化模型
    optimizer.zero_grad()  # 清理所有参数的梯度。
    loss.backward()  # 反向传播
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)  # 将所有的梯度限制在-1到1之间
    optimizer.step()  # 更新模型的参数

num_episodes = 1  # 迭代次数
for i_episode in range(num_episodes):
    # 初始化环境和状态
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen  # 定义状态，即为当前状态和最后的状态差。
    for t in count():  #感觉和while True一个意思
        # 选择并执行一个动作
        action = select_action(state)
        _, reward, done, _ = env.step(int(action[0, 0]))  # 从环境汇中获取奖励
        reward = Tensor([reward])  # 将奖励转换成为tensor
        
        # 观察新的状态，确定下一个状态（PS：在这一步里面获取了未来信息，引用在资本市场上，未来的状态具有一定的概率分布特征。）
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None
        
        # 将转换保存起来
        memory.push(state, action, next_state, reward)
        
        # 切换到下一状态
        state = next_state
        
        # 优化模型
        optimize_model()
        # 一次游戏结束，就画图显示
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
print('完成')
env.render()
env.close()
