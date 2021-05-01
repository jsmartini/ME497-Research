from rlsolver import *
import torch.nn as nn
import torch.functional as F
from torch.nn import ReLU, Softmax
from torch.optim import Adam, RMSprop, SGD
from torch.nn import MSELoss, L1Loss, BCELoss
from torch.utils.tensorboard import SummaryWriter

global MODELPATH
MODELPATH = "model.pkl"

torch.manual_seed(1)
relu = ReLU()
softmax = Softmax()

writer = SummaryWriter("runs/DQN")

class DQN(nn.Module):

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.L1 = nn.Linear(state_size, 100)
        self.L2 = nn.Linear(100, 100)
        self.L3 = nn.Linear(100, 250)
        self.L4 = nn.Linear(250, 500)
        self.L5 = nn.Linear(500, 250)
        self.L6 = nn.Linear(250, 50)
        self.L7 = nn.Linear(50, action_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = self.L1(x)
        out = relu(out)
        out = self.L2(out)
        out = relu(out)
        out = self.L3(out)
        out = relu(out)
        out = self.dropout(out)
        out = relu(out)
        out = self.L4(out)
        out = relu(out)
        out = self.L5(out)
        out = relu(out)
        out = self.L6(out)
        out = self.dropout(out)
        out = relu(out)
        out = self.L7(out)
        out = softmax(out)
        return out


if __name__ == "__main__":
    #cartpole example
    from gym import make
    cartpole = make("CartPole-v0")
    action_size = cartpole.action_space.n
    state_size  = cartpole.observation_space.shape[0]
    network = DQN(state_size, action_size)
    optimizer = Adam(network.parameters(), lr=0.0001)
    LossFunc = MSELoss()
    init_tensorboard("DQN")
    RLSOLVER(
        model = network,
        opt=optimizer,
        env=cartpole,
        LossFunc=LossFunc,
        GAMMA=0.004,
        REWARD_FUNCTION = lambda REWARD, STATE, DONE: -REWARD/200 if DONE else (REWARD - ((STATE[0]**2/11.52) + (STATE[2]**2/288)))/200,
        epsilon=100,rpbuff_size=500,minibatch_size=12
    )
    torch.save(network.state_dict(), MODELPATH)