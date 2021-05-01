from sdt_github2 import SDT
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from gym import make
from rlsolver import *
from torch.nn import BCELoss, MSELoss, L1Loss

cartpole = make("CartPole-v0")
 # Parameters
input_dim = cartpole.observation_space.shape[0]    # the number of input dimensions
output_dim = cartpole.action_space.n        # the number of outputs (i.e., # classes on MNIST)
depth = 9         # tree depth
lamda = 1e-2           # coefficient of the regularization term
lr = 1e-4              # learning rate
weight_decaly = 5e-4   # weight decay
batch_size = 128       # batch size
epochs = 50            # the number of training epochs
log_interval = 100     # the number of batches to wait before printing logs
use_cuda = False       # whether to use GPU

    # Model and Optimizer
network = SDT(input_dim, output_dim, depth, lamda, use_cuda)
optimizer = Adam(network.parameters(),lr=lr,weight_decay=weight_decaly)
LossFunc = MSELoss()
init_tensorboard("SDQT2")
import rlsolver
from rlsolver import WRITER, ITERATIONS

def fit(dataset: Dataset, model: nn.Module, opt, lossfunc, epochs=3):
    #fits model to dataset
    global ITERATIONS
    dataloader = DataLoader(dataset)
    model.train();
    for E in range(epochs):
        epoch_loss = 0;
        for idx, (x, y) in enumerate(dataloader):
            batch_size = x.size()[0]
            out, penalty = model(x, is_training_data=True)
            loss = lossfunc(out, y)
            loss += penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        print(f"EPOCH {E+1}/{epochs}:\tLOSS:{epoch_loss/len(dataset)}");
    WRITER.add_scalar(
        "loss/iterations",
        
        epoch_loss/len(dataset),
        ITERATIONS

    )
    WRITER.flush()

from rlsolver import MiniBatch
def replaySDT(self,model:nn.Module, GAMMA, minibatch_size):
    sampled_memories = sample(self.BUFFER, min([minibatch_size, len(self)]));
    X = []; Y = [];
    with torch.no_grad():
        for STATE, ACTION, REWARD, NEXTSTATE, DONE in sampled_memories:
            #print(STATE)
            qACT= model(STATE).view(-1)
            #print(qACT)
            #q-value interpolation with discount constant 
            qACT[ACTION] = REWARD if DONE else REWARD + GAMMA*torch.max(model(NEXTSTATE).view(-1)).item()  #bellman equation for estimating Q-Values for each state
            qACT = qACT.view(-1)
            #print(f"STATE: {STATE}")
            #print(f"Q-ACTION: {qACT}")
            X.append(STATE);
            Y.append(qACT);
    return MiniBatch(X, Y);

rlsolver.fit = fit #fit function for SDT but complaint with the RLSOLVER
rlsolver.ReplayBuffer.replay = replaySDT    #replay function for SDT to fit dimension requirements

RLSOLVER(
        model = network,
        opt=optimizer,
        env=cartpole,
        LossFunc=LossFunc,
        GAMMA=0.004,
        REWARD_FUNCTION = lambda REWARD, STATE, DONE: -REWARD/200 if DONE else (REWARD - ((STATE[0]**2/11.52) + (STATE[2]**2/288)))/200,
        epsilon=512,rpbuff_size=1024,minibatch_size=128
    )

