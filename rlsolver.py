import torch
import torchvision
import tensorboard
import torch.nn as nn
import torch.functional as F
from collections import namedtuple
from torch.utils.data import Dataset, DataLoader
import numpy as np
from random import sample
from torch.optim import Optimizer, Adam
from gym import Env
from torch.utils.tensorboard import SummaryWriter
from progress.bar import Bar
from collections import deque
#dataclass to store memories
MEMORY = namedtuple("MEMORY",
    ("STATE", "ACTION", "REWARD", "NEXTSTATE", "DONE")
);
from datetime import datetime
#init tensorboard summary writer for model analysis
global WRITER
global ITERATIONS
global MODEL 
ITERATIONS = 0
def init_tensorboard(name):
    global MODEL
    MODEL = name
    global WRITER
    WRITER = SummaryWriter(f"runs/{name}-{datetime.now().strftime('%H-%M-%S')}")
    
#global TBWRITE 
#TBWRITE = writer.write  #aliasing the writing functions for tensorboard

def fit(dataset: Dataset, model: nn.Module, opt, lossfunc, epochs:int):
    #fits model to dataset
    global WRITER
    global ITERATIONS
    global MODEL
    model.train();
    for E in range(epochs):
        epoch_loss = 0;
        for idx, (x, y) in enumerate(dataset):
            opt.zero_grad();
            prediction = model(x);
            loss = lossfunc(prediction, y);
            loss.backward();
            opt.step();
            epoch_loss += loss.item();
        
        print(f"EPOCH {E+1}/{epochs}:\tLOSS:{epoch_loss}");
    WRITER.add_scalar(
        "loss/iterations",
        
        epoch_loss/len(dataset),
        ITERATIONS

    )
    WRITER.flush()

class MiniBatch(Dataset):
        #pytorch dataset class  for doing inter-episode training
        def __init__(self, X:list, Y: list):
            self.X = X;
            self.Y = Y;

        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, index):
            #returns a datapoint in next(...)
            return (self.X[index], self.Y[index]);

class ReplayBuffer(object):

    def __init__(self, buffer_size=256):
        #self.BUFFER = [];
        #self.MAXLEN = buffer_size;
        self.BUFFER = deque(maxlen=buffer_size)
    def __len__(self):
        return len(self.BUFFER);

    def _pop_min(self, new_memory: MEMORY):
        #if full, this is called to remove the memory episode that possesses the minimum reward in the dataset
        min_idx = 0
        minimum_reward_memory = self.BUFFER[0]
        update1= False
        for idx, memory in enumerate(self.BUFFER[1:]):
            if minimum_reward_memory.REWARD < new_memory.REWARD:
                min_idx = idx;
                minimum_reward_memory = memory;
                update1 = True;
                break
        if not update1:
            return
        print(f"pop min {self.BUFFER[min_idx].REWARD} <- {new_memory.REWARD}:\t idx:\t {min_idx}")
        
        self.BUFFER[min_idx] = new_memory;

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __add__(self, memory: MEMORY):
        #allows usage of the + / =+ operators to append new memory namedTuples
        #if len(self) == self.MAXLEN:
            #self._pop_min(memory);
            #return
        self.BUFFER.append(memory);

    def replay(self,model:nn.Module, GAMMA, minibatch_size):
        sampled_memories = sample(self.BUFFER, min([minibatch_size, len(self)]));
        X = []; Y = [];
        with torch.no_grad():
            for STATE, ACTION, REWARD, NEXTSTATE, DONE in sampled_memories:
                print(STATE)
                qACT = model(STATE)
                print(qACT)
                #q-value interpolation with discount constant 
                qACT[ACTION] = REWARD if DONE else REWARD + GAMMA*torch.max(model(NEXTSTATE)).item()  #bellman equation for estimating Q-Values for each state
                X.append(STATE);
                Y.append(qACT);
        return MiniBatch(X, Y);


def RLSOLVER(model: nn.Module, opt, env: Env, LossFunc,GAMMA, epsilon = 1024, rpbuff_size = 4096, minibatch_size=128, CONVERGING_REWARD_THRESHOLD = 200, REWARD_FUNCTION = lambda reward, state, done: reward):
    global WRITER
    global ITERATIONS
    
    RPBUFFER = ReplayBuffer(buffer_size=rpbuff_size)        #replay buffer object
    ITERATIONS = 0
    
    def act(STATE, epsilon=epsilon):
        
        #action prediction generator function
        model.eval()
        #generator function needs to be called inside of next() to output the value and not the generator object
        if epsilon > 0:
            epsilon -= 1
            print(epsilon)
            yield env.action_space.sample(), epsilon
        else:
            
            yield torch.argmax(model(STATE)).item(), epsilon 


    CONVERGED = False
    #CONVERGED controls training algorithm and signifies 
    while not CONVERGED:
        ITERATIONS += 1
        print(f"ITERATION: {ITERATIONS}")
        #STATE -> state n+1
        #state -> current state n
        #uses this notation to work with namedtuple MEMEORY
        done = False
        new_env = True
        ACCUMULATED_REWARDS = 0
        b = Bar("processing", max = CONVERGING_REWARD_THRESHOLD)
        while not done:
            b.next()
            print()
            if new_env: #flag to initialize new environment
                #grabs current state if reset
                state = torch.Tensor(env.reset())
                new_env = False
            if epsilon <= 0:
                pass
                #env.render()
            ACTION, epsilon = next(act(state, epsilon))       #working with a generator function in python requires next() as it is a list generator
          
            STATE, REWARD, DONE, info = env.step(ACTION)
            STATE = torch.Tensor(STATE)
            
            ACCUMULATED_REWARDS += REWARD
            WRITER.add_scalar(
                f"Current Episode Accumulated Rewards {MODEL}", 
                ACCUMULATED_REWARDS,
                ITERATIONS
            )
            REWARD = REWARD_FUNCTION(REWARD, STATE, DONE)        #parrot function if not defined, for dynamic case-by-case basis dynamic rewards based on state status
            WRITER.add_scalar(
                f"Current State Variable Reward {MODEL}",
                REWARD,
                ITERATIONS
            )
            if ACCUMULATED_REWARDS == CONVERGING_REWARD_THRESHOLD-1:  #evaluates current model
                CONVERGED = True
                print("Model Has Converged")
            new_memory = MEMORY(state,ACTION, REWARD, STATE, DONE)
            RPBUFFER + new_memory      #neat little feature of the magic __add__
            state = STATE
            done = DONE
        b.finish()
        training_set = RPBUFFER.replay(model, GAMMA, minibatch_size)
        if epsilon <= 0:
            fit(training_set, model, opt, LossFunc, epochs = 1)


        
        
        







