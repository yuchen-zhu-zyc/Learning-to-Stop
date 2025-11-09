"""Environment module for optimal stopping problems.

This module defines an abstract base class and various concrete implementations
of environments for optimal stopping problems. Each environment represents a
stochastic process where an agent must decide when to stop to minimize expected
costs.

The base Environment class provides an interface for:
- Initial distribution sampling
- State propagation via Markov transition kernels
- Cost computation (stopping, running, and terminal costs)

Concrete implementations include:
- RollaDie: Simple dice rolling environment
- MoveorStay: Binary decision environment
- MoveToRight: Linear state progression
- CongestionDice: Dice with congestion-dependent transitions
- DistCost: Distance-based cost to target distribution
- TowardsUnif2D: 2D grid environment moving toward uniform distribution
- DiceCommonNoise: Dice with common noise affecting transitions
- MatchDistribution2D: 2D grid environment matching target distribution
"""

import abc
import torch
import numpy as np


class Environment(abc.ABC):
    @abc.abstractmethod
    def get_initial_dist(self):
        pass    
    @abc.abstractmethod
    def propagate(self):
        pass
    @abc.abstractmethod
    def stop_cost(self):
        pass
    def get_initial_dist(self, batch):
        alpha = torch.ones(self.state_dim)
        mu_0 = torch.distributions.Dirichlet(alpha).sample((batch,)).to(self.device)
        nu_cont = mu_0
        nu_stop = torch.zeros_like(mu_0).to(self.device)
        return nu_cont, nu_stop
    
    def get_initial_dist_detailed(self, batch):
        alpha = torch.ones(2 * self.state_dim)
        nu_all = torch.distributions.Dirichlet(alpha).sample((batch,)).to(self.device)
        nu_cont, nu_stop = nu_all[:, :self.state_dim], nu_all[:, self.state_dim:]
        return nu_cont, nu_stop
    
    def get_test_dist(self):
        mu_0 = self.test_mu0
        nu_cont = mu_0.to(self.device)
        nu_stop = torch.zeros_like(mu_0).to(self.device)
        return nu_cont, nu_stop
    
    def propagate(self, nu_stop, nu_cont, stop_prob):
        next_nu_stop = nu_stop + stop_prob * nu_cont
        next_nu_cont = torch.matmul(nu_cont * (1 - stop_prob), self.P)
        return next_nu_stop, next_nu_cont
    
class RollaDie(Environment):
    def __init__(self, sides = 6, T = 6, initial_mu = None, device = "cuda:0"):
        self.sides = sides
        self.T = T
        self.device = device
        self.P = 1/self.sides * torch.ones((self.sides, self.sides)).to(device)
        self.initial_mu = initial_mu if initial_mu is not None else 1/self.sides * torch.ones(self.sides).to(device)
        
        self.state_dim = self.sides
        self.test_mu0 = self.initial_mu
    
    def stop_cost(self, mu, time):
        batch = mu.shape[0]
        cost = torch.arange(1, 1 + self.sides).reshape(1, -1).repeat(batch, 1)
        stop_cost = cost.float().to(self.device)
        running_cost = torch.zeros_like(mu).to(self.device)
        terminal_cost = torch.zeros_like(mu).to(self.device)  
        return stop_cost, running_cost, terminal_cost
    
class MoveorStay(Environment):    
    def __init__(self, T = 2, initial_mu = None, device = "cuda:0"):

        self.T = T
        self.device = device
        self.P = torch.tensor([[0, 1], [1, 0]]).float().to(device)
        self.state_dim = 2
        self.initial_mu = initial_mu if initial_mu is not None else 1/self.state_dim * torch.ones(self.state_dim).to(device)
        self.test_mu0 = self.initial_mu
    
    def stop_cost(self, mu, time):
        cost = torch.where(mu >= 0.5, torch.tensor([5]), torch.tensor([1])).to(self.device)
        stop_cost = cost.float().to(self.device)
        
        running_cost = torch.zeros_like(mu).to(self.device)
        terminal_cost = torch.zeros_like(mu).to(self.device)
        
        return stop_cost, running_cost, terminal_cost
        
class MoveToRight(Environment):
    
    def __init__(self, T = 4, state_dim = 5, initial_mu = None, device = "cuda:0"):
        self.T = T
        self.state_dim = state_dim
        self.device = device
        self.P = torch.zeros((state_dim, state_dim)).float()
        for i in range(state_dim - 1):
            self.P[i, i + 1] = 1.0
        self.P[state_dim - 1, state_dim - 1] = 1.0  # Stay at N if already there
        self.P = self.P.to(device)
        
        self.initial_mu = initial_mu if initial_mu is not None else torch.tensor([1.0] + [0.0] * (state_dim - 1)).to(self.device)
        self.test_mu0 = self.initial_mu
        
    def stop_cost(self, mu, time):
        stop_cost = mu.float().to(self.device)
        running_cost = torch.zeros_like(mu).to(self.device)
        terminal_cost = torch.zeros_like(mu).to(self.device)
        
        return stop_cost, running_cost, terminal_cost
        
class CongestionDice(Environment):
    def __init__(self, sides = 6, T = 6, initial_mu = None, device = "cuda:0"):
        self.sides = sides
        self.T = T
        self.device = device
        self.initial_mu = initial_mu if initial_mu is not None else 1/self.sides * torch.ones(self.sides).to(device)
        
        self.state_dim = self.sides
        self.test_mu0 = self.initial_mu
    
    def stop_cost(self, mu, time):
        batch = mu.shape[0]
        cost = torch.arange(1, 1 + self.sides).reshape(1, -1).repeat(batch, 1)
        stop_cost = cost.float().to(self.device)
        
        running_cost = torch.zeros_like(mu).to(self.device)
        terminal_cost = torch.zeros_like(mu).to(self.device)
        
        return stop_cost, running_cost, terminal_cost
    
    def nonlinear_markov_kernel(self,mu_batch, C_cong = 0.8):
        batch_size, N = mu_batch.shape
        C = C_cong  # As given in the problem statement

        # Calculate the diagonal elements (staying probabilities)
        staying_prob = (1 / N) * (1 + C * mu_batch)  # Shape: (batch_size, N)
        P_batch = torch.zeros((batch_size, N, N), device=mu_batch.device)
        P_batch[:, range(N), range(N)] = staying_prob
        
        remaining_prob = 1 - staying_prob  # Shape: (batch_size, N)
        uniform_prob = remaining_prob / (N - 1)  # Shape: (batch_size, N)
        P_batch += uniform_prob.unsqueeze(2).expand(-1, -1, N)  # Broadcasting to shape (batch_size, N, N)
        P_batch[:, range(N), range(N)] -= uniform_prob  # Shape: (batch_size, N)
        return P_batch.to(self.device)
    
    def propagate(self, nu_stop, nu_cont, stop_prob):
        mu = nu_cont + nu_stop
        P = self.nonlinear_markov_kernel(mu)
        next_nu_stop = nu_stop + stop_prob * nu_cont
        next_nu_cont = nu_cont * (1 - stop_prob)
        next_nu_cont = torch.bmm(next_nu_cont.unsqueeze(1), P).squeeze(1)
        return next_nu_stop, next_nu_cont
    
class DistCost(Environment):
    def __init__(self, T = 4, state_dim = 5, initial_mu = None, target_dist = None, device = "cuda:0"):
        self.T = T
        self.state_dim = state_dim
        self.device = device
        P = torch.zeros((state_dim, state_dim)).float()
        for i in range(state_dim):
            if i == 0:
                P[i, i] = 0.75  # Probability of staying
                P[i, i + 1] = 0.25  # Probability of moving to the right
            elif i == state_dim - 1:
                P[i, i] = 0.75  # Probability of staying
                P[i, i - 1] = 0.25  # Probability of moving to the left
            else:
                P[i, i] = 0.5  # Probability of not moving
                P[i, i - 1] = 0.25  # Probability of moving to the left
                P[i, i + 1] = 0.25  # Probability of moving to the right

        self.P = P
        self.P = self.P.to(device)
        
        self.initial_mu = initial_mu if initial_mu is not None else torch.tensor([1.0] + [0.0] * (state_dim - 1)).to(self.device)
        self.target_dist = target_dist if target_dist is not None else 1/state_dim * torch.tensor([1] * state_dim)
        self.target_dist = self.target_dist.to(self.device)
        self.test_mu0 = self.initial_mu.to(self.device)
        
    def stop_cost(self, mu, time):
        squared_diff = (mu - self.target_dist) ** 2
        rowwise_mse = squared_diff.sum(dim=1, keepdim=True)  # Shape (B, 1)
        stop_cost = rowwise_mse.expand(-1, mu.size(1)).to(self.device)  # Shape (B, N)
        
        running_cost = torch.zeros_like(mu).to(self.device)
        terminal_cost = torch.zeros_like(mu).to(self.device)
        
        return stop_cost, running_cost, terminal_cost

class TowardsUnif2D(Environment):
    def __init__(self, T = 4, grid_dim = 5, initial_mu = None, device = "cuda:0"):
        self.T = T
        self.grid_dim = grid_dim
        self.state_dim = grid_dim ** 2
        self.device = device
        self.P = torch.zeros((grid_dim, grid_dim)).float()
        for i in range(grid_dim - 1):
            self.P[i, i + 1] = 1.0
        self.P[grid_dim - 1, grid_dim - 1] = 1.0  # Stay at N if already there
        self.P = self.P.to(device)
        
        if initial_mu is not None:
            self.initial_mu = initial_mu.to(device)
        else:
            predefined_mu = torch.tensor([1.0] + [0.0] * (grid_dim - 1)).repeat(grid_dim).reshape(1, -1)
            self.initial_mu = predefined_mu.to(device)
            
        self.test_mu0 = self.initial_mu
        
    def stop_cost(self, mu, time):
        
        running_cost = torch.zeros_like(mu).to(self.device)
        stop_cost = mu.float()
        stop_cost = stop_cost.to(self.device)
        
        terminal_cost = torch.zeros_like(mu).to(self.device)
        return stop_cost, running_cost, terminal_cost
    
    def propagate(self, nu_stop, nu_cont, stop_prob):
        next_nu_stop = nu_stop + stop_prob * nu_cont
        next_nu_cont = nu_cont * (1 - stop_prob)
        next_nu_cont = next_nu_cont.reshape(-1, self.grid_dim, self.grid_dim)
        next_nu_cont = torch.matmul(next_nu_cont, self.P).reshape(-1, self.state_dim)
        return next_nu_stop, next_nu_cont
    
class DiceCommonNoise(Environment):
    
    def __init__(self, sides = 6, T = 15, running_cst = 0.05, initial_mu = None, device = "cuda:0"):
        self.sides = sides
        self.T = T
        self.device = device
        self.P = 1/self.sides * torch.ones((self.sides, self.sides)).to(device)
        self.initial_mu = initial_mu if initial_mu is not None else 1/self.sides * torch.ones(self.sides).to(device)
        
        self.state_dim = self.sides
        self.test_mu0 = self.initial_mu
        
        self.running_cst = running_cst
    
    def stop_cost(self, mu, time):        
        running_cost = self.running_cst * torch.ones_like(mu)
        running_cost = running_cost.to(self.device)
        
        stop_cost = torch.zeros_like(mu)
        stop_cost = stop_cost.to(self.device)
        
        if time == self.T:
            squared_diff = (mu) ** 2
            rowwise_mse = squared_diff.sum(dim=1, keepdim=True)  # Shape (B, 1)
            terminal_cost = rowwise_mse.expand(-1, mu.size(1)).to(self.device)  # Shape (B, N)
        else:
            terminal_cost = torch.zeros_like(mu)
            terminal_cost = terminal_cost.to(self.device)
        return stop_cost, running_cost, terminal_cost
    
    def sample_common_noise(self, batch):
        return torch.randint(0, self.sides, (batch,)).to(self.device)
    
    def nonlinear_markov_kernel(self, commom_noise):
        N = self.sides
        B = commom_noise.size(0)
        P_batch = torch.zeros((B, N, N), device=self.device)
        
        for i in range(B):
            target_state = commom_noise[i].item()
            P_batch[i, :, target_state] = 1.0
        return P_batch.to(self.device)
        
    def propagate(self, nu_stop, nu_cont, stop_prob):
        batch = nu_cont.shape[0]
        common_noise = self.sample_common_noise(batch)
        P = self.nonlinear_markov_kernel(common_noise)
        next_nu_stop = nu_stop + stop_prob * nu_cont
        next_nu_cont = nu_cont * (1 - stop_prob)
        next_nu_cont = torch.bmm(next_nu_cont.unsqueeze(1), P).squeeze(1)
        return next_nu_stop, next_nu_cont
      
class MatchDistribution2D(Environment):
    def __init__(self, T = 30, grid_dim = 4, transit_mode = "stay", initial_mu = None, target_mu = None, running_cst = 0.05, common_noise = False, device = "cuda:0"):
        self.T = T
        self.grid_dim = grid_dim
        self.state_dim = grid_dim ** 2
        self.device = device
        self.P = self.create_transition_matrix() if transit_mode == "stay" else self.create_transition_matrix_2()
        self.P = self.P.to(device)
        self.common_noise = common_noise
        
        if self.common_noise:
            self.nonlinear_P = self.prepare_nonlinear_markov_kernel()
        
        if initial_mu is not None:
            self.initial_mu = initial_mu.to(device)
        else:
            predefined_mu = torch.tensor([1.0] + [0.0] * (grid_dim - 1)).repeat(grid_dim).reshape(1, -1)
            self.initial_mu = predefined_mu.to(device)
            
        self.test_mu0 = self.initial_mu
        
        self.running_cst = running_cst
        self.target_mu = target_mu.to(device)
        
    def propagate(self, nu_stop, nu_cont, stop_prob):
        if not self.common_noise:
            next_nu_stop = nu_stop + stop_prob * nu_cont
            next_nu_cont = torch.matmul(nu_cont * (1 - stop_prob), self.P)
        else:
            batch = nu_cont.shape[0]
            common_noise = self.sample_common_noise(batch)
            P = self.nonlinear_P[common_noise.to(torch.int64).reshape(-1)]
            next_nu_stop = nu_stop + stop_prob * nu_cont
            next_nu_cont = nu_cont * (1 - stop_prob)
            next_nu_cont = torch.bmm(next_nu_cont.unsqueeze(1), P).squeeze(1)    
        return next_nu_stop, next_nu_cont
    
    def stop_cost(self, mu, time):
        
        if time == self.T:
            squared_diff = (mu - self.target_mu) ** 2
            rowwise_mse = squared_diff.sum(dim=1, keepdim=True)  # Shape (B, 1)
            terminal_cost = rowwise_mse.expand(-1, mu.size(1)).to(self.device)  # Shape (B, N)
        else:
            terminal_cost = torch.zeros_like(mu)
            terminal_cost = terminal_cost.to(self.device)
        
        running_cost = self.running_cst * torch.ones_like(mu).to(self.device)
        stop_cost = torch.zeros_like(mu).to(self.device)
        
        return stop_cost, running_cost, terminal_cost
    def create_transition_matrix(self):
        num_states = self.state_dim
        N = self.grid_dim
        P = torch.zeros((num_states, num_states), device=self.device)
        for x in range(N):
            for y in range(N):
                current_state = x * N + y
                transitions = []
                # Probability of staying still
                stay_prob = 0.2
                # Probability of moving left
                if y > 0:
                    left_state = x * N + (y - 1)
                    transitions.append((left_state, 0.2))
                else:
                    stay_prob += 0.2
                # Probability of moving right
                if y < N - 1:
                    right_state = x * N + (y + 1)
                    transitions.append((right_state, 0.2))
                else:
                    stay_prob += 0.2
                # Probability of moving up
                if x > 0:
                    up_state = (x - 1) * N + y
                    transitions.append((up_state, 0.2))
                else:
                    stay_prob += 0.2
                # Probability of moving down
                if x < N - 1:
                    down_state = (x + 1) * N + y
                    transitions.append((down_state, 0.2))
                else:
                    stay_prob += 0.2
                
                # Set the probabilities in the transition matrix
                P[current_state, current_state] = stay_prob
                for state, prob in transitions:
                    P[current_state, state] = prob
        return P
    
    def create_transition_matrix_2(self):
        num_states = self.state_dim
        N = self.grid_dim
        P = torch.zeros((num_states, num_states), device=self.device)
        for x in range(N):
            for y in range(N):
                current_state = x * N + y
                transitions = []
                # Check possible moves and their probabilities
                if y > 0:  # Move left
                    left_state = x * N + (y - 1)
                    transitions.append(left_state)
                if y < N - 1:  # Move right
                    right_state = x * N + (y + 1)
                    transitions.append(right_state)
                if x > 0:  # Move up
                    up_state = (x - 1) * N + y
                    transitions.append(up_state)
                if x < N - 1:  # Move down
                    down_state = (x + 1) * N + y
                    transitions.append(down_state)
                
                num_transitions = len(transitions)
                if num_transitions > 0:
                    prob = 1.0 / num_transitions
                else:
                    prob = 1.0
                
                # Set the probabilities in the transition matrix
                for state in transitions:
                    P[current_state, state] = prob
                
                # Probability of staying still is any remaining probability
                P[current_state, current_state] = prob if num_transitions == 0 else 0
        return P
    def sample_common_noise(self, batch):
        return torch.randint(0, self.state_dim, (batch, 1)).to(self.device)
    

    def prepare_nonlinear_markov_kernel(self):
        common_noise = torch.arange(self.state_dim, device=self.P.device).reshape(-1, 1)
        B = common_noise.shape[0]
        P_batch = self.P.unsqueeze(0).repeat(B, 1, 1)
        N = self.grid_dim
        batch_obstacles = common_noise
        for b in range(B):
            obstacles = batch_obstacles[b]
            obstacles_set = set(obstacles.tolist())
            for obs in obstacles:
                x, y = obs // N, obs % N
                neighbors = []
                if y > 0:  # Neighbor to the left
                    neighbors.append(x * N + (y - 1))
                if y < N - 1:  # Neighbor to the right
                    neighbors.append(x * N + (y + 1))
                if x > 0:  # Neighbor above
                    neighbors.append((x - 1) * N + y)
                if x < N - 1:  # Neighbor below
                    neighbors.append((x + 1) * N + y)
        
                for neighbor in neighbors:
                    current_state = neighbor.item()
                    x, y = current_state // N, current_state % N
                    transitions = []
                    # Check possible moves and their probabilities
                    if y > 0 and (x * N + (y - 1)) not in obstacles_set:  # Move left
                        left_state = x * N + (y - 1)
                        transitions.append(left_state)
                    if y < N - 1 and (x * N + (y + 1)) not in obstacles_set:  # Move right
                        right_state = x * N + (y + 1)
                        transitions.append(right_state)
                    if x > 0 and ((x - 1) * N + y) not in obstacles_set:  # Move up
                        up_state = (x - 1) * N + y
                        transitions.append(up_state)
                    if x < N - 1 and ((x + 1) * N + y) not in obstacles_set:  # Move down
                        down_state = (x + 1) * N + y
                        transitions.append(down_state)
                    num_transitions = len(transitions)
                    if num_transitions > 0:
                        prob = 1.0 / num_transitions
                    else:
                        prob = 1.0
                    # Set the probabilities in the transition matrix
                    P_batch[b, current_state, :] = 0
                    for state in transitions:
                        P_batch[b, current_state, state] = prob
                    # Probability of staying still is any remaining probability
                    P_batch[b, current_state, current_state] = prob if num_transitions == 0 else 0
        return P_batch
