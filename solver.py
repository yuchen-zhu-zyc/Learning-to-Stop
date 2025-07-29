import torch
import numpy as np
import matplotlib.pyplot as plt
from network import *
import abc
import torch.distributions as dist
from tqdm import tqdm
from plot_util import plot_evolution, plot_stop_prob, plot_evolution_2D, plot_stop_prob_2D
from environment import *

import pdb

def debug():
    pdb.set_trace()


class DirectApproach:
    def __init__(self, problem, batch = 128, fn = "test", load = None, synchron = False, multi_d = False, writer = None, 
                 hidden = 512, num_res_blocks = 2):
        self.problem = problem
        self.device = problem.device
        self.batch = batch
        self.synchron = synchron
        self.multi_d = multi_d
        
        self.writer = writer
        self.wandb = writer is not None
        
        self.hidden = hidden
        self.num_res_blocks = num_res_blocks
        
        if load is not None:
            self.decision_net= torch.load(load)
        else:
            self.decision_net = self._build_decision_net()
        
        self.decision_net = self.decision_net.to(self.device)
        
        self.taskname = f'DR_{self.problem.__class__.__name__ }' if not self.synchron \
                            else f'DR_{self.problem.__class__.__name__ }_synchron'
        self.runname = f'{self.taskname}_{fn}'   
    
    def _build_decision_net(self):
        if not self.synchron:
            decision_net = PolicyStateEmbT(state_num = self.problem.state_dim, nu_dim = 2 * self.problem.state_dim, out_dim = 1, 
                                        hidden_dim = self.hidden, num_res_blocks = self.num_res_blocks)
        else:
            decision_net = PolicyDistT(nu_dim = 2 * self.problem.state_dim, out_dim=1, hidden_dim=self.hidden, num_res_blocks=self.num_res_blocks)
            
        num_param = sum(p.numel() for p in decision_net.parameters() if p.requires_grad)
        print(f'Number of parameters: {num_param}')
        return decision_net
    
    def train(self, n_iter = 1000, lr = 1e-1):
        loss_track_train = np.zeros((n_iter + 1))
        loss_track_test = np.zeros((n_iter + 1))
        
        optimizer = torch.optim.AdamW(self.decision_net.parameters(), lr = lr)
        
        _n_iter = tqdm(range(n_iter + 1), desc="Training iteration..") if not self.wandb else range(n_iter + 1)
        for i in _n_iter:
            # nu_cont_0, nu_stop_0 = self.problem.get_initial_dist_detailed(self.batch)
            nu_cont_0, nu_stop_0 = self.problem.get_initial_dist(self.batch)
            
            optimizer.zero_grad()
            loss = self.compute_loss(self.decision_net, nu_cont_0, nu_stop_0)
            loss.backward()
            optimizer.step()
            test_loss = self.evaluate(self.decision_net, print_decision = False, pure=False, plot_bar=False)
            train_loss, test_loss = loss.detach().cpu().item(), test_loss.detach().cpu().item()
            loss_track_train[i] = train_loss
            loss_track_test[i] = test_loss
            
            
            self.writer.add_scalar(i, 'train_loss', train_loss)
            self.writer.add_scalar(i, 'test_loss', test_loss)
            
            if i % 1000 == 0:
                print(f'Iteration {i}, train loss = {train_loss}')
                print(f'Iteration {i}, test loss = {test_loss}')
            torch.cuda.empty_cache()
                
        fn_name = self.runname + ".pth"
        torch.save(self.decision_net, f"paper_results/direct/{fn_name}")
        np.save(f"paper_results/direct/{self.runname}_trainloss.npy", loss_track_train)
        np.save(f"paper_results/direct/{self.runname}_testloss.npy", loss_track_test)
                
        return self.decision_net
                
                
    def compute_loss(self, decision_net, nu_cont_0, nu_stop_0):
        batch = nu_cont_0.shape[0]
        nu_cont = nu_cont_0
        nu_stop = nu_stop_0
        mu = nu_cont + nu_stop
        max_T = self.problem.T
        total_loss = 0
        # print("new iter")
        for t in range(max_T + 1):
            if t == max_T:
                stop_prob = torch.ones((batch, self.problem.state_dim)).float().to(self.device)
            else:
                stop_prob = self.get_decision_prob(decision_net, nu_cont, nu_stop, t, synchron=self.synchron)
            mu = nu_stop + nu_cont
            stop_cost_vec, running_cost_vec, terminal_cost_vec = self.problem.stop_cost(mu, time = t)
            stop_cost = stop_cost_vec * nu_cont * stop_prob
            running_cost = running_cost_vec * nu_cont * (1 - stop_prob)
            terminal_cost = terminal_cost_vec
            total_loss += torch.sum(stop_cost + running_cost + terminal_cost)
            if t != max_T:
                nu_stop, nu_cont = self.problem.propagate(nu_stop, nu_cont, stop_prob)
        total_loss = total_loss / batch
        return total_loss
    
    def evaluate(self, decision_net, print_decision = True, pure = False, plot_bar = False, return_dist = False):
        nu_cont, nu_stop = self.problem.get_test_dist()
        nu_cont_list, nu_stop_list = [nu_cont], [nu_stop]
        stop_prob_list = []
        batch = nu_cont.shape[0]
        with torch.no_grad():
            total_loss = 0
            for t in range(self.problem.T + 1):
                if t == self.problem.T:
                    stop_prob = torch.ones((batch, self.problem.state_dim)).float().to(self.device)
                else:
                    stop_prob = self.get_decision_prob(decision_net, nu_cont, nu_stop, t, pure=pure, synchron = self.synchron)
                mu = nu_stop + nu_cont
                stop_cost_vec, running_cost_vec, terminal_cost_vec = self.problem.stop_cost(mu, time = t)
                stop_cost = stop_cost_vec * nu_cont * stop_prob
                running_cost = running_cost_vec * nu_cont * (1 - stop_prob)
                terminal_cost = terminal_cost_vec       
                cost = torch.sum(stop_cost + running_cost + terminal_cost)
                total_loss += cost
                
                if t < self.problem.T:
                    stop_prob_list.append(stop_prob.detach().cpu().numpy())

                if print_decision:
                    print(f"t = {t}, prob: ", stop_prob.detach().cpu())
                    print(f"t = {t}, cost: ", cost.detach().cpu().item())
                # if t != self.problem.T:
                nu_stop, nu_cont = self.problem.propagate(nu_stop, nu_cont, stop_prob)
                nu_cont_list.append(nu_cont)
                nu_stop_list.append(nu_stop)
        
        stop_prob_arr = np.concatenate(stop_prob_list, axis = 0)
        nu_cont_arr = torch.cat(nu_cont_list, dim = 0).unsqueeze(1).detach().cpu().numpy()
        nu_stop_arr = torch.cat(nu_stop_list, dim = 0).unsqueeze(1).detach().cpu().numpy()
        nu_dist = np.concatenate([nu_stop_arr, nu_cont_arr], axis = 1)
        
        if plot_bar:
            fn1 = f'paper_results/figure/{self.runname}_bar.pdf'
            benchmark_value = getattr(self.problem, 'target_dist', None)
            if benchmark_value is not None:
                benchmark_value = benchmark_value.reshape(-1).cpu().numpy()
                
            if not self.multi_d:
                fig = plot_evolution(nu_dist, fn= fn1, benchmark=benchmark_value, return_fig = self.wandb)
            else:
                grid_dim = self.problem.grid_dim
                fig = plot_evolution_2D(nu_dist.reshape(-1, 2, grid_dim, grid_dim), fn= fn1, return_fig=self.wandb)
            
            if self.wandb:
                self.writer.add_plot_image(key = f'{self.runname}_bar', fig = fig, step = None)
            
                
            fn2 = f'paper_results/figure/{self.runname}_prob.pdf'
            if not self.multi_d:
                fig = plot_stop_prob(stop_prob_arr, fn2, return_fig=self.wandb)   
            else:
                grid_dim = self.problem.grid_dim
                fig = plot_stop_prob_2D(stop_prob_arr.reshape(-1, grid_dim, grid_dim), fn2, return_fig=self.wandb)
                
            if self.wandb:
                self.writer.add_plot_image(key = f'{self.runname}_prob', fig = fig, step = None)
        
        if not return_dist:
            return total_loss / batch
        else:
            return total_loss / batch, nu_dist
        
    def get_decision_prob(self, decision_net, nu_cont, nu_stop, t, pure = False, synchron = False):
        batch = nu_cont.shape[0]
        state_dim = self.problem.state_dim
        
        state = torch.arange(state_dim).reshape(1, -1, 1).repeat(batch, 1, 1)
        nu_cont_rep = nu_cont.reshape(batch, 1, -1).repeat(1, state_dim, 1)
        nu_stop_rep = nu_stop.reshape(batch, 1, -1).repeat(1, state_dim, 1)
        
        aggregated_nu = torch.cat([nu_cont_rep, nu_stop_rep], dim = -1).reshape(batch * state_dim, -1).to(self.device)
        aggregated_state = state.reshape(batch * state_dim).to(self.device)
        tt = torch.tensor([t]).float().repeat(batch * state_dim).to(self.device)
        
        if not synchron:
            stop_prob = decision_net(aggregated_state, aggregated_nu, tt).reshape(batch, state_dim).to(self.device)
        else:
            stop_prob = decision_net(aggregated_nu, tt).reshape(batch, state_dim).to(self.device)
        if pure:
            stop_prob = torch.where(stop_prob > 0.5, torch.tensor(1.0), torch.tensor(0.0)).to(self.device)
        return stop_prob
        
        
class DPP:
    def __init__(self, problem, batch, fn = "test", load = None, synchron = False, multi_d = False, writer = None, 
                 hidden = 512, num_res_blocks = 2):
        self.problem = problem
        self.batch = batch
        self.T = problem.T
        self.device = problem.device
        self.multi_d = multi_d
        
        self.synchron = synchron
        self.writer = writer
        self.wandb = writer is not None
        
        self.hidden = hidden
        self.num_res_blocks = num_res_blocks
        
        if load is not None:
            self.decision_net_list = torch.load(load)
        else:
            self.decision_net_list = self._build_decision_net()
        
        self.taskname = f'DPP_{self.problem.__class__.__name__ }' if not self.synchron \
                            else f'DPP_{self.problem.__class__.__name__ }_synchron'
        self.runname = f'{self.taskname}_{fn}'   
        
    
    def _build_decision_net(self):
        decision_net_list = []
        # 0 to T - 1
        for t in range(self.T):
            if not self.synchron:
                decision_net_list.append(PolicyStateEmb(state_num = self.problem.state_dim, nu_dim = 2 * self.problem.state_dim, out_dim = 1, 
                                           hidden_dim = self.hidden, num_res_blocks = self.num_res_blocks))
            else:
                decision_net_list.append(PolicyDist(nu_dim=2 * self.problem.state_dim, out_dim=1, hidden_dim=self.hidden, num_res_blocks=self.num_res_blocks))
            
        num_param = sum(p.numel() for p in decision_net_list[0].parameters() if p.requires_grad)
        print(f'Number of parameters: {num_param}')
        return decision_net_list
    
    def train(self, n_iter = 1000, lr = 1e-4, copy_prev = False):
        loss_track_train = np.zeros((self.T, n_iter + 1))
        loss_track_test = np.zeros((self.T, n_iter + 1))
        
        for t in range(self.T - 1, -1, -1):
            print("Start training at time: ", t)
            decision_net = self.decision_net_list[t]
            if t < self.T - 1:
                if copy_prev:
                    decision_net.load_state_dict(self.decision_net_list[t + 1].state_dict())
                    
            decision_net.train()
            decision_net.to(self.device)
            
            optimizer = torch.optim.AdamW(decision_net.parameters(), lr = lr)
            _n_iter = tqdm(range(n_iter + 1), desc=f"Training iteration at t = {t}..") if not self.wandb else range(n_iter + 1)
            for i in _n_iter:
                if t == 0:
                    nu_cont_0, nu_stop_0 = self.problem.get_initial_dist(self.batch)
                else:
                    nu_cont_0, nu_stop_0 = self.problem.get_initial_dist_detailed(self.batch)
                # nu_cont_0, nu_stop_0 = self.problem.get_initial_dist_detailed(self.batch)
                    
                nu_cont_0, nu_stop_0 = nu_cont_0.to(self.device), nu_stop_0.to(self.device)
                optimizer.zero_grad()
                loss = self.compute_loss_t(decision_net, nu_cont_0, nu_stop_0, t)
                loss.backward()
                optimizer.step()
                
                test_loss = self.evaluate(self.decision_net_list, print_decision = False, pure=False, plot_bar=False, return_dist=False)
                
                train_loss, test_loss = loss.detach().cpu().item(), test_loss.detach().cpu().item()
                loss_track_train[t, i] = train_loss
                loss_track_test[t, i] = test_loss
                
                self.writer.add_scalar( (self.T -1 - t) * (n_iter + 1) + i, f'train_loss_{t}', train_loss)
                self.writer.add_scalar( (self.T -1 - t) * (n_iter + 1) + i, f'test_loss_{t}', test_loss)
                
                if i % 5000 == 0:
                    print(f'Time {t}, Iteration {i}, train loss = {loss.item()}')
                    # test_loss = self.evaluate(decision_net, print_decision = False)
                    # print(f'Time {t}, Iteration {i}, test loss = {test_loss.item()}')
        
        fn_name = self.runname + ".pth"
        torch.save(self.decision_net_list, f"paper_results/dpp/{fn_name}")
        np.save(f"paper_results/dpp/{self.runname}_trainloss.npy", loss_track_train)
        np.save(f"paper_results/dpp/{self.runname}_testloss.npy", loss_track_test)
        
        return self.decision_net_list            
        
                    
    def evaluate(self, decision_net_list, print_decision = True, pure = False, plot_bar = False, return_dist = False):
        nu_cont, nu_stop = self.problem.get_test_dist()
        nu_cont_list, nu_stop_list = [nu_cont], [nu_stop]
        stop_prob_list = []
        
        batch = nu_cont.shape[0]
        with torch.no_grad():
            total_loss = 0
            for t in range(self.problem.T + 1):
                if t == self.problem.T:
                    stop_prob = torch.ones((batch, self.problem.state_dim)).float().to(self.device)
                else:
                    decision_net = decision_net_list[t].to(self.device)
                    stop_prob = self.get_decision_prob(decision_net, nu_cont, nu_stop, pure=pure, synchron = self.synchron)
                if t < self.problem.T:
                    stop_prob_list.append(stop_prob)
                
                mu = nu_stop + nu_cont
                stop_cost_vec, running_cost_vec, terminal_cost_vec = self.problem.stop_cost(mu, time = t)
                stop_cost = stop_cost_vec * nu_cont * stop_prob
                running_cost = running_cost_vec * nu_cont * (1 - stop_prob)
                terminal_cost = terminal_cost_vec       
                cost = torch.sum(stop_cost + running_cost + terminal_cost)
                total_loss += cost
                
                if print_decision:
                    print(f"t = {t}, cost: ", cost.detach().cpu().item())
                nu_stop, nu_cont = self.problem.propagate(nu_stop, nu_cont, stop_prob)
                nu_cont_list.append(nu_cont)
                nu_stop_list.append(nu_stop)
                
        stop_prob_arr = torch.cat(stop_prob_list, dim = 0).detach().cpu().numpy()          
        nu_cont_arr = torch.cat(nu_cont_list, dim = 0).unsqueeze(1).detach().cpu().numpy()
        nu_stop_arr = torch.cat(nu_stop_list, dim = 0).unsqueeze(1).detach().cpu().numpy()
        nu_dist = np.concatenate([nu_stop_arr, nu_cont_arr], axis = 1)
        
        if plot_bar:
            fn1 = f'paper_results/figure/{self.runname}_bar.pdf'
            benchmark_value = getattr(self.problem, 'target_dist', None)
            if benchmark_value is not None:
                benchmark_value = benchmark_value.reshape(-1).cpu().numpy()
                
            if not self.multi_d:
                fig = plot_evolution(nu_dist, fn= fn1, benchmark=benchmark_value, return_fig=self.wandb)
            else:
                grid_dim = self.problem.grid_dim
                fig = plot_evolution_2D(nu_dist.reshape(-1, 2, grid_dim, grid_dim), fn= fn1, return_fig=self.wandb)
            
            if self.wandb:
                self.writer.add_plot_image(key = f'{self.runname}_bar', fig = fig, step = None)    
                
            fn2 = f'paper_results/figure/{self.runname}_prob.pdf'
            if not self.multi_d:
                fig = plot_stop_prob(stop_prob_arr, fn2, return_fig=self.wandb)   
            else:
                grid_dim = self.problem.grid_dim
                fig = plot_stop_prob_2D(stop_prob_arr.reshape(-1, grid_dim, grid_dim), fn2, return_fig=self.wandb)
            
            if self.wandb:
                self.writer.add_plot_image(key = f'{self.runname}_prob', fig = fig, step = None) 
    
        if not return_dist:
            return total_loss / batch
        else:
            return total_loss / batch, nu_dist
      
    def compute_loss_t(self, decision_net, nu_cont_0, nu_stop_0, t):
        decision_net_list = self.decision_net_list
        batch = nu_cont_0.shape[0]
        nu_cont = nu_cont_0
        nu_stop = nu_stop_0
        mu = nu_cont + nu_stop
        total_loss = 0
        for tt in range(t, self.T + 1):
            if tt == t:
                # Stop probability at time t, keep gradient
                stop_prob = self.get_decision_prob(decision_net, nu_cont, nu_stop, synchron=self.synchron)
            elif tt == self.T:
                # Stop probability at time T, constant no gradient
                stop_prob = torch.ones((batch, self.problem.state_dim)).float().to(self.device)
            else:
                # Stop probability at time t + 1 to T - 1, trained, no gradient
                decision_net_t = decision_net_list[tt]
                with torch.no_grad():
                    stop_prob = self.get_decision_prob(decision_net_t, nu_cont, nu_stop, synchron=self.synchron)
            mu = nu_stop + nu_cont
            stop_cost_vec, running_cost_vec, terminal_cost_vec = self.problem.stop_cost(mu, time = tt)
            stop_cost = stop_cost_vec * nu_cont * stop_prob
            running_cost = running_cost_vec * nu_cont * (1 - stop_prob)
            terminal_cost = terminal_cost_vec
            total_loss += torch.sum(stop_cost + running_cost + terminal_cost)
            if tt != self.T:
                nu_stop, nu_cont = self.problem.propagate(nu_stop, nu_cont, stop_prob)
        
        total_loss = total_loss / batch
        return total_loss
    
    def get_decision_prob(self, decision_net, nu_cont, nu_stop, pure = False, synchron = False):
        batch = nu_cont.shape[0]
        state_dim = self.problem.state_dim
        
        state = torch.arange(state_dim).reshape(1, -1, 1).repeat(batch, 1, 1)
        nu_cont_rep = nu_cont.reshape(batch, 1, -1).repeat(1, state_dim, 1)
        nu_stop_rep = nu_stop.reshape(batch, 1, -1).repeat(1, state_dim, 1)
        
        aggregated_nu = torch.cat([nu_cont_rep, nu_stop_rep], dim = -1).reshape(batch * state_dim, -1).to(self.device)
        aggregated_state = state.reshape(batch * state_dim).to(self.device)
        
        if synchron:
            stop_prob = decision_net(aggregated_nu).reshape(batch, state_dim).to(self.device)
        else:
            stop_prob = decision_net(aggregated_state, aggregated_nu).reshape(batch, state_dim).to(self.device)
        if pure:
            stop_prob = torch.where(stop_prob > 0.5, torch.tensor(1.0), torch.tensor(0.0)).to(self.device)
        return stop_prob
        
            
        
            