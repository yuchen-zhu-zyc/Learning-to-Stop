# Official Repo for paper "Learning to Stop: Deep Learning for Mean Field Optimal Stopping"

[![Paper](https://img.shields.io/badge/Paper-PDF-brightgreen)](https://arxiv.org/abs/2410.08850v2)

## Highlight
![Demo of algo](assets/cover_2_mono.png)
![Evolution of Algo](assets/letter_M_uniform_dist_evol_CN_marked.png)


## Usage
The training typically follows the following structure.

```
initial_mu = torch.tensor([1, 0, 0, 0, 0, 0]).float().reshape(1, -1)
problem = DiceCommonNoise(sides = 6, T = 15, initial_mu=initial_mu, device = "cuda:0")

writer = WandBWriter(log_dir = "wandb", 
                    wandb_project = "MFOS")

solver = DPP(problem, batch = 2048, hidden=256, num_res_blocks=5, writer = writer, fn = "test", synchron=False)
### Or with solver = DirectApproach(...) if want to use solve with Direct Approach


trained_net = solver.train(n_iter = 1000, lr = 1e-3, copy_prev=False)
optimal_cost = solver.evaluate(trained_net, print_decision = False, pure = False, plot_bar=True)
print("Optimal Cost: ", optimal_cost)
wandb.finish()
```
Different applications used in the paper are implemented in ```problem.py```. 

Two solvers, based on dynamic programming principle and direct approach, respectively, are implemented in ```solver.py```. 

Helper function to visualize training/inference results, are implemented in ```plot_util.py```.


## Citation
If you find our work helpful, we would appreciate your citation at

```
@inproceedings{
    magnino2025learning,
    title={Learning to Stop: Deep Learning for Mean Field Optimal Stopping},
    author={Lorenzo Magnino and Yuchen Zhu and Mathieu Lauriere},
    booktitle={Forty-second International Conference on Machine Learning},
    year={2025},
    url={https://openreview.net/forum?id=LJvtuXhcIs}
    }
```
