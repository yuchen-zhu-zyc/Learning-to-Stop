import os

import wandb
import torch
from datetime import datetime
import copy

class BaseWriter(object):
    def __init__(self, opt):
        self.rank = opt.global_rank
    def add_scalar(self, step, key, val):
        pass # do nothing
    def add_image(self, step, key, image):
        pass # do nothing
    def close(self): pass

class WandBWriter(BaseWriter):
    def __init__(self, wandb_api_key, log_dir, wandb_project, wandb_user, name):
        assert wandb.login(key= wandb_api_key)
        
        self.name = name
        
        wandb.init(dir=str(log_dir),
                    project=wandb_project,
                #    config=opt,
                    entity= wandb_user,
                    name = name)
                   # name= name + ' '+ datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def add_scalar(self, step, key, val):
        wandb.log({key: val}, step=step)

    def add_image(self, step, key, image):
        image = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        if step is not None:
            wandb.log({key: wandb.Image(image)}, step=step)
        else:
            wandb.log({key: wandb.Image(image)})

    def add_plot(self, step, key, fig):
        wandb.log({key: fig}, step=step)

    def add_plot_image(self, step, key, fig):
        if step is not None:
            wandb.log({key: wandb.Image(fig)}, step=step)
        else:
            wandb.log({key: wandb.Image(fig)})