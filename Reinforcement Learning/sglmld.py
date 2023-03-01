import torch
from torch.distributions import Normal
from .optimizer import Optimizer
import numpy as np
import random
from math import sqrt

class SGLMLD(Optimizer):
    """
    Barely modified version of pytorch SGD to implement pSGLD
    The RMSprop preconditioning code is mostly from pytorch rmsprop implementation.
    """

    def __init__(self, params, lr=1e-3, noise=1e-6, alpha=0.99, eps=1e-8, centered=False, addnoise=True):
        defaults = dict(lr=lr, noise=noise, alpha=alpha, eps=eps, centered=centered, addnoise=addnoise)
        super(SGLMLD, self).__init__(params, defaults)
        
    def __setstate__(self, state):
        super(SGLMLD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('centered', False)

    def step(self, lr=None, noise=None, add_noise = False, c_difference=0):
        """
        Performs a single optimization step.
        """
        loss = None
        #print("~~~ gonna step")
        for group in self.param_groups:
            # print("test:", group['fixed_c'])
            # if group['fixed_c'] >= c_difference:
            #     #c_difference = group['fixed_c']
            # else:
            #     c_difference = 0
            if lr:
                group['lr'] = lr
            if noise:
                group['noise'] = noise
            for p in group['params']:
                #if (p.grad is None) or (torch.any(torch.isnan(p.grad))):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                #print("*** printing dp")
                #print(d_p)
                #print("*** printing  p")
                #print(p)
                #print(p.grad)
                #test = torch.any(torch.isnan(p.grad))
                #if test:
                    #print("yess") 
                #print(torch.any(torch.isnan(p.grad)))
                
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)
                        
                square_avg = state['square_avg']
                alpha = group['alpha']
                state['step'] += 1

                lr_t = group['lr'] * np.power((1 - 1e-5), state['step'] - 1)
                noise_t = group['noise'] * np.power((1 - 5e-5), state['step'] - 1)
                # sqavg x alpha + (1-alph) sqavg *(elemwise) sqavg
                #square_avg.mul_(alpha).addcmul_(1-alpha, d_p, d_p) #replacing with non-deprecated function
                square_avg.mul_(alpha).addcmul_(d_p, d_p, value=1-alpha)
                #print("p.grad: ", p.grad)
                #print(d_p)
                #print("is centered: ", group['centered'])
                #print("===== square_avg")
                #print(square_avg)
                #print("group['eps']")
                #print(group['eps'])
                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1-alpha, d_p)
                    avg = square_avg.addcmul(-1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])
                    
                
                if group['addnoise']:
                    #print("got here =")
                    
                    size = d_p.size()
                    #print("at SGLMLD")
                    #print("size: ", size)
                    #print("lr_t: ", lr_t)
                    #print("avg: ", avg)
                    #print("noise_t:",noise_t)
                    #print(torch.ones(size).div_(lr_t).div_(avg))#.sqrt())
                    #print(torch.ones(size).div_(lr_t).div_(avg).sqrt())
                    langevin_noise = Normal(
                        torch.zeros(size),
                        torch.ones(size).div_(lr_t).div_(avg).sqrt()
                    )
                    
                    

                    #new_c_difference = np.arctan(c_difference)
                    #new_c_difference = c_difference * c_difference
                    new_c_difference = c_difference
                    adjusted_noise_t = sqrt((noise_t*noise_t)+(new_c_difference/100))

                    if random.random() > 0.99999:
                        print("noise_t:",noise_t)    
                        print("changed:",adjusted_noise_t)
                        print("c_difference:", c_difference)
                    
                    
                    p.data.add_(d_p.div_(avg) + np.sqrt(2) * adjusted_noise_t * langevin_noise.sample(), alpha=-lr_t) 
                else:
                    #p.data.add_(-group['lr'], d_p.div_(avg))
                    p.data.addcdiv_(-lr_t, d_p, avg)
        #print("success")
        #print("==one iter")
        return loss
