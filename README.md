# DSA4199-FYP
 Additional optimizers in ```DSA4199-FYP/Reinforcement Learning/``` have to be manually added to pytorch's ```torch/optim/```  
 Below are some examples of good and bad performance.  

## Walker2d
Stochastic Gradient Langevin Dynamics | Stochastic Gradient Landscape Modified Langevin Dynamics
:-------------------------:|:-------------------------:
![walker2d-bad](https://user-images.githubusercontent.com/65672421/222369630-9a6d54ba-fce4-492d-a341-581313f451ae.gif) | ![walker2d-good](https://user-images.githubusercontent.com/65672421/222370146-2d62fa91-36b4-4001-93d6-c8e3b1e0d271.gif)
Falls over, inches itself forward | Keeps stepping forward

 ## HalfCheetah, using SGLMLD
Minimizing actor only, $f(x)=arctan(x), \alpha=\frac{1}{100}$ | Minimizing actor only, $f(x)=arctan(x), \alpha=\frac{1}{10,000}$
:-------------------------:|:-------------------------:
![halfcheetah-bad](https://user-images.githubusercontent.com/65672421/222123494-e20353be-ffff-412d-bb24-ee108ab59a13.gif) | ![halfcheetah-good](https://user-images.githubusercontent.com/65672421/222123513-bdb72a42-c7df-459d-aa8c-43981fc07277.gif)
Flips over, stops moving forward | Keeps moving forward  



