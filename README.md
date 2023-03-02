# DSA4199-FYP
 Additional optimizers in ```DSA4199-FYP/Reinforcement Learning/``` have to be manually added to pytorch's ```torch/optim/```  
 Below are some examples of good and bad performance.  

## Walker2d
Stochastic Gradient Langevin Dynamics | Stochastic Gradient Landscape Modified Langevin Dynamics
:-------------------------:|:-------------------------:
![walker2d-bad](https://user-images.githubusercontent.com/65672421/222369630-9a6d54ba-fce4-492d-a341-581313f451ae.gif) | ![walker2d-good](https://user-images.githubusercontent.com/65672421/222370146-2d62fa91-36b4-4001-93d6-c8e3b1e0d271.gif)
Falls over, inches itself forward | Keeps stepping forward

 ## HalfCheetah
SGLMLD, minimizing actor only, $f(x)=arctan(x), \alpha=\frac{1}{100}$ | SGLMLD, minimizing actor only, $f(x)=arctan(x), \alpha=\frac{1}{10,000}$
:-------------------------:|:-------------------------:
![halfcheetah-bad](https://user-images.githubusercontent.com/65672421/222123494-e20353be-ffff-412d-bb24-ee108ab59a13.gif) | ![halfcheetah-good](https://user-images.githubusercontent.com/65672421/222123513-bdb72a42-c7df-459d-aa8c-43981fc07277.gif)
Flips over, stops moving forward | Keeps moving forward  


 ## Hopper
SGLD | SGLMLD, maximizing adversary only, $f(x)=x^2, \alpha=\frac{1}{10,000}$
:-------------------------:|:-------------------------:
![hopper-bad](https://user-images.githubusercontent.com/65672421/222380711-a0c38460-5d47-4182-9b07-35abdb17be73.gif) | ![hopper-good](https://user-images.githubusercontent.com/65672421/222380754-4539ec85-55d5-4add-9095-53640a981984.gif)
Hops forward but falls down | Hops forward for longer before falling


 ## HalfCHeetah
SGLMLD, min actor ($f(x)=x,\alpha=\frac{1}{100,000}$),<br/> max adversary ($f(x)=x,\alpha=1$) | SGLMLD, min actor ($f(x)=x,\alpha=\frac{1}{100,000}$)<br/>, max adversary ($f(x)=x,\alpha=\frac{1}{100,000}$)
:-------------------------:|:-------------------------:
![halfcheetah-comb-bad](https://user-images.githubusercontent.com/65672421/222392779-80d57650-aad8-4957-bbb4-d867633cc756.gif) | ![halfcheetah-comb-good](https://user-images.githubusercontent.com/65672421/222392811-868f1f1b-2a5a-4a63-9a88-c2a676ac5d5f.gif)
Flips over, slowly moves forward | Keeps moving forward
