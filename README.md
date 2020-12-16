# Implementations for Group Testing on a Network
Code for paper: Group Testing on a Network

Group testing---where multiple samples are tested together using a single test kit and individual tests are performed only for samples in positive groups---is a popular strategy to optimize the use of testing resources. We investigate how to effectively group samples for testing based on a transmission network. We formalize the group assembling problem as a graph partitioning problem, where the goal is to minimize the expected number of tests needed to screen the entire network. The problem is shown to be computationally hard and thus we focus on designing effective heuristics for it. Using realistic epidemic models on real contact networks, we show that our approaches save up to 33\% of resources---compared to the best baseline---at 4\% prevalence, are still effective at higher prevalence, and are robust to missing transmission data.

Evaluation is performed using python notebooks:

[Testing Performance](https://github.com/arleilps/group-testing/blob/main/Testing%20Performance%20Experiments.ipynb)

[Robustness](https://github.com/arleilps/group-testing/blob/main/Robustness.ipynb)

[Running time](https://github.com/arleilps/group-testing/blob/main/Scalability.ipynb)

For more details, see the paper:  
[Group Testing on a Network](http://www.cs.ucsb.edu/~arlei/pubs/aaai21.pdf "")  
Arlei Silva, Ambuj K Singh  
AAAI Conference on Artificial Intelligence (AAAI), 2021. 

Arlei Silva (arlei@cs.ucsb.edu)
