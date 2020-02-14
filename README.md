## Multi-Armed Bandits Algorithms

Implementation of several multi-armed bandits strategies, including: Epsilon-Greedy, UCB, UCB-V, and Lin-UCB. Plus application to online ads selection.

### Task

We will apply these bandits algorithms to online ads selection. We have a data file for 5000 articles. For each article we have its context (article profile) and the click-through rates of ads from 10 advertisers (one per line).
The data is contained in `ctr_data.txt`.

The format of one line is:
```
<article id>:<the article representation in 5 dimensions separated by ";">:<click rates on the ads of 10 advertisers separated by ";">
```

For each visit, the objective is to choose the ad from one of the 10 advertisers that will generate the highest click-through rate.

It is possible to apply contextual bandits algorithm (such as Lin-UCB) or regular bandits algorithm if we ignore the contextual information.

