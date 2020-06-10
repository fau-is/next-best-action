# Prescriptive Business Process Monitoring for Recommending Next Best Actions

## Paper

...

## Technical details

We conducted all experiments on a workstation with 12 CPUs, 128 GB RAM, and a single GPU NVIDIA Quadro RXT6000.
In Table 1, we present the times for training and testing of the baseline and our prescriptive business process monitoring technique.

Table 1: Times for training and testing (in seconds).
|               | Helpdesk |        |        |        |  Bpi2019 |         |         |         |
|---------------|:--------:|--------|--------|--------|:--------:|---------|---------|---------|
| Experiment    | Baseline | k=5    | k=10   | k=15   | Baseline | k=5     | k=10    | k=15    |
| Training time | 132.10   | 125.02 | 125.06 | 100.07 | 355.66   | 757.17  | 743.00  | 750.10  |
| Testing time  | 310.20   | 905.43 | 904.26 | 976.88 | 3609.98  | 6829.20 | 4652.48 | 5861.92 |

We implemented our technique in Python 3.7 and used the deep-learning framework TensorFlow 1.14.1 to build deep learning models. 
