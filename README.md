# Prescriptive Business Process Monitoring for Recommending Next Best Actions

## Technique
This project contains the source code for a novel Prescriptive Business Process Monitoring technique (PrBPM) technique for recommending next best actions.
Note: The project's source code based on the source code of the paper "Predictive business process monitoring with LSTM neural networks" from Tax et al. (2017). 

## Paper
If you use the code or fragments of it, please cite our paper:

```
@inprocessings{weinzierl2020nba,
    title={Precriptive Business Process Monitoring for Recommending Next Best Actions},
    author={Weinzierl, Sven and Dunzer, Sebastian and Zilker, Sandra and Matzner, Martin},
    booktitle={Proceedings of the 18th International Conference on Business Process Management Forum},
    year={2020},
    publisher={Springer}
}
```

You can access the last paper [here](https://www.researchgate.net/publication/342391344_Prescriptive_Business_Process_Monitoring_for_Recommending_Next_Best_Actions).

You can access the paper in which we presented the first prototype [here](https://library.gito.de/open-access-pdf/C12_Prescriptive_process_monitoring_-_a_technique_for_determining_next_best_actions_resub.pdf). 

## Technical details

We conducted all experiments on a workstation with 12 CPUs, 128 GB RAM.
In Table 1, we present the times for training and testing of the baseline and our prescriptive business process monitoring technique.

Table 1: Times for training and testing (in seconds).
|               | Helpdesk |        |        |        |  Bpi2019 |         |         |         |
|---------------|:--------:|--------|--------|--------|:--------:|---------|---------|---------|
| Experiment    | Baseline | k=5    | k=10   | k=15   | Baseline | k=5     | k=10    | k=15    |
| Training time | 132.10   | 125.02 | 125.06 | 100.07 | 355.66   | 757.17  | 743.00  | 750.10  |
| Testing time  | 310.20   | 905.43 | 904.26 | 976.88 | 3609.98  | 6829.20 | 4652.48 | 5861.92 |

We implemented our technique in Python 3.6.8 and used the deep-learning framework TensorFlow 2.3 to build deep learning models. 
