# NbodyGNN-WWW

### 0. The visualization of convergent feature evolution 
![alt text](https://github.com/papersubmit123/NbodyGNN-WWW/blob/main/figures/t1.png)
![alt text](https://github.com/papersubmit123/NbodyGNN-WWW/blob/main/figures/t2.jpg)
![alt text](https://github.com/papersubmit123/NbodyGNN-WWW/blob/main/figures/t3.png)
![alt text](https://github.com/papersubmit123/NbodyGNN-WWW/blob/main/figures/t5.png)
### 1.  The code running instructions are provided in each folder.

```
cd heterophilic_graphs/

cd homophilic_graphs/
 ```

### 2. Below are the experimental results:


| Method | Computer | Photo | CoauthorCS | CoauthorPhy | Ogbn-arxiv |
|--------|----------|-------|------------|-------------|------------|
| MLP    | 44.9±5.8 | 69.6±3.8 | 88.3±0.7 | 88.9±1.1   | 55.50±0.23 |
| GCN    | 82.6±2.4 | 91.2±1.2 | 91.1±0.5 | 92.8±1.0   | 72.17±0.33 |
| GAT    | 78.0±19.0| 85.7±20.3| 90.5±0.6 | 92.5±0.9   | **73.65±0.11** |
| HGCN   | 80.62±1.80| 88.21±1.42| 90.64±0.28| 90.74±1.45| 59.63±0.37 |
| GraphCON| 80.09±3.33| 90.13±1.93| 87.47±1.69| 92.46±1.10| 67.43±1.30 |
| NbodyGNN| **82.81±3.23**| **91.47±1.06**| **91.27±0.48**| **93.53±0.59**| 70.82±0.13 |

Tab 1. Node classification accuracy(%) on more datasets. The best results are highlighted in bold.



| Dataset | Computer | Photo | CoauthorCS | CoauthorPhy | Ogbn-Arxiv |
|---------|----------|-------|------------|-------------|------------|
| #Nodes  | 13381    | 7487  | 18333      | 34493       | 169343     |
| #Edges  | 245778   | 119043| 81894      | 247962      | 1166243    |
| #Features| 767      | 745   | 6805       | 8415        | 128        |
| #Classes | 10       | 8     | 15         | 5           | 40         |

Tab. 2: Statistic of new datasets.


