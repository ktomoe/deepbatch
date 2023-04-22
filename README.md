# deepbatch
An efficient approach based on graph neural network for predicting wait time in job schedulers.

## Confirmed environment
* Ubuntu 20.04.5 LTS
* Python Python 3.8.10

## Installation of other libraries
```
 $ git clone https://github.com/UTokyo-ICEPP/multiml.git
 $ cd multiml
 $ pip install -e .[pytorch]
 $ pip install  dgl -f https://data.dgl.ai/wheels/cu113/repo.html
 $ pip intall pyyaml
 $ pip install xgboost
```

## Make snapshots from parallel workload archive
* Download data from [the parallel workload archive](https://www.cs.huji.ac.il/labs/parallel/workload/) to your data directory
* Edit **data_dir** in config.yml of deepbatch
```
$ make_zarr.py
```

## Run experiments 
```
$ run_gat.py  # GAT model
$ run_mlp.py  # MLP model
$ run_bdt.py  # BDT model

```
