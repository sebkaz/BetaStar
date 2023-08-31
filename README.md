# Classification Supported by Community-Aware Node Features

To perform computations, we use the Python Kedro framework. 
This framework is designed to operate on Python versions up to and including 3.10.
The Python environment can be managed through pip or the Conda environment manager can be used as well.

```bash
python3.10 -m venv venv 
source venv/bin/activate
pip install --upgrade pip setuptools
pip install -r src/requirements.txt
```

## Graphs datasets

We possess a total of 10 datasets, each of which comprises two files situated in the `data/01_raw/` folder. 
The initial file represents the graph in the edge list format, which is located in `data/01_raw/edges`. 
The second file consists of details concerning community and non-community features and is located in `data/01_raw/data`.

We consider undirected, connected, and simple (no loops nor parallel edges are allowed) graphs so that all node features are well defined and all methods that we use work properly. 
In each graph, we have some “ground-truth” labels for the nodes which is used to benchmark classification algorithms.
We used two families of graphs. 
The first family consists of synthetic networks.
The main goal of experiments on this family is to perform a sanity test to evaluate whether the basic functionality of community-aware node features is working correctly or not. 
In these networks, the target class depends on the overall community structure of the graph.

The second family of networks we used in our experiments are empirical real-world graphs. 
We tried to select a collection of graphs with different properties (density, community structure, degree distribution, clustering coefficient, etc.). 

## Kedro pipelines

We introduce four pipelines. You can find them in the `src/betastar/pipelines` folder. 
1. embeddings - Preprocessing of the graph and generation of `node2vec` embeddings.
2. task1 - information overlap between community-aware and classical features.
3. task2 - one-way predictive power of community-aware and classical features.
4. task3 - combined variable importance for prediction of community-aware and classical features.

The main configuration file is `conf/base/catalog.yml` and it is set for the 'facebook' dataset.

To run the pipeline for a specific dataset, you must activate the corresponding environment. 
For instance, if you want to work with the `amazon` dataset, execute command with --env parameter:
```base
kedro run --env=amazon
```


## Embedding preparation
We will use `node2vec` and `struc2vec` to generate embeddings for each node in the graph.

Node2vec embeddings are generated using the embedding pipeline.
```bash
kedro run --pipeline=embeddings
```

Struc2vec embeddings are generated using the `struc2vec` package added and modified (for python 3.10 env) separatly.


Urgent: this procedure takes a lot of time, so we recommend to use the prepared input files from `struct2vec/emb` folder.


The embeddings are generated using the following command:
```bash
cd struct2vec
python src/main.py --input graph/beta/facebook.csv --output emb/facebook.emb --num-walks 10 --walk-length 50 --window-size 5 --dimensions 16 --OPT1 True --OPT2 True --OPT3 True --worker 1
```

