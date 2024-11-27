# Classification Supported by Community-Aware Node Features

To perform computations, we use the Python Kedro framework. 
This framework is designed to operate on Python > 3.11 version.
The Python environment can be managed through `pip` or the Conda environment manager can be used as well.

```bash
python3.11 -m venv venv 
source venv/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.lock
```

## Graphs datasets

We possess a total of `11 datasets`, each of which comprises two files situated in the `data/01_raw/` folder. 
The initial file represents the graph in the edge list format, which is located in `data/01_raw/edges`.
The second file consists of details concerning community and non-community features and is located in `data/01_raw/data`.
You can generate this features by julia codes from `julia_codes` folder.

We consider undirected, connected, and simple (no loops nor parallel edges are allowed) graphs so that all node features are well defined and all methods that we use work properly. 
In each graph, we have some `ground-truth' labels for the nodes which is used to benchmark classification algorithms.
We used two families of graphs. 
The first family consists of synthetic networks from ABCD+o generator.
The main goal of experiments on this family is to perform a sanity test to evaluate whether the basic functionality of community-aware node features is working correctly or not. 
In these networks, the target class depends on the overall community structure of the graph.

The second family of networks we used in our experiments are empirical real-world graphs. 
We tried to select a collection of graphs with different properties (density, community structure, degree distribution, clustering coefficient, etc.). 

|Dataset | # of nodes | Average degree | # of communities | Target proportion (%) | Target description |
|---|---|---|---|---|---|
|Reddit|10,980|14.30|12|3.661|Is node a banned user|
|Grid|13,478|2.51|78|0.861|Is node a plant|
|LastFM|7624|7.29|28|20.619|Is node in country #17|
|Facebook|22,470|15.20|58|25.670|Is node a politician|
|Amazon|9314|37.49|39|8.601|Is node fraudulent|
|Twitch|168,114|80.87|19|47.01|Is streamed content mature|

## Kedro pipelines

We introduce four pipelines. 

You can find them in the `src/betastar/pipelines` folder. 

1. embeddings - Preprocessing of the graph and generation of fast `node2vec` embeddings.
2. task1 - information overlap between community-aware and classical features.
3. task2 - one-way predictive power of community-aware and classical features.
4. task3 - combined variable importance for prediction of community-aware and classical features.

The main configuration file is `conf/base/catalog.yml` and it is set for the 'facebook' dataset.

To run the pipeline for a specific dataset, you must activate the corresponding kedro environment. 
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


Urgent: this procedure takes a lot of time, so we recommend to use the prepared input files from `struc2vec/emb` folder.


The embeddings are generated using the following command:
```bash
cd struct2vec
python src/main.py --input graph/beta/facebook.csv --output emb/facebook.emb --num-walks 10 --walk-length 50 --window-size 5 --dimensions 16 --OPT1 True --OPT2 True --OPT3 True --worker 1
```
# Cite
```
@article{Kaminski:2024aa,
	author = {Kami{\'n}ski, Bogumi{\l} and Pra{\l}at, Pawe{\l} and Th{\'e}berge, Fran{\c c}ois and Zaj{\k a}c, Sebastian},
	date = {2024/06/15},
	doi = {10.1007/s13278-024-01281-2},
	id = {Kami{\'n}ski2024},
	isbn = {1869-5469},
	journal = {Social Network Analysis and Mining},
	number = {1},
	pages = {117},
	title = {Predicting properties of nodes via community-aware features},
	url = {https://doi.org/10.1007/s13278-024-01281-2},
	volume = {14},
	year = {2024},
	bdsk-url-1 = {https://doi.org/10.1007/s13278-024-01281-2}}
```
