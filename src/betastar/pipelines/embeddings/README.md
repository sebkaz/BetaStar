# Pipeline embeddings


## Overview

<!---
Please describe your modular pipeline here.
-->
These pipeline describes the process of creating embeddings from a graph. 
We save all results as a parquet type file.

The pipeline is composed of the following steps (nodes):
1. preprocess graph: In python graph you should count node ids from 0.
This step will preprocess the graph to make sure that the graph is in the correct format.
2. Because of different graph formats for struc2vec we convert the graph file to the correct format. (see description below)
3. Embeddings with node2vec procedure. 


#### graph preprocess

From `conf/base/catalog.yml` you have read configuration for the input graph file. 
All graph files should be stored in `data/01_raw/edges` directory.
Graphs edges *.dat file with input node \\t output node pairs. 

For example:
```text
1	16495
2	2789
2	3654
2	3678
...
```

Config 
```yaml
graph:
  type: pandas.ParquetDataset
  filepath: data/01_raw/edges/edge_facebook.parquet
```

Output is a graph file in the same format as the input graph file but with start ids from zero.
```text
in,out
0,16494
1,2788
1,3653
1,3677
```
In `conf/base/catalog.yml` you have read configuration for the output graph file.
```yaml
#preprocessed graph - start from 0
graph_pre:
  type: pandas.ParquetDataset
  filepath: data/02_graphs/facebook_pre.parquet
```


#### Node preprocess 2
In the case of struc2vec we need to convert the graph file to the following format (in \\s out - start from 0) with space as separator.

Conversion file is saved (as in `conf/base/catalog.yml` file) in `struc2vec/graph/beta/facebook.csv`.
```yaml
graph_struct:
  type: pandas.CSVDataSet
  filepath: struc2vec/graph/beta/facebook.csv
  save_args:
    sep: ' '
    header: False
    index: False
```

You can choose kinds of embeddings in `conf/base/parameters.yml` file:
```yaml
embeddings:
  - node2vec
  - struc2vec
```

##### Struc2vec embeddings must bed done idependently for the kedro pipeline.
To run struc2vec embeddings you should run the following command:
```bash
cd struc2vec
python src/main.py --input graph/beta/facebook.csv --output emb/facebook.emb --num-walks 10 --walk-length 50 --window-size 5 --dimensions 16 --OPT1 True --OPT2 True --OPT3 True --worker 1
```
Result of struc2vec embeddings is saved in `struc2vec/emb/facebook.emb` directory.

If You choose struc2vec in embeddings parameters you must make struc2vec embeddings before running the kedro pipeline.

#### Node2vec embeddings

Result of node2vec embeddings is saved as in config file `conf/base/catalog.yml`:
```yaml
embedded_graph_node2vec:
  type: pandas.CSVDataSet
  filepath: data/03_embeddings/facebook_node2vec_embedding.csv
```
All parameters for node2vec embeddings are set in `conf/base/parameters.yml` file:
```yaml
node2vec:
  dim: 16
  walk_length: 50
  window: 10
  p: 1
  q: 1
  workers: 1
  seed: 123
```

workers=1 and seed=123 are set for reproducibility.

#### Concat data
In this step we concatenate all embeddings with graph community and non-community features.
The result is saved (as in `conf/base/catalog.yml` file) in:
```yaml
model_data:
  type: pandas.CSVDataSet
  filepath: data/04_models/facebook_model_data.csv
```
#### NaN Preprocessing 

In this step we preprocess NaN data in `target` and other features.

