# struc2vec

This repository provides a reference implementation of *struc2vec* as described in the paper:<br>
> struc2vec: Learning Node Representations from Structural Identity.<br>
> Leonardo F. R. Ribeiro, Pedro H. P. Saverese, Daniel R. Figueiredo.<br>
> Knowledge Discovery and Data Mining, SigKDD, 2017.<br>

The *struc2vec* algorithm learns continuous representations for nodes in any graph. *struc2vec* captures structural equivalence between nodes.  

Before executing *struc2vec*, it is necessary to install the following packages:
<br/>
```bash
pip install -r requirements.txt
```

### Basic Usage

#### Example
To run *struc2vec* on Mirrored Zachary's karate club network, execute the following command from the project home directory:<br/>
	``python src/main.py --input graph/karate-mirrored.edgelist --output emb/karate-mirrored.emb``

You can check out the other options available to use with *struc2vec* using:<br/>
	``python src/main.py --help``

#### Input
The supported input format is an edge list:

	node1_id_int node2_id_int
		

#### Output
The output file has *n+1* lines for a graph with *n* vertices. 
The first line has the following format:

	num_of_nodes dim_of_representation

The next *n* lines are as follows:
	
	node_id dim1 dim2 ... dimd

where dim1, ... , dimd is the *d*-dimensional representation learned by *struc2vec*.


### Miscellaneous

Please send any questions about the code and/or the algorithm to <leonardofribeiro@gmail.com>.

*Note:* This is only a reference implementation of the framework *struc2vec*.
