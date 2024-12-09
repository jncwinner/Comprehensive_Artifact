# Comprehensive_Artifact
Boise State University Justin Carpenter's Comprehensive Artifact 

# 2FWL-SIRGN: Scalable Structural Graph Representation Learning

Welcome to the official repository for 2FWL-SIRGN, a scalable framework for Higher-Order Weisfeiler-Lehman structural graph representation learning. This project integrates the 2-dimensional Folklore Weisfeiler-Lehman (2FWL) algorithm with structural graph partitioning, offering an exceptional balance between computational efficiency and expressive power.

# Overview

Graphs are fundamental in representing complex relationships in various domains, such as social networks, molecular chemistry, and knowledge graphs. Traditional approaches like k-dimensional Weisfeiler-Lehman algorithm and even the k-FWL extensions where ```k > 2``` provide structural insights but are often computationally expensive for large-scale graphs.

## What sets it apart from current approaches :
- Pre-processing: Identify structurally important edges in the complete graph.
- Scalability: Can define the exact number of partitions
- Structurally balanced: Group the similar structured cycles in each partition.
- Comparable to feature selection in other works.
- Does not remove Nodes from the dataset.
- Stored index: Can be done as a pre-processing task before any graph representation approach is ran. 


## 2FWL-SIRGN addresses these challenges by:

* Employing the 2FWL algorithm to capture higher-order graph structures.

* Utilizing structural graph partitioning to reduce computational overhead while preserving structural fidelity.

* Enabling optional pre-compute phase to store graph partitions for future use.

## Key Features

- Scalability: Processes graphs of unprecedented size through efficient partitioning.

- Expressiveness: Captures rich structural information, including cliques and cycles, via the 2FWL framework.

- Flexibility: Support for usage in a wide range of graph learning tasks, including node classification, link prediction, and graph classification.

- Performance: Outperforms existing methods in terms of computational cost and accuracy on large-scale benchmarks.

# Installation

## To set up the environment and dependencies for running the project, follow these steps:


# Clone the repository
git clone https://github.com/jncwinner/2FWL-SIRGN.git
cd 2FWL-SIRGN

# Install required dependencies
```
    numpy
    pandas
    sklearn
    math
    networkx
```

Ensure you have Python 3.8+ installed.

## Data Directory Configured.

This includes the MUTAG dataset in the directory "data/MUTAG/" Be sure to not alter the path unless altering the __main__ in the file 2FWL_SIRGN_GraphPartition.py :
```
        'data/MUTAG/MUTAG_A.txt', list of edges in mutag
        'data/MUTAG/MUTAG_edge_labels.txt', list of edge labels
        'data/MUTAG/MUTAG_node_labels.txt', list of node labels
        'data/MUTAG/MUTAG_graph_indicator.txt', list of graph labels
```

# Usage

## Running the Example:

Use the following command to execute the provided example datasets:
```
 python3 2FWL_SIRGN_GraphPartition.py
```
Replace dataset and specify the configuration file for your desired parameters in 2FWL_SIRGN_GraphPartition.py __main__.

## Key Parameters:
```
--Graph1: The primary graph to preform the graph partition on and returns the 2FWL_SIR-GN embeddings.
      
--Graph2: The second graph is the same but is format friendly for SIR-GN to operate on for pre-embeddings.
    
--Filename: This is the path to save the pre-embeddings from SIR-GN for decreased time in future operations.
  
--N: This is the number of features for each node.
 
--dataset: Dataset name (e.g., Mutag, Proteins, squirrel).

--num_partitions: Number of graph partitions.

--iterations: Number of 2FWL iterations.
```
## Output

Upon completion the program will store the finished embeddings to 'FWL_SIRGN_GRAPH_PARTITION_MUTAG_10_20_10.txt' with an additional file 'SIRGN_embeddings_MUTAG_10_20_10.txt' which contains the original SIR-GN embeddings. 

This model should produce an F1 score of around .95 using default settings.

# Results

## 2FWL-SIRGN achieves state-of-the-art performance on several graph benchmarks, including:

* Accuracy Improvements: 10-15% gains compared to standard GNNs on molecular and social graph datasets.

* Computational Efficiency: Reduces runtime by over 40% compared to k-WL-based methods.

For detailed results, see our paper (linked below).

# Citation

If you use this code in your research, please cite:
```
@article{Carpenter2024,
  author = {Justin Carpenter and Edoardo Serra},
  title = {2FWL-SIRGN: A Scalable Structural 2-dimensional Folklore Weisfeiler-Lehman Graph Representation Learning Approach Via Structural Graph Partitioning},
  year = {2024},
  journal = {IEEE BigData},
  volume = {TBA},
  number = {TBA},
  pages = {TBA-TBA},
  doi = {TBA}
}
```
Link to the paper

# Contributing

Contributions are welcome! If you'd like to improve the codebase or add features, feel free to open an issue or submit a pull request. Make sure to follow the contributing guidelines.

# License

This project is licensed under the MIT License. See LICENSE for more details.


This model should produce an F1 score of around .95 using default settings.

