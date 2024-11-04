# Comprehensive_Artifact
Boise State University Justin Carpenter's Comprehensive Artifact 

# Structural Graph Partition

This repository provides a new approach to partition large scale connected/disconnected graphs without the loss of important structural information. The approach can be implemented into any sized graph with the usage of common practice 
database structures and a loader for proper graph formating. The Algorithm included is an implementation of SIR-GN that utilizes a higher-order Weisfiler Lehman, 2 dimensional FWL. The cost of higher-order approaches, tend to outweigh
the improvements in expresivity power. This is why, with the application of graph partitioning, we can still operate on very large graphs. 

    correct sentence: "There weren't many waitresses sitting down."
    incorrect sentence: "There weren't all waitresses sitting down."

The model is assessed to be correct if it scores the correct sentence with a higer probability than the incorrect one. BLiMP was originally conceived as an unsupervised task to test NLP models without the use of fine-tuning. The ELECTRA model's novel pretraining method, however, is not readily compatible with the original evaluation strategy used in BLiMP. Therefore this program runs BLiMP with only minimal fine-tuning by default, using only 10 percent of the data for training, in order to apply the task as closely as possible to its original conception. Currently, this program only supports pytorch implementations of the ELECTRA transformer model. 

## Requirements

    numpy
    pandas
    sklearn
    math
    networkx

## Install and Running

To install, clone repository and install dependencies.

This includes the MUTAG dataset in the directory "data/MUTAG/" Be sure to not alter the path unless altering the __main__ in the file 2FWL_SIRGN_GraphPartition.py :

        'data/MUTAG/MUTAG_A.txt', list of edges in mutag
        'data/MUTAG/MUTAG_edge_labels.txt', list of edge labels
        'data/MUTAG/MUTAG_node_labels.txt', list of node labels
        'data/MUTAG/MUTAG_graph_indicator.txt', list of graph labels

Run using:

        python3 2FWL_SIRGN_GraphPartition.py

2FWL_SIRGN_GraphPartition.py Paramiters: (Graph1, Graph2, filename, n, node_labels, partitions, iterations)    
    
    -Graph1: The primary graph to preform the graph partition on and returns the 2FWL_SIR-GN embeddings.             
    -Graph2: The second graph is the same but is format friendly for SIR-GN to operate on for pre-embeddings.        
    -Filename: This is the path to save the pre-embeddings from SIR-GN for decreased time in future operations.      
    -N: This is the number of hops away from each node to calculate.                                                 
    -Node_labels: This is a list of all the nodes in the graph and their corresponding labels.                       
    -Partitions: This is the number of partitions to split the graph into to increase performance                    
    -Iterations: This is the number of iterations to do the 2FWL-SIRGN algorithm on the graph   


## Details

By default the program runs for with 20 partitions and 10 iterations on the MUTAG dataset

## Output

Upon completion the program will store the finished embeddings to 'FWL_SIRGN_GRAPH_PARTITION_MUTAG_10_20_10.txt' with an additional file 'SIRGN_embeddings_MUTAG_10_20_10.txt' which contains the original SIR-GN embeddings. 

This model should produce an F1 score of around .95 using default settings.



