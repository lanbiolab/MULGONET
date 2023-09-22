# MULGONET
MULGONET

MULGONET: an interpretable neural network framework to integrate multi-omics data for cancer recurrence prediction and biomarker discovery




Overview of the MULGONET framework. (a) Data processing. Collection and preprocessing of multi-omics data and gene ontology data. Then, the GO hierarchy are constructed as a directed acyclic graph, and the graph is subjected to relationship extraction to obtain the GO hierarchical matrix. Finally, the annotation relationships between the genes and GO terms are used to generate the gene-GO relationship matrix. (b) Network construction. MULGONET is constructed based on the gene-GO relationship layers and GO hierarchy networks. Nodes on the far left represent omics data types, and the subsequent five layers represent higher-level biological entities(GO terms). The upper layer is constructed based on BP hierarchy, and the lower layer is constructed based on MF hierarchy. Finally, the outputs of the two sub-networks are merged by a multilayer perceptron to perform the prediction of the cancer recurrence. (c) Interpretability of model. The relative importance scores of the MULGONET nodes are calculated by the integrated gradients methods, and the node scores are visualized to reveal the regulatory process of cancer recurrence.


<b>Files</b>
MULGONET.py: MULGONET model
evaluates.py: Evaluation functions
preprocessing.py : Data preprocessing
training.py: Training and testing functions
weight_coef.py: Feature importance functions


