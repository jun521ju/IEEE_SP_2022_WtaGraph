# WtaGraph
 WtaGraph is a web tracking and advertising detection framework based on Graph Neural Networks (GNNs). 
 The basic idea behind WtaGraph is that we construct a graph that represents HTTP network traffic and formulate web tracking and advertising detection as a task of edge classification in the graph.
 For more details, please refer to our [full paper](https://zhiju.me/assets/files/WtaGraph_SP22.pdf).
 Feel free to [contact Zhiju Yang](https://zhiju.me) if you run into any problem running WtaGraph.
 
 ## Instructions
 #### Download Our Dataset
 We shared our dataset (including two prebuilt graphs, node and edge features, and edge labels) on Zenodo.
 Please [download the dataset](https://zenodo.org/record/5166790) and put them in the subfolders of `data` folder as indicated by those files ending with `.placeholder`.
 
 #### Understand The Code
  - `data` folder: contains feature data and graph data
  - `gnn` folder: contains the code of our `WTA-GNN` for training and evaultion 
  - `graph` folder: contains the code for loading graph and corresponding feature data
  - `output` fodler: model output. Note: we put two pre-trained model in this folder
  - `main.py`: the entrance
  - `requirements.txt`: module version information
 
 #### Run The Code
 Once you have downloaded the dataset and put the files in right folders, you can run the code with `python3 main.py YOUR_ARGS_HERE`.
 Specfically, there are four functions:
 - `start_train(args)`: where you can train the WTA-GNN model on the given graph
 - `start_train_cv(args)`: where you can train the WTA-GNN model on the given graph with cross-validation
 - `eval_saved_model(args)`: where you can evaluate a pre-trained model (saved in the `output` folder)
 - `eval_model_inductive(args)`: where you can evaluate a pre-trained model in the inductive learning setting (check more details in our paper)
 
 Make sure to provide the correct parameters for the `args` (check the `get_args()` in the `main.py` for parameter details), otherwise it will use the default parameters.

 

