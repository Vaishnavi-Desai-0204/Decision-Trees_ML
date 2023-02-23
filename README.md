# Decision-Trees_ML

This project is an implementation of a binary decision tree and a random forest model in Python, using numpy library.

The Node class that contains the decision tree node data. A DecisionTree class is then defined that uses the Node class and contains the code to grow the decision tree using a recursive approach. The best split of features is found at each node by computing the information gain and selecting the feature that provides the maximum gain. The tree is grown until a stopping criterion is met, such as reaching a maximum depth or a minimum number of samples per node.

Next, the code defines several helper functions, such as calculating entropy, splitting the data into left and right nodes, and finding the most common label.

Then, three main functions are defined to use the decision tree for binary classification. DT_train_binary is used to train a binary decision tree on a given dataset, DT_make_prediction is used to predict the target labels of a new dataset, and DT_test_binary is used to test the accuracy of the decision tree on a test dataset.

Finally, the code defines RF_build_random_forest function to build a random forest classifier that generates multiple decision trees, and averages their predictions to provide a more accurate model. The random forest model is built using a bootstrapping approach and by randomly selecting a subset of features at each node in each decision tree.
