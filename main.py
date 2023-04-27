# %% [markdown]
# # Introduction
#
# Decision Trees are a popular machine learning algorithm that is commonly used for classification and regression tasks. The algorithm is used to predict the target variable based on a set of input features. In this investigation, we aim to build a Decision Tree model to predict the number of cylinders of the motor of a car based on various car parameters such as fuel type, door number, engine size, car body type, etc.
#
# The number of cylinders in a car engine is an important feature that directly affects the performance of the vehicle. Therefore, predicting the number of cylinders accurately is important for various applications such as automotive manufacturing, sales forecasting, and vehicle maintenance.
#
# We will utilize a dataset that contains information on various car parameters and the number of cylinders of the motor. The dataset includes features such as fuel type, door number, engine size, car body type, etc. We will use this dataset to train a Decision Tree model that can predict the number of cylinders in the motor based on these features.
#
# By building a Decision Tree model, we can gain insights into the most important features that affect the number of cylinders in the motor of a car. We can also use the model to make predictions on new data and evaluate its performance using various metrics such as accuracy, precision, and recall.
#
# Overall, this investigation aims to demonstrate the utility of Decision Trees for predicting the number of cylinders in a car engine based on various car parameters.

# %% [markdown]
# # Preparation
#
# ## Notebook configuration
# To start, we will be importing the necessary libraries required for our analysis. For our Decision Tree model, we will be utilizing the scikit-learn library, which provides a comprehensive set of tools for machine learning tasks. Additionally, we will be using pandas, a popular data manipulation library in Python, for loading and manipulating our dataset.

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T20:29:45.221390Z","iopub.execute_input":"2023-04-27T20:29:45.222099Z","iopub.status.idle":"2023-04-27T20:29:45.242143Z","shell.execute_reply.started":"2023-04-27T20:29:45.222042Z","shell.execute_reply":"2023-04-27T20:29:45.240758Z"}}
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz
from enum import Enum

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %% [markdown]
# ## Data Loading
# Next, we will be loading the dataset into our code using pandas. This step involves reading the dataset from its source location and creating a pandas DataFrame, which is a 2-dimensional labeled data structure with columns of potentially different types.

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T20:29:45.244814Z","iopub.execute_input":"2023-04-27T20:29:45.245724Z","iopub.status.idle":"2023-04-27T20:29:45.284019Z","shell.execute_reply.started":"2023-04-27T20:29:45.245668Z","shell.execute_reply":"2023-04-27T20:29:45.282345Z"}}
# Load Data
raw_data = pd.read_csv("/kaggle/input/car-data/CarPrice_Assignment.csv")
raw_data.head()

# %% [markdown]
# ## Data encoding and cleaning
#
# Once we have loaded the dataset, we can start the data preparation process. This step involves handling missing values, encoding categorical variables, and scaling numerical variables, among other things. However, the specifics of data preparation will depend on the nature of the dataset and the machine learning task at hand.
#
# In summary, the preliminary steps involved in preparing the dataset and loading the necessary libraries for our Decision Tree model are crucial for ensuring the accuracy and reliability of our predictions. By utilizing appropriate libraries and following best practices for data preparation, we can create a robust Decision Tree model that accurately predicts the number of cylinders in a car engine based on various car parameters.
#
# ## Data Frame subset
# It's important to recognize, and correctly select the relevant features for our machine learning model. In this section, we will discuss the process of feature selection and why it is crucial for building an accurate and efficient Decision Tree model.
#
# Feature selection is the process of identifying and selecting the most important features from a dataset that contribute the most to the outcome variable. In our case, we want to predict the number of cylinders in a car engine based on various car parameters such as fuel type, door number, engine size, car body type, price, drivewheel, etc.
#
# To start, we will extract the useful columns from our pandas DataFrame that contain the relevant data for our model. In this step, we will eliminate unnecessary columns such as car model name, brand name, row ID, etc. These columns provide no useful data for our current hypothesis and can add noise to our analysis.

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T20:29:45.285872Z","iopub.execute_input":"2023-04-27T20:29:45.286346Z","iopub.status.idle":"2023-04-27T20:29:45.293777Z","shell.execute_reply.started":"2023-04-27T20:29:45.286305Z","shell.execute_reply":"2023-04-27T20:29:45.292673Z"}}
data = raw_data[[
    "fueltype",
    "doornumber",
    "carbody",
    "drivewheel",
    "enginelocation",
    "horsepower",
    "peakrpm",
    "aspiration",
    "price",
    "enginesize",
    "cylindernumber"
]]

# %% [markdown]
# After selecting the relevant features, we will check for any missing values in the dataset and handle them accordingly. Missing values can impact the accuracy of our model and can lead to biased predictions.

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T20:29:45.296825Z","iopub.execute_input":"2023-04-27T20:29:45.297613Z","iopub.status.idle":"2023-04-27T20:29:45.319610Z","shell.execute_reply.started":"2023-04-27T20:29:45.297571Z","shell.execute_reply":"2023-04-27T20:29:45.318027Z"}}
data.dropna(inplace=True)

# %% [markdown]
# As a data scientist specializing in Decision Trees, I recognize the importance of preprocessing our data to ensure that it is in a suitable format for our machine learning algorithms. In this section, we will discuss the process of encoding categorical variables into numerical values, which is a crucial step in building an accurate Decision Tree model.
#
# Decision Trees require numerical inputs, so we must encode categorical variables into numerical values. This step involves mapping each unique category to a numerical value using techniques such as one-hot encoding or label encoding. In our case, the dataset includes verbal representations of numbers, such as "two", "four", etc. To convert these values into numeric representations, we will use the following utility function `encode_verbal_number()`:

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T20:29:45.322337Z","iopub.execute_input":"2023-04-27T20:29:45.323438Z","iopub.status.idle":"2023-04-27T20:29:45.332637Z","shell.execute_reply.started":"2023-04-27T20:29:45.323358Z","shell.execute_reply":"2023-04-27T20:29:45.331262Z"}}
def encode_verbal_number(df, column_name):
    """
    Encodes a column containing verbal representations of numbers into numerical values.

    Parameters:
    df (pandas DataFrame): The DataFrame containing the column to be encoded
    column_name (str): The name of the column to be encoded

    Returns:
    The DataFrame with the encoded column
    """
    number_mapping = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12}
    df[column_name] = df[column_name].map(number_mapping)
    return df

# %% [markdown]
# For all other categorical parameters, we will use one-hot encoding. One-hot encoding converts each unique category into a binary column, where each row has a 1 in the column that corresponds to its category and 0s in all other columns. This technique is useful for handling categorical variables with more than two categories.

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T20:29:45.334594Z","iopub.execute_input":"2023-04-27T20:29:45.335067Z","iopub.status.idle":"2023-04-27T20:29:45.363615Z","shell.execute_reply.started":"2023-04-27T20:29:45.335022Z","shell.execute_reply":"2023-04-27T20:29:45.362276Z"}}
data = encode_verbal_number(data, "doornumber")
data = pd.get_dummies(data, columns=[
    "fueltype",
    "carbody",
    "drivewheel",
    "enginelocation",
    "aspiration"
])

# %% [markdown]
# The next step is crucial to machine learning models, normalizing numeric data before using it in our model. Normalization ensures that the data is on a comparable scale and prevents features with larger numeric ranges from dominating over features with smaller ranges. This is particularly relevant in our dataset, where some variables, such as car prices, can range in the thousands, while other variables, such as horsepower, can range in the hundreds.
#
# To normalize our data, we will use a technique called Min-Max scaling, which scales the data between 0 and 1 by subtracting the minimum value and dividing by the range. This will ensure that all features are on a comparable scale and prevent unwanted bias in our model.

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T20:29:45.366514Z","iopub.execute_input":"2023-04-27T20:29:45.367486Z","iopub.status.idle":"2023-04-27T20:29:45.383481Z","shell.execute_reply.started":"2023-04-27T20:29:45.367182Z","shell.execute_reply":"2023-04-27T20:29:45.382280Z"}}
numeric_columns = [
    "enginesize",
    "horsepower",
    "peakrpm",
    "price",
]
data[numeric_columns] = (
    data[numeric_columns] - data[numeric_columns].min()
) / (
    data[numeric_columns].max() - data[numeric_columns].min()
)

# %% [markdown]
# After normalizing our data, all of our numeric features will be on the same scale, ranging from 0 to 1. This step is crucial to ensure that our model can accurately capture the relationships between features and produce reliable predictions.
#
# ## X and Y Data Frames
# The next important step is to divide our data into input and output variables in order to train our machine learning model. In our case, we are trying to predict the number of cylinders in a car's motor based on various features such as fuel type, door number, car body type, engine size, and others.
#
# To begin, we will define our input and output variables using standard notation, with the output variable represented by the letter Y and the input variables represented by the letter X. The Y variable will contain the number of cylinders, which is the value we are trying to predict, while the pre-processed columns, including "fueltype", "doornumber", "carbody", "drivewheel", "enginelocation", "horsepower", "peakrpm", "aspiration", "price", and "enginesize", will form the X variable.

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T20:29:45.384870Z","iopub.execute_input":"2023-04-27T20:29:45.385563Z","iopub.status.idle":"2023-04-27T20:29:45.410840Z","shell.execute_reply.started":"2023-04-27T20:29:45.385502Z","shell.execute_reply":"2023-04-27T20:29:45.408977Z"}}
data.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T20:29:45.412671Z","iopub.execute_input":"2023-04-27T20:29:45.413179Z","iopub.status.idle":"2023-04-27T20:29:45.429758Z","shell.execute_reply.started":"2023-04-27T20:29:45.413109Z","shell.execute_reply":"2023-04-27T20:29:45.428039Z"}}
y_column_name = 'cylindernumber'
Y = data[[y_column_name]]
X = data.loc[:, data.columns != y_column_name]

# %% [markdown]
# Dividing our data into input and output variables is a standard practice in machine learning, and it allows us to build a model that can make predictions based on new input data. In our case, the input variables are the preprocessed features of a car such as fuel type, horsepower, and price, while the output variable is the number of cylinders in the car's engine.
#
# To split our data into training and testing subsets, we can use the train_test_split function from the sklearn.model_selection module.

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T20:29:45.436027Z","iopub.execute_input":"2023-04-27T20:29:45.436473Z","iopub.status.idle":"2023-04-27T20:29:45.445575Z","shell.execute_reply.started":"2023-04-27T20:29:45.436433Z","shell.execute_reply":"2023-04-27T20:29:45.443982Z"}}
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# %% [markdown]
# This will split the data into 80% for training and 20% for testing, with a fixed random state of 42 to ensure reproducibility.

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T20:29:45.448309Z","iopub.execute_input":"2023-04-27T20:29:45.448809Z","iopub.status.idle":"2023-04-27T20:29:45.467643Z","shell.execute_reply.started":"2023-04-27T20:29:45.448767Z","shell.execute_reply":"2023-04-27T20:29:45.466290Z"}}
Y_train.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T20:29:45.469373Z","iopub.execute_input":"2023-04-27T20:29:45.469793Z","iopub.status.idle":"2023-04-27T20:29:45.502452Z","shell.execute_reply.started":"2023-04-27T20:29:45.469755Z","shell.execute_reply":"2023-04-27T20:29:45.501391Z"}}
X_train.head()

# %% [markdown]
# ## Decision Tree Clasifier
# After pre-processing our data and defining the input (X) and output (Y) Data Frames, we can now proceed to train our Decision Tree model using the `DecisionTreeClassifier` class from the Scikit-learn library. This class provides various configuration options, but for our current use case, we will be using the `max_depth`, and the `criterion` parameters, the latter specifies which function to use to measure the quality of a split, we will be using `entropy`. This parameter controls the maximum depth of the tree and helps prevent overfitting by limiting the tree's size. By setting a reasonable `max_depth` value, we can balance the model's complexity and accuracy to produce reliable predictions. In our case we are going to use a `max_depth` of `3`.

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T20:29:45.503584Z","iopub.execute_input":"2023-04-27T20:29:45.503937Z","iopub.status.idle":"2023-04-27T20:29:45.510948Z","shell.execute_reply.started":"2023-04-27T20:29:45.503901Z","shell.execute_reply":"2023-04-27T20:29:45.509907Z"}}
tree_clf = DecisionTreeClassifier(max_depth =  3, criterion = "entropy")

# %% [markdown]
# After configuring our `DecisionTreeClassifier`, we are ready to train our model using the fit function. This function takes the input (X) and output (Y) Data Frames that we prepared earlier as its arguments, and it trains the Decision Tree model on this data. During the training process, the Decision Tree algorithm learns to identify patterns and relationships between the input and output variables, and it uses this knowledge to make predictions on new, unseen data.

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T20:29:45.512171Z","iopub.execute_input":"2023-04-27T20:29:45.512555Z","iopub.status.idle":"2023-04-27T20:29:45.537931Z","shell.execute_reply.started":"2023-04-27T20:29:45.512518Z","shell.execute_reply":"2023-04-27T20:29:45.536774Z"}}
tree_clf.fit(X_train,Y_train)

# %% [markdown]
# ## Model Overview
# After training the Decision Tree model, we can visualize the decision-making process by generating a diagram using the `graphviz` and `export_graphviz` libraries, along with our trained model. This diagram will provide an intuitive representation of how the model makes predictions and calculates class probabilities. The column names of our input Data Frame and output Data Frame will be used as class names in the diagram. To facilitate plotting other models, we are going to create a utility function called `plot_tree` this way we can just call this function and get a graph.

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T20:29:45.540038Z","iopub.execute_input":"2023-04-27T20:29:45.540587Z","iopub.status.idle":"2023-04-27T20:29:45.557588Z","shell.execute_reply.started":"2023-04-27T20:29:45.540533Z","shell.execute_reply":"2023-04-27T20:29:45.556460Z"}}
def plot_tree(_tree_clf):
    dot = tree.export_graphviz(_tree_clf, out_file=None,
                           feature_names=list(X),
                           class_names=_tree_clf.classes_,
                           filled=True, rounded=True,
                           special_characters=True)
    return graphviz.Source(dot)

graph = plot_tree(tree_clf)

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T20:29:45.558851Z","iopub.execute_input":"2023-04-27T20:29:45.560423Z","iopub.status.idle":"2023-04-27T20:29:45.618246Z","shell.execute_reply.started":"2023-04-27T20:29:45.560369Z","shell.execute_reply":"2023-04-27T20:29:45.616486Z"}}
plot_tree(tree_clf)

# %% [markdown]
# # Conclusions SKLearn Model
# After analyzing the resulting Decision Tree, we can identify the most influential features in our model. The first and most important feature is the Engine Size, as it is the root node of the tree. Additionally, other important features include Peak RPM, Price, and Fuel Type. These features align with our intuition that a larger engine size would typically correspond to more cylinders, and vice versa. We can further interpret the Decision Tree to understand how the model is making decisions and assigning probabilities to each class based on the feature values.
#
# ## Testing
# To validate our Decision Tree model, we need to use the previously split training data set, and in the future perform predictions on new data points. For this purpose, we must create a new data point that includes all the input features that we used for training the model. However, we need to handle the one-hot encoding of several columns manually. To achieve this, we will develop a utility class for each of the categorical columns of our input. Moreover, we must normalize the numeric features since our model is trained on the normalized version of these columns.
#
# It's essential to note that validating a model is a critical step in ensuring its effectiveness. Predicting on new examples allows us to estimate the model's accuracy and identify any potential problems such as overfitting or underfitting. Therefore, it's important to perform validation before deploying the model to real-world scenarios. To achieve this we can use the `accuracy_score` utility function from SkLearn.

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T20:29:45.620223Z","iopub.execute_input":"2023-04-27T20:29:45.620763Z","iopub.status.idle":"2023-04-27T20:29:45.636478Z","shell.execute_reply.started":"2023-04-27T20:29:45.620702Z","shell.execute_reply":"2023-04-27T20:29:45.634788Z"}}
Y_pred = tree_clf.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)

# %% [markdown]
# After training and testing our Decision Tree model with a maximum depth of 3, we can see that it achieved an accuracy score of **87.8%**. This score indicates that our model is performing well in predicting the number of cylinders based on the selected features.
#
# To further evaluate our model, we can test it with a larger depth of 5, using the same criterion. This will help us determine if increasing the depth of the tree will improve its accuracy or if it will lead to overfitting.
#
# We can create a new `DecisionTreeClassifier` object with a maximum depth of 5 and fit it with our training data:

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T20:29:45.639077Z","iopub.execute_input":"2023-04-27T20:29:45.639686Z","iopub.status.idle":"2023-04-27T20:29:45.705060Z","shell.execute_reply.started":"2023-04-27T20:29:45.639641Z","shell.execute_reply":"2023-04-27T20:29:45.702547Z"}}
tree_clf_5 = DecisionTreeClassifier(max_depth =  5, criterion = "entropy")
tree_clf_5.fit(X_train,Y_train)
plot_tree(tree_clf_5)

# %% [markdown]
# The Decision Tree models with maximum depths of 3 and 5 have shown different results. Although the initial nodes are similar, as the tree grows, new categories start to appear. In particular, the model with a maximum depth of 5 shows new categories such as the drive wheel and the number of doors. It's important to note that increasing the depth of a tree may improve its accuracy on the training set, but it may also cause overfitting, which can lead to poor performance on new data.
#
# We can now test for accuaracy in the same way as we did for the model with a maximum depth of 3, and then compare it to our newest model.

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T20:29:45.707586Z","iopub.execute_input":"2023-04-27T20:29:45.709131Z","iopub.status.idle":"2023-04-27T20:29:45.722247Z","shell.execute_reply.started":"2023-04-27T20:29:45.709047Z","shell.execute_reply":"2023-04-27T20:29:45.720366Z"}}
Y_pred_5 = tree_clf_5.predict(X_test)

accuracy_depth_5 = accuracy_score(Y_test, Y_pred_5)

print("Accuracy:", accuracy_depth_5)

# %% [markdown]
# # Comparison Conclision
# After analyzing the results of our second model, we can observe that it has achieved an accuracy score of **90.2%**. This represents a **2.4%** improvement over our first model, which is a positive indication of our model's performance. However, it is important to note that increasing accuracy through model complexity comes with certain trade-offs.
#
# One potential problem we may face is overfitting. With an increase in model complexity, our Decision Tree may create branches that are too specific to certain instances in the training data, leading to poor generalization and reduced performance on new, unseen data. It is therefore crucial to monitor the model's performance on the validation and test sets to avoid overfitting.
#
# In this case, our second model's increased accuracy did not appear to negatively impact its overall performance. However, it is important to keep in mind the potential consequences of overfitting and regularly evaluate our model's performance on unseen data.
#
# # Comparison with manual implementation
# I would like to reinforce the deterministic nature of Decision Trees. To demonstrate this, we will implement a custom version of the DecisionTreeClassifier of SkLearn from scratch. By doing this, we can ensure that with the same training and testing data subsets and using the same criterion, we can achieve the same accuracy and nodes as the SkLearn's version.
#
# However, using our custom implementation may come with some drawbacks. For instance, we may lose some utility functions such as visualization and automatic encoding of string labels. Hence, we need to transform the output Data Frame to have only numeric data.
#
# The following code is based off of the following [Medium post](https://towardsdatascience.com/implementing-a-decision-tree-from-scratch-f5358ff9c4bb) and the following [GitHub Repository](https://github.com/marvinlanhenke/DataScience/tree/main/MachineLearningFromScratch).

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T20:29:45.724888Z","iopub.execute_input":"2023-04-27T20:29:45.726010Z","iopub.status.idle":"2023-04-27T20:29:45.756112Z","shell.execute_reply.started":"2023-04-27T20:29:45.725946Z","shell.execute_reply":"2023-04-27T20:29:45.754891Z"}}
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _is_finished(self, depth):
        if (depth >= self.max_depth
            or self.n_class_labels == 1
            or self.n_samples < self.min_samples_split):
            return True
        return False

    def _entropy(self, y):
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy

    def _create_split(self, X, thresh):
        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()
        return left_idx, right_idx

    def _information_gain(self, X, y, thresh):
        parent_loss = self._entropy(y)
        left_idx, right_idx = self._create_split(X, thresh)
        n, n_left, n_right = len(y), len(left_idx), len(right_idx)

        if n_left == 0 or n_right == 0:
            return 0

        child_loss = (n_left / n) * self._entropy(y[left_idx]) + (n_right / n) * self._entropy(y[right_idx])
        return parent_loss - child_loss

    def _best_split(self, X, y, features):
        split = {'score':- 1, 'feat': None, 'thresh': None}

        for feat in features:
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = self._information_gain(X_feat, y, thresh)

                if score > split['score']:
                    split['score'] = score
                    split['feat'] = feat
                    split['thresh'] = thresh

        return split['feat'], split['thresh']

    def _build_tree(self, X, y, depth=0):
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))

        # stopping criteria
        if self._is_finished(depth):
            most_common_Label = np.argmax(np.bincount(y))
            return Node(value=most_common_Label)

        # get best split
        rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, rnd_feats)

        # grow children recursively
        left_idx, right_idx = self._create_split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feat, best_thresh, left_child, right_child)

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def predict(self, X):
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)

# %% [markdown]
# Before training the model with our custom `DecisionTree` class, we need to encode any string labels present in our dataset. This is because our custom algorithm cannot directly handle string labels during the fitting process. Instead, we need to convert them to numeric labels using methods such as Label Encoding or One-Hot Encoding. This process ensures that our Decision Tree can effectively split the data based on the attributes and classify the target variable accurately.

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T20:29:45.758037Z","iopub.execute_input":"2023-04-27T20:29:45.759059Z","iopub.status.idle":"2023-04-27T20:29:45.780064Z","shell.execute_reply.started":"2023-04-27T20:29:45.759000Z","shell.execute_reply":"2023-04-27T20:29:45.778774Z"}}
Y_train = encode_verbal_number(Y_train, "cylindernumber")
Y_test = encode_verbal_number(Y_test, "cylindernumber")

# %% [markdown]
# Once we have encoded our data, we can create an instance of our custom Decision Tree with a maximum depth of 3, similar to our previous model. Setting a maximum depth can help prevent overfitting by limiting the number of splits the algorithm can make. This can improve the generalization of the model to new, unseen data. And, in this specific case it helps us validate our hypothesis about the deterministic nature of Decision Trees.

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T20:29:45.781718Z","iopub.execute_input":"2023-04-27T20:29:45.783298Z","iopub.status.idle":"2023-04-27T20:29:45.898610Z","shell.execute_reply.started":"2023-04-27T20:29:45.783244Z","shell.execute_reply":"2023-04-27T20:29:45.896551Z"}}
clf = DecisionTree(max_depth = 3)
clf.fit(X_train.values, Y_train.values[:, 0])

y_pred_man = clf.predict(X_test.values)
accuracy_man = accuracy_score(Y_test, y_pred_man)

print("Accuracy:", accuracy_man)

# %% [markdown]
# # Custom vs SkLearn Decision Tree model Conclusions
# With this we can confirm that one of the key advantages of Decision Trees is their deterministic nature. This means that if we use the same training and testing data subsets, and the same splitting criterion, we should obtain the same results every time we run the model even if we use a custom implementation, or a framework.
#
# To validate this property of Decision Trees, we are comparing the accuracy of our custom Decision Tree with entropy against the SkLearn DecisionTreeClassifier. Since the accuracy is the same, we can be confident that our custom model is indeed deterministic.
#
# In this case the accuracy of both is 87.8%, thus we can conclude that our hypothesis is correct and that Decision Trees are deterministic. This result reinforces the reliability of Decision Trees as a machine learning algorithm that can produce consistent and interpretable results.
#
# # Custom instances generation
# It's important to able to create new instances to test the effectiveness of models developed. However, this process can be complex, particularly when configuring the one-hot encoding accurately and in the correct sequence. Therefore, it is imperative to have a utility function or class that simplifies this task.
#
# To this end, I propose the implementation of a class that specifically handles the creation of new instances. The class can encapsulate all the necessary functionalities required to create new instances, including the one-hot encoding configuration, and ensure consistency in the process. This would not only simplify the creation of new instances but also enhance the overall efficiency of the model testing process.
#
# In conclusion, having a utility function or class to handle the creation of new instances is crucial in Data Science, particularly in Decision Trees. It simplifies the process, ensures accuracy, and enhances efficiency, ultimately contributing to the effectiveness of the models developed.
#
# ## Enums for one hot encoding
# The first step would be to generate the Enumerators for the different columns that have been one-hot encoded, this way we can ensure that when we call our function we are actually using the values that we are expecting, preventing the incorrect use of another class.

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T20:29:45.900449Z","iopub.execute_input":"2023-04-27T20:29:45.901788Z","iopub.status.idle":"2023-04-27T20:29:45.910779Z","shell.execute_reply.started":"2023-04-27T20:29:45.901738Z","shell.execute_reply":"2023-04-27T20:29:45.909277Z"}}
class FuelTypes(Enum):
    Gas = "gas"
    Diesel = "diesel"

class DoorNumberTypes(Enum):
    Two = 2
    Four = 4

class CarBodyTypes(Enum):
    Convertible = "convertible"
    Hatchback = "hatchback"
    Sedan = "sedan"
    Wagon = "wagon"
    HardTop = "hardtop"

class DriveWheelTypes(Enum):
    RearWheelDrive = "rwd"
    FrontWheelDrive = "fwd"
    FourWheelDrive = "4wd"

class EngineLocationTypes(Enum):
    Front = "front"
    Rear = "rear"

class AspirationTypes(Enum):
    Standard = "std"
    Turbo = "turbo"

# %% [markdown]
# After declaring our Enums for the relevant columns, it is essential to create utility functions that facilitate these processes.
#
# The first utility function, `normalize_from_df`, is crucial in ensuring consistency between the value ranges of the training and testing data. This function normalizes a value based on the Data Frame from which it was extracted, ensuring that the same normalization ranges are used for both the training and testing data. This function ensures the model's accuracy by maintaining consistency in the data normalization process.
#
# The second utility function, `new_instance`, plays a pivotal role in creating new instances for testing the model's efficacy. This function takes in the user-defined values and generates an appropriate array that matches the new instance with the correct one-hot encoding. This function ensures that the new instance is correctly encoded, and its feature values match the same range as the training data, leading to accurate predictions.

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T20:29:45.912427Z","iopub.execute_input":"2023-04-27T20:29:45.912834Z","iopub.status.idle":"2023-04-27T20:29:45.931319Z","shell.execute_reply.started":"2023-04-27T20:29:45.912795Z","shell.execute_reply":"2023-04-27T20:29:45.929984Z"}}
def normalize_from_df(df, column, value):
    return (
        value - df[column].min()
    ) / (
        df[column].max() - df[column].min()
    )

def new_instance(
    df,
    doornumber,
    horse_power,
    peak_rpm,
    price,
    engine_size,
    fuel_type,
    car_body,
    drive_wheel,
    engine_location,
    aspiration
):
    normalized_hp = normalize_from_df(df, "horsepower", horse_power)
    normalized_prpm = normalize_from_df(df, "peakrpm", peak_rpm)
    normalized_price = normalize_from_df(df, "price", price)
    normalized_es = normalize_from_df(df, "enginesize", engine_size)

    return [
        doornumber,
        horse_power,
        peak_rpm,
        price,
        engine_size,
        1 if fuel_type == FuelTypes.Diesel else 0,
        1 if fuel_type == FuelTypes.Gas else 0,
        1 if car_body == CarBodyTypes.Convertible else 0,
        1 if car_body == CarBodyTypes.HardTop else 0,
        1 if car_body == CarBodyTypes.Hatchback else 0,
        1 if car_body == CarBodyTypes.Sedan else 0,
        1 if car_body == CarBodyTypes.Wagon else 0,
        1 if drive_wheel == DriveWheelTypes.FourWheelDrive else 0,
        1 if drive_wheel == DriveWheelTypes.FrontWheelDrive else 0,
        1 if drive_wheel == DriveWheelTypes.RearWheelDrive else 0,
        1 if engine_location == EngineLocationTypes.Front else 0,
        1 if engine_location == EngineLocationTypes.Rear else 0,
        1 if aspiration == AspirationTypes.Standard else 0,
        1 if aspiration == AspirationTypes.Turbo else 0,
    ]

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T20:29:45.932780Z","iopub.execute_input":"2023-04-27T20:29:45.933796Z","iopub.status.idle":"2023-04-27T20:29:45.958127Z","shell.execute_reply.started":"2023-04-27T20:29:45.933752Z","shell.execute_reply":"2023-04-27T20:29:45.956562Z"}}
test_instance = new_instance(
    X,
    2,
    200,
    5000,
    35000,
    200,
    FuelTypes.Gas,
    CarBodyTypes.Convertible,
    DriveWheelTypes.RearWheelDrive,
    EngineLocationTypes.Rear,
    AspirationTypes.Turbo
)
test_instance

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T21:04:56.061575Z","iopub.execute_input":"2023-04-27T21:04:56.062241Z","iopub.status.idle":"2023-04-27T21:04:56.076226Z","shell.execute_reply.started":"2023-04-27T21:04:56.062181Z","shell.execute_reply":"2023-04-27T21:04:56.074181Z"}}
pred_man = clf.predict([test_instance])
pred_5 = tree_clf_5.predict(np.array(test_instance).reshape(1, -1))
pred_3 = tree_clf.predict(np.array(test_instance).reshape(1, -1))

print("Predictions: ")
print("Model 5", pred_5)
print("Model 3", pred_3)
print("Model Manual", pred_man)