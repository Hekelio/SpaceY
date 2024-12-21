## Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier ##Install the package "scikit-learn"
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn import tree
from matplotlib import pyplot as plt
import os

# Set working directory (update with your path)
os.chdir("/Users/krashihekelio/Downloads")

# Configure pandas display
pd.set_option('display.max_columns', None)

# Load dataset
mydata = pd.read_csv("Motor_Collisions_Data_Cleaned_Time.csv", low_memory=False)

## Determine features and target (class)
#split dataset in features and target variable
feature_cols = ['Crash_Day', 'Period_of_Day', 'Vehicle_Brand_Num', 'VEHICLE_TYPE_Num', 'Impact_Point_Num']
target_col = ['Fatality']
X = mydata[feature_cols] # Features
y = mydata[target_col].values.ravel() # Target variable



## Splitting the data into two parts: (1) a training set and (2) a test set.
# Split dataset into training set and test set
# The random_state parameter is used to control this randomness, providing a way to reproduce results or ensure consistency across different runs of the same code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test


## Build Decision Tree
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_split=3)
# Train Decision Tree Classifer
results = clf.fit(X_train,y_train)


## Visualizing Decision Trees
plt.figure(figsize=(40,20))
tree.plot_tree(results, feature_names = X.columns)
plt.savefig('treeplot.png')
## You will see a .png file named decistion_tree.png being created in the main folder.

## Evaluating Model
# Predict the response for test dataset
y_pred1 = clf.predict(X_train)
y_pred2 = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy of train dataset(criterion=entropy, max_depth=4, min_samples_split=3):",metrics.accuracy_score(y_train, y_pred1))
print("Accuracy of test dataset(criterion=entropy, max_depth=4, min_samples_split=3):",metrics.accuracy_score(y_test, y_pred2))



## Optimize Your Decision Tree
# Create Decision Tree classifer object – set the criterion to entropy and control the maximum depths
clf = DecisionTreeClassifier(criterion="entropy", max_depth=6, min_samples_split=3)
##criterion{“gini”, “entropy”, “log_loss”}, default=”gini”
##The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
##min_samples_split: The minimum number of samples required to split an internal node. Default=2


# Train Decision Tree Classifer
results = clf.fit(X_train,y_train)
#Plot the results
plt.figure(figsize=(40,20))
tree.plot_tree(results, feature_names = X.columns)
plt.savefig('treeplot2.png')
#Predict the response for test dataset
y_pred3 = clf.predict(X_train)
y_pred4 = clf.predict(X_test)

print("Accuracy of train dataset (criterion=entropy, max_depth=6, min_samples_split=3):",metrics.accuracy_score(y_train, y_pred3))
print("Accuracy of test dataset (criterion=entropy, max_depth=6, min_samples_split=3):",metrics.accuracy_score(y_test, y_pred4))

### Random Forests
# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=10, random_state=1)

# Train the model on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the testing data
predictions = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Random Forest Accuracy: {accuracy:.2f}")

total_vehicles_per_brand = mydata.groupby('Vehicle_Brand').size()

fatalities_per_brand = mydata[mydata['Fatality'] == 1].groupby('Vehicle_Brand').size()

fatality_percentage = (fatalities_per_brand / total_vehicles_per_brand) * 100

sorted_fatality_percentage = fatality_percentage.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(sorted_fatality_percentage.index, sorted_fatality_percentage.values, color='skyblue', edgecolor='black')
plt.title("Fatality Percentage by Vehicle Brand", fontsize=14)
plt.xlabel("Vehicle Brand", fontsize=12)
plt.ylabel("Fatality Percentage (%)", fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()