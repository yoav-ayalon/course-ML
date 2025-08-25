import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import itertools

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, make_scorer

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.cluster import KMeans
from scipy.stats import mode
from sklearn.decomposition import PCA

from xgboost import XGBClassifier

# ------------------------------------------------- Load the data ------------------------------------------------------
data = pd.read_csv("XY_train.csv")
#print(data.info())

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- Date preparation - from Part A ------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# -------------------------------------------- Driving Experience change -----------------------------------------------
data['DRIVING_EXPERIENCE'] = data['DRIVING_EXPERIENCE'].where(
    data['AGE'] - data['DRIVING_EXPERIENCE'] >= 16, data['AGE'] - 16)
df_driving_experience = data['DRIVING_EXPERIENCE']

# -------------------------------------------- Credit Score change -----------------------------------------------------
data_notNull = data[data['CREDIT_SCORE'].notnull()]
data_missing = data[data['CREDIT_SCORE'].isnull()]
income_stats = data_notNull.groupby('INCOME')['CREDIT_SCORE'].agg(['mean', 'std'])


def fill_with_distribution(row):
    income_category = row['INCOME']
    # Get the mean and std for the income category
    mean = income_stats.loc[income_category, 'mean']
    std = income_stats.loc[income_category, 'std']
    return np.random.normal(mean, std)


data_missing.loc[:, 'CREDIT_SCORE'] = data_missing.apply(fill_with_distribution, axis=1)
data = pd.concat([data_notNull, data_missing])

# -------------------------------------------- Annual Mileage change ---------------------------------------------------
data_notNull = data[data['ANNUAL_MILEAGE'].notnull()]
data_missing = data[data['ANNUAL_MILEAGE'].isnull()]
mileage_stats = data_notNull.groupby('OUTCOME')['ANNUAL_MILEAGE'].agg(['mean', 'std'])


def fill_with_mileage_distribution(row):
    outcome_category = row['OUTCOME']
    mean = mileage_stats.loc[outcome_category, 'mean']
    std = mileage_stats.loc[outcome_category, 'std']
    return np.random.normal(mean, std)


data_missing.loc[:, 'ANNUAL_MILEAGE'] = data_missing.apply(fill_with_mileage_distribution, axis=1)
data = pd.concat([data_notNull, data_missing])

# -------------------------------------------- Vehicle Type & ID remove ------------------------------------------------
data = data.drop('VEHICLE_TYPE', axis=1)
data = data.drop('ID', axis=1)

# -------------------------------------------- categorization ----------------------------------------------------------
data['AGE'] = data['AGE'].apply(lambda x: 'young' if 16 <= x < 32 else ('adult' if 32 <= x < 67 else 'old'))
data['ANNUAL_MILEAGE'] = data['ANNUAL_MILEAGE'].apply(lambda x: 'low' if x <= 10000 else ('middle' if x <= 15000 else 'many'))
data['DRIVING_EXPERIENCE'] = data['DRIVING_EXPERIENCE'].apply(lambda x: 'low' if x <= 5 else ('middle' if x <= 15 else 'high'))
data['SPEEDING_VIOLATIONS'] = data['SPEEDING_VIOLATIONS'].apply(lambda x: 'none' if x <= 0 else ('few' if x <= 5 else 'many'))
data['PAST_ACCIDENTS'] = data['PAST_ACCIDENTS'].apply(lambda x: 'none' if x <= 0 else ('few' if x <= 3 else 'many'))
mean = np.mean(data['CREDIT_SCORE'])
std_dev = np.std(data['CREDIT_SCORE'])

# Define thresholds based on standard deviations
low_threshold = mean - std_dev  # μ - σ
high_threshold = mean + std_dev  # μ + σ


# Categorize data based on thresholds
def categorize(score):
    if score < low_threshold: return 'Low'
    elif low_threshold <= score <= high_threshold: return 'Medium'
    else: return 'High'

data['CREDIT_SCORE'] = pd.Series(data['CREDIT_SCORE']).apply(categorize)

data['GENDER'] = data['GENDER'].map({'male': 0, 'female': 1})
data['EDUCATION'] = data['EDUCATION'].map({'high school': 1, 'university': 2, 'none': 3})
data['INCOME'] = data['INCOME'].map({'poverty': 1, 'working class': 2, 'middle class': 3, 'upper class' : 4})
data['CREDIT_SCORE'] = data['CREDIT_SCORE'].map({'Low': 1, 'Medium': 2, 'High': 3})
data['VEHICLE_YEAR'] = data['VEHICLE_YEAR'].map({'before 2015': 1, 'after 2015': 2})
data['POSTAL_CODE'] = data['POSTAL_CODE'].map({10238: 1, 32765: 2, 92101: 3, 21217 : 4})
data['ANNUAL_MILEAGE'] = data['ANNUAL_MILEAGE'].map({'low': 1, 'middle': 2, 'many': 3})
data['AGE'] = data['AGE'].map({'young': 1, 'adult': 2, 'old': 3})
data['DRIVING_EXPERIENCE'] = data['DRIVING_EXPERIENCE'].map({'low': 1, 'middle': 2, 'high': 3})
data['SPEEDING_VIOLATIONS'] = data['SPEEDING_VIOLATIONS'].map({'none': 1, 'few': 2, 'many': 3})
data['PAST_ACCIDENTS'] = data['PAST_ACCIDENTS'].map({'none': 1, 'few': 2, 'many': 3})

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- Part B ------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# -------------------------------------------- split the data ----------------------------------------------------------
X = data.drop(columns=['OUTCOME'])  # Features
Y = data['OUTCOME']  # Target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=data['OUTCOME'], random_state=42)

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- Decision Trees ----------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# -------------------------------------------- build the full tree -----------------------------------------------------
X_train_DT = X_train
X_test_DT = X_test

model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train_DT, Y_train)

# add columns name to the tree - "feature_names"
if hasattr(X_train_DT, 'columns'): feature_names = X_train_DT.columns
else: feature_names = [f"Feature {i}" for i in range(X_train_DT.shape[1])]

# plt.figure(figsize=(12, 10))
# plot_tree(model, filled=True, feature_names=feature_names, class_names=['0', '1'])
# plt.tight_layout()
# plt.show()

print("\nFull tree F1 score and Accuracy:")
# Train F1 Score
train_predictions = model.predict(X_train_DT)
f1_train = f1_score(y_true=Y_train, y_pred=train_predictions)
print(f"F1 Score train set: {f1_train:.4f}")

# Train Accuracy
train_accuracy = accuracy_score(Y_train, train_predictions)
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")

#  Test F1 Score
test_predictions = model.predict(X_test_DT)
f1_test = f1_score(y_true=Y_test, y_pred=test_predictions)
print(f"F1 Score test set: {f1_test:.4f}")

# Test Accuracy
test_accuracy = accuracy_score(Y_test, test_predictions)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# -------------------------------------------- plots of desicion tree parameters before tuning -------------------------
# ---------------------------  Plot 1 - 3D plot of F1 score of max depth & min sample leaf, entropy & gini -------------

# Initialize parameters
max_depth_list = np.arange(1, 50, 5)
min_samples_leaf_list = np.arange(1, 20, 2)

# Function to calculate results for a given criterion
def calculate_results(criterion):
    results = pd.DataFrame()
    for max_depth in max_depth_list:
        for min_samples_leaf in min_samples_leaf_list:
            model = DecisionTreeClassifier(
                criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
            model.fit(X_train_DT, Y_train)
            results = results._append({
                'max_depth': max_depth,
                'min_samples_leaf': min_samples_leaf,
                'train_acc': f1_score(Y_train, model.predict(X_train_DT), average='weighted'),
                'test_acc': f1_score(Y_test, model.predict(X_test_DT), average='weighted')
            }, ignore_index=True)
    return results

# Calculate results for both criteria
results_entropy = calculate_results('entropy')
results_gini = calculate_results('gini')

fig = plt.figure(figsize=(16, 8))

# Subplot for entropy
ax1 = fig.add_subplot(121, projection='3d')
train_data_entropy = results_entropy.pivot(index='min_samples_leaf', columns='max_depth', values='train_acc')
test_data_entropy = results_entropy.pivot(index='min_samples_leaf', columns='max_depth', values='test_acc')
X, Y = np.meshgrid(train_data_entropy.columns, train_data_entropy.index)
# Train (Entropy)
ax1.scatter(X, Y, train_data_entropy, c='blue', marker='o', label='Train (Entropy)')
# Test (Entropy)
ax1.scatter(X, Y, test_data_entropy, c='red', marker='x', label='Test (Entropy)')
ax1.set_title('Entropy: Train and Test F1 Score')
ax1.set_xlabel('max_depth')
ax1.set_ylabel('min_samples_leaf')
ax1.set_zlabel('F1 Score')
ax1.legend()

# Subplot for gini
ax2 = fig.add_subplot(122, projection='3d')
train_data_gini = results_gini.pivot(index='min_samples_leaf', columns='max_depth', values='train_acc')
test_data_gini = results_gini.pivot(index='min_samples_leaf', columns='max_depth', values='test_acc')
# Train (Gini)
ax2.scatter(X, Y, train_data_gini, c='blue', marker='o', label='Train (Gini)')
# Test (Gini)
ax2.scatter(X, Y, test_data_gini, c='red', marker='x', label='Test (Gini)')
ax2.set_title('Gini: Train and Test F1 Score')
ax2.set_xlabel('max_depth')
ax2.set_ylabel('min_samples_leaf')
ax2.set_zlabel('F1 Score')
ax2.legend()

plt.tight_layout()
plt.show()

# ---------------------------  Plot 2 - F1 score of max depth, entropy & gini ------------------------------------------
# Initialize lists for the first model (max_depth)
max_depth_list_1 = np.arange(1, 50, 1)
res_1 = pd.DataFrame()
for max_depth in max_depth_list_1:
    model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=42)
    model.fit(X_train_DT, Y_train)
    res_1 = res_1._append({'max_depth': max_depth,
                           'train_acc': f1_score(Y_train, model.predict(X_train_DT)),
                           'test_acc': f1_score(Y_test, model.predict(X_test_DT))}, ignore_index=True)

# Initialize lists for the second model (min_samples_leaf)
max_depth_list_2 = np.arange(1, 50, 1)
res_2 = pd.DataFrame()
for max_depth in max_depth_list_2:
    model = DecisionTreeClassifier(criterion='gini', max_depth=max_depth, random_state=42)
    model.fit(X_train_DT, Y_train)
    res_2 = res_2._append({'max_depth': max_depth,
                           'train_acc': f1_score(Y_train, model.predict(X_train_DT)),
                           'test_acc': f1_score(Y_test, model.predict(X_test_DT))}, ignore_index=True)

global_y_min = min(res_1['train_acc'].min()-0.05, res_1['test_acc'].min()-0.05,
                   res_2['train_acc'].min()-0.05, res_2['test_acc'].min()-0.05)
global_y_max = max(res_1['train_acc'].max()+0.05, res_1['test_acc'].max()+0.05,
                   res_2['train_acc'].max()+0.05, res_2['test_acc'].max()+0.05)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(res_1['max_depth'], res_1['train_acc'], marker='o', markersize=4, label='Train accuracy')
axes[0].plot(res_1['max_depth'], res_1['test_acc'], marker='o', markersize=4, label='Test accuracy')
axes[0].set_title('Entropy - max depth')
axes[0].set_xlabel('max depth')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim(global_y_min, global_y_max)  # Apply global y-axis range
axes[0].legend()

axes[1].plot(res_2['max_depth'], res_2['train_acc'], marker='o', markersize=4, label='Train accuracy')
axes[1].plot(res_2['max_depth'], res_2['test_acc'], marker='o', markersize=4, label='Test accuracy')
axes[1].set_title('Gini - max depth')
axes[1].set_xlabel('max depth')
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim(global_y_min, global_y_max)  # Apply global y-axis range
axes[1].legend()

plt.tight_layout()
plt.show()

# -------------------------------------------- Hyperparameter-tuning ---------------------------------------------------
# -------------------------------------------- build GridSearchCVh -----------------------------------------------------
"""""
param_grid =\
    {'criterion': ['entropy', 'gini'],
    'max_depth': np.arange(3, 22, 1),
    'min_samples_leaf': np.arange(5, 25, 1),
    'max_features': [10,11,12,13,14, None],
    'ccp_alpha': np.arange(0, 0.001, 0.0005)}

grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=param_grid, cv=10, scoring='f1',n_jobs=-1, verbose=0, return_train_score=True)
grid_search.fit(X_train_DT, Y_train)

# Convert cv_results_ to a DataFrame and Save the DataFrame to a CSV file
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.to_csv('grid_search_DT_results.csv', index=False)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

formatted_params = {
    key: f"{value:.5f}" if isinstance(value, float) else value
    for key, value in best_params.items()}

print(f"Best parameters: {formatted_params}")
best_model.fit(X_train_DT, Y_train)
print(f"average F1 score obtained from the cross-validation: {grid_search.best_score_:.4f}")

# on the train set
train_predictions = best_model.predict(X_train_DT)
train_f1 = f1_score(y_true=Y_train, y_pred=train_predictions, average='weighted')
print(f"F1 Score of the best model found on the entire train set: {train_f1:.4f}")
# on the test set
test_predictions = best_model.predict(X_test_DT)
test_f1 = f1_score(y_true=Y_test, y_pred=test_predictions, average='weighted')
print(f"F1 Score test set: {test_f1:.4f}")

"""""

# -------------------------------------------- best tree from Hyper-parameters tuning ----------------------------------
# Best parameters from GridSearchCV
best_params = {
    'ccp_alpha': 0.00000,
    'criterion': 'gini',
    'max_depth': 7,
    'max_features': 14,
    'min_samples_leaf': 9}

best_tree_from_GridSearchCV = DecisionTreeClassifier(
    ccp_alpha=best_params['ccp_alpha'],
    criterion=best_params['criterion'],
    max_depth=best_params['max_depth'],
    max_features=best_params['max_features'],
    min_samples_leaf=best_params['min_samples_leaf'])

best_tree_from_GridSearchCV.fit(X_train_DT, Y_train)

# add columns name to the tree - "feature_names"
if hasattr(X_train_DT, 'columns'): feature_names = X_train_DT.columns
else: feature_names = [f"Feature {i}" for i in range(X_train_DT.shape[1])]

# plt.figure(figsize=(18, 12))
# plot_tree(best_tree_from_GridSearchCV, feature_names=feature_names, filled=True, class_names=['0', '1'], fontsize=6,proportion=False)
# plt.tight_layout()
# plt.show()

# full tree with purn to visualition
plt.figure(figsize=(18, 12))
plot_tree(best_tree_from_GridSearchCV, feature_names=feature_names, filled=True, class_names=['0', '1'] , max_depth=2, fontsize=20,proportion=False)
plt.tight_layout()
plt.show()

print("\nBest tree F1 score and Accuracy after Hyperparameter-tuning:")
# Train F1 Score
train_predictions = best_tree_from_GridSearchCV.predict(X_train_DT)
f1_train = f1_score(y_true=Y_train, y_pred=train_predictions)
print(f"F1 Score train set: {f1_train:.4f}")
# Train Accuracy
train_accuracy = accuracy_score(Y_train, train_predictions)
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")

# Test F1 Score
test_predictions = best_tree_from_GridSearchCV.predict(X_test_DT)
f1_test = f1_score(y_true=Y_test, y_pred=test_predictions)
print(f"F1 Score test set: {f1_test:.4f}")
# Test Accuracy
test_accuracy = accuracy_score(Y_test, test_predictions)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# -------------------------------------------- plots of Hyperparameter-tuning desicion tree ----------------------------
# Load the data file of the GridSearch
current_dir = os.path.dirname(__file__)  # Get the current working directory, Directory of the current script
csv_path = os.path.join(current_dir, 'grid_search_DT_results.csv') # Construct the relative path to the CSV file
cv_results = pd.read_csv(csv_path) # Load the CSV file

# max depth vs. max features in mean test F1 score
heatmap_data = cv_results[['param_max_depth', 'param_max_features', 'mean_test_score']]
# Handle duplicates by averaging scores for each combination of parameters
heatmap_data = heatmap_data.groupby(['param_max_depth', 'param_max_features'], as_index=False).mean()
heatmap_pivot = heatmap_data.pivot(index='param_max_depth', columns='param_max_features', values='mean_test_score')
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_pivot, annot=True, fmt=".4f", cmap="Blues", cbar_kws={'label': 'F1 Score'})
plt.title('F1 Score Heatmap for max depth vs. max features')
plt.xlabel('max features')
plt.ylabel('max depth')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# min samples leaf vs. criterion in mean test F1 score
heatmap_data = cv_results[['param_min_samples_leaf', 'param_criterion', 'mean_test_score']]
# Handle duplicates by averaging scores for each combination of parameters
heatmap_data = heatmap_data.groupby(['param_min_samples_leaf', 'param_criterion'], as_index=False).mean()
heatmap_pivot = heatmap_data.pivot(index='param_min_samples_leaf', columns='param_criterion', values='mean_test_score')
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_pivot, annot=True, fmt=".4f", cmap="Reds", cbar_kws={'label': 'F1 Score'})
plt.title('F1 Score Heatmap for min samples leaf vs. criterion')
plt.xlabel('criterion')
plt.ylabel('min samples leaf')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# -------------------------------------------- Feature importancese ----------------------------------------------------
feature_importances = best_tree_from_GridSearchCV.feature_importances_

if hasattr(X_train_DT, 'columns'):
    feature_names = X_train_DT.columns
else:
    feature_names = [f"Feature {i}" for i in range(X_train_DT.shape[1])]

# Sort features by importance
sorted_indices = np.argsort(feature_importances)[::-1]  # Indices of sorted importances in descending order
sorted_features = [feature_names[i] for i in sorted_indices]
sorted_importances = feature_importances[sorted_indices]

# # Display feature importances
# print("Feature Importances (sorted):")
# for feature, importance in zip(sorted_features, sorted_importances):
#     print(f"{feature}: {importance:.4f}")

plt.figure(figsize=(10, 6))
bars = plt.barh(sorted_features, sorted_importances, color='skyblue')
# Add importance values next to each bar
for bar, importance in zip(bars, sorted_importances):
    plt.text(bar.get_width() -0.005, bar.get_y() + bar.get_height() / 2,f"{importance:.4f}", va='center', fontsize=10)
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importances in the best model DT')
plt.gca().invert_yaxis()  # Flip y-axis for descending order
plt.tight_layout()
plt.show()


# -------------------------------------------- Prediction on the best tree ---------------------------------------------
def traverse_tree(tree, feature_names, node_index=0, sample=None):
    # If it's a leaf node
    if tree.children_left[node_index] == tree.children_right[node_index]:  # Leaf node
        print(f"\nLeaf node reached. Predicted class: {tree.value[node_index].argmax()}")
        print(f"Class probabilities: {tree.value[node_index][0]}")
        return

    # Get the feature and threshold for this node
    feature_index = tree.feature[node_index]
    threshold = tree.threshold[node_index]

    # Get feature name
    feature_name = feature_names[feature_index]

    # Print decision at this node
    print(f"If {feature_name} <= {threshold:.2f}?")

    # Decide the next node based on the sample value
    if sample[feature_index] <= threshold:
        print(f" -> Yes: {feature_name}={sample[feature_index]} <= {threshold:.2f})")
        next_node = tree.children_left[node_index]
    else:
        print(f" -> No: {feature_name}={sample[feature_index]} > {threshold:.2f})")
        next_node = tree.children_right[node_index]

    # Recurse into the next node
    traverse_tree(tree, feature_names, node_index=next_node, sample=sample)


example_data = [1, 1, 3, 2, 0, 1, 1, 1, 2, 2, 2, 1, 2, 2]
print("\nDecisions made by the tree:")
traverse_tree(best_tree_from_GridSearchCV.tree_, feature_names, sample=example_data)


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- Neural Networks ---------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
x_train_NN = X_train
x_test_NN = X_test

# standardizing the data using StandardScaler
standard_scaler = StandardScaler()
x_train_NN = standard_scaler.fit_transform(x_train_NN)
x_test_NN = standard_scaler.transform(x_test_NN)

# # check of the best way to standardizing the data
# # normalizing the data using MinMaxScaler
# minmax_scaler = MinMaxScaler()
# x_train_NN = minmax_scaler.fit_transform(x_train_NN)
# x_test_NN = minmax_scaler.transform(x_test_NN)
# # normalizing the data using RobustScaler
# robust_scaler = RobustScaler()
# x_train_NN = robust_scaler.fit_transform(x_train_NN)
# x_test_NN = robust_scaler.transform(x_test_NN)

model = MLPClassifier(random_state=42)
model.fit(x_train_NN, Y_train)

print("\nDefult NN F1 score and Accuracy:")
# Train
train_predictions = model.predict(x_train_NN)
train_f1 = f1_score(y_true=Y_train, y_pred=train_predictions, average='weighted')
print(f"F1 Score train set: {train_f1:.4f}")
train_accuracy = accuracy_score(Y_train, train_predictions)
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")

# Test
test_predictions = model.predict(x_test_NN)
test_f1 = f1_score(y_true=Y_test, y_pred=test_predictions, average='weighted')
print(f"F1 Score test set: {test_f1:.4f}")
test_accuracy = accuracy_score(Y_test, test_predictions)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# -------------------------------------------- Hyperparameter-tuning ---------------------------------------------------
# -------------------------------------------- build grid search -------------------------------------------------------
"""""
param_grid = {
    'hidden_layer_sizes': [
        (50,), (100,),        
        (100, 50), (250, 100),     
        (200, 150, 100)],
    'activation': ['relu', 'tanh', 'logistic'],  
    'solver': ['adam', 'sgd'],                   
    'alpha': [0.001, 0.01],                     
    'learning_rate_init': [0.001],              
    'max_iter': [500],                           
    'batch_size': [32, 64, 128]}


# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='f1', verbose=3, return_train_score=True)

# Fit the grid search to the training data
grid_search.fit(x_train_NN, Y_train)
best_params = grid_search.best_params_

# Convert cv_results_ to a DataFrame and Save the DataFrame to a CSV file
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.to_csv('grid_search_NN_results.csv', index=False)

print(f"Best parameters: {best_params}")
print("Parameter grid:", param_grid)

# Print the best parameters and best F1 score
print(f"average F1 score obtained from the cross-validation: {grid_search.best_score_:.4f}")
"""""
# ------------------------------------------ Evaluate the best model ---------------------------------------------------
# Load the data file of the GridSearch
current_dir = os.path.dirname(__file__)
csv_path = os.path.join(current_dir, 'grid_search_NN_results.csv')
results = pd.read_csv(csv_path)

# Best hyperparameters from the grid search for heatmaps
best_params = {
    'activation': 'logistic',
    'alpha': 0.01,
    'batch_size': 32,
    'hidden_layer_sizes': str((50,)),  # Convert to string for comparison
    'learning_rate_init': 0.001,
    'max_iter': 500,
    'solver': 'adam',
    'random_state': 42  # Ensure reproducibility
}

# Ensure correct data type for filtering
results['param_hidden_layer_sizes'] = results['param_hidden_layer_sizes'].astype(str)
results['param_batch_size'] = results['param_batch_size'].astype(str)
results['param_alpha'] = results['param_alpha'].astype(float)
results['param_learning_rate_init'] = results['param_learning_rate_init'].astype(float)
results['param_max_iter'] = results['param_max_iter'].astype(int)

# ------------------------------- Heatmap 1: Hidden Layer Sizes vs Activation vs Max Iter ------------------------------
filtered_results_1 = results[
    (results['param_solver'] == best_params['solver']) &
    (results['param_alpha'] == best_params['alpha']) &
    (results['param_learning_rate_init'] == best_params['learning_rate_init']) &
    (results['param_batch_size'] == str(best_params['batch_size']))]

pivot_table_1 = filtered_results_1.pivot_table(values='mean_test_score', index='param_hidden_layer_sizes', columns=['param_activation', 'param_max_iter'], aggfunc='mean')

plt.figure(figsize=(12, 8))
plt.title("F1 Score - Hidden Layer Sizes vs Activation vs Max Iter")
sns.heatmap(pivot_table_1, annot=True, fmt=".3f", cmap='Blues')
plt.xlabel("Activation and Max Iterations")
plt.ylabel("Hidden Layer Sizes")
plt.show()

# --------------------------- Heatmap 2: hidden layer vs Alpha -----------------------------------------
filtered_results_2 = results[
    (results['param_learning_rate_init'] == best_params['learning_rate_init']) &
    (results['param_activation'] == best_params['activation']) &
    (results['param_solver'] == best_params['solver']) &
    (results['param_batch_size'] == str(best_params['batch_size'])) &
    (results['param_max_iter'] == best_params['max_iter'])]

pivot_table_2 = filtered_results_2.pivot_table(values='mean_test_score', index='param_hidden_layer_sizes', columns='param_alpha', aggfunc='mean')

plt.figure(figsize=(12, 8))
plt.title("F1 Score - Hidden layer vs Alpha")
sns.heatmap(pivot_table_2, annot=True, fmt=".3f", cmap='Greens')
plt.xlabel("Alpha")
plt.ylabel("Hidden layer")
plt.show()

# ---------------------------------- Heatmap 3: Solver vs Batch Size ---------------------------------------------------
filtered_results_3 = results[
    (results['param_hidden_layer_sizes'] == best_params['hidden_layer_sizes']) &
    (results['param_activation'] == best_params['activation']) &
    (results['param_alpha'] == best_params['alpha']) &
    (results['param_learning_rate_init'] == best_params['learning_rate_init']) &
    (results['param_max_iter'] == best_params['max_iter'])]

pivot_table_3 = filtered_results_3.pivot_table(values='mean_test_score', index='param_solver', columns='param_batch_size',aggfunc='mean')

plt.figure(figsize=(12, 8))
plt.title("F1 Score - Solver vs Batch Size")
sns.heatmap(pivot_table_3, annot=True, fmt=".3f", cmap='Reds')
plt.xlabel("Batch Size")
plt.ylabel("Solver")
plt.show()

# ----------------------------- Train the network with the best configuration found ---------------------------------
# Train the network with the best hyperparameters for training
best_params = {
    'activation': 'logistic',
    'alpha': 0.01,
    'batch_size': 32,
    'hidden_layer_sizes': ((50,)),
    'learning_rate_init': 0.001,
    'max_iter': 500,
    'solver': 'adam',
    'random_state': 42  # Ensure reproducibility
}
best_model = MLPClassifier(**best_params)
best_model.fit(x_train_NN, Y_train)

print("\nBest NN F1 score and Accuracy after Hyperparameter-tuning:")
# Evaluate on the training set
train_predictions = best_model.predict(x_train_NN)
train_f1 = f1_score(y_true=Y_train, y_pred=train_predictions, average='weighted')
print(f"F1 Score train set: {train_f1:.4f}")
train_accuracy = accuracy_score(Y_train, train_predictions)
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")

# Evaluate on the test set
test_predictions = best_model.predict(x_test_NN)
test_f1 = f1_score(y_true=Y_test, y_pred=test_predictions, average='weighted')
print(f"F1 Score test set: {test_f1:.4f}")
test_accuracy = accuracy_score(Y_test, test_predictions)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Compute confusion matrix for the training set
train_conf_matrix = confusion_matrix(Y_train, train_predictions)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(train_conf_matrix, annot=True, fmt="d", cmap='Blues', cbar=False,
            xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title("Confusion Matrix - Train Set")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Compute confusion matrix for the test set
test_conf_matrix = confusion_matrix(Y_test, test_predictions)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(test_conf_matrix, annot=True, fmt="d", cmap='Blues', cbar=False,
            xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title("Confusion Matrix - Test Set")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- K-Means -----------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def relabel_clusters(y_true, y_prediction):
    # Identify unique cluster labels in predictions
    labels = np.unique(y_prediction)
    new_labels = np.zeros_like(labels)

    for i, label in enumerate(labels):
        # Find indices where current cluster label is assigned
        mask = (y_prediction == label)
        new_labels[i] = mode(y_true[mask])[0]

    # Create new prediction array with corrected labels
    new_y_pred = np.copy(y_prediction)
    for i, label in enumerate(labels):
        new_y_pred[y_prediction == label] = new_labels[i]

    return new_y_pred


x_train_Kmeans = X_train
x_test_Kmeans = X_test

# for standardizing the data later:
standard_scaler = StandardScaler()

# standardize the data:
x_train_Kmeans = pd.DataFrame(standard_scaler.fit_transform(x_train_Kmeans.values), columns=x_train_Kmeans.columns,
                                index=x_train_Kmeans.index)
x_test_Kmeans = pd.DataFrame(standard_scaler.fit_transform(x_test_Kmeans.values), columns=x_test_Kmeans.columns,
                                index=x_test_Kmeans.index)

# using PCA to lower the dimension to visualize the data:
pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(x_train_Kmeans)
X_test_pca = pca.transform(x_test_Kmeans)

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(x_train_Kmeans)

y_prediction_test = kmeans.predict(x_test_Kmeans)
y_prediction_train = kmeans.predict(x_train_Kmeans)

# Training and test predictions
y_prediction_test = relabel_clusters(Y_test, y_prediction_test)
y_prediction_train = relabel_clusters(Y_train, y_prediction_train)

# Calculation F1 Score with revised labels
print("\nK-means F1 score and Accuracy:")
f1_train_classified = f1_score(Y_train, y_prediction_train, average='weighted')
print(f'F1 Score train set: {f1_train_classified:.4f}')
accuracy_train_classified = accuracy_score(Y_train, y_prediction_train)
print(f'Train Accuracy: {accuracy_train_classified * 100:.2f}%')

f1_test_classified = f1_score(Y_test, y_prediction_test, average='weighted')
print(f"F1 Score test set: {f1_test_classified:.4f}")
accuracy_test_classified = accuracy_score(Y_test, y_prediction_test)
print(f'Test Accuracy {accuracy_test_classified * 100:.2f}%')


# Predicting the clusters on the reduced data
y_prediction_train_pca = kmeans.predict(x_train_Kmeans)
y_prediction_test_pca = kmeans.predict(x_test_Kmeans)
centers_pca = pca.transform(kmeans.cluster_centers_)

# Create a DataFrame for train_pca
df_pca_train = pd.DataFrame(data=X_train_pca, columns=['PC1', 'PC2'])
df_pca_train['cluster'] = y_prediction_train_pca

# Create a DataFrame for test_pca
df_pca_test = pd.DataFrame(data=X_test_pca, columns=['PC1', 'PC2'])
df_pca_test['cluster'] = y_prediction_test_pca

# Training Data classes
plt.figure(figsize=(10, 6))
scatter_classes = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=Y_train, cmap='plasma', alpha=0.6, edgecolor='k')
plt.title("Classes - Train Data")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.xlim(-5, 7)
plt.ylim(-3, 5)
plt.grid(True)
classes = ['did not claim his insurance', 'claimed his insurance']
colors = [scatter_classes.cmap(scatter_classes.norm(value)) for value in [0, 1]]
patches = [plt.plot([], [], marker="o", ms=10, ls="", mec=None, color=colors[i], label="{:s}".format(classes[i]))[0]
            for i in range(len(classes))]
plt.legend(handles=patches, loc='best')
plt.show()

# KMeans Clusters - Train Data Plot
plt.figure(figsize=(10, 6))
scatter_clusters = plt.scatter(df_pca_train['PC1'], df_pca_train['PC2'], c=df_pca_train['cluster'], cmap='Accent', alpha=0.6, edgecolor='k')
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], marker='+', s=100, color='red')
plt.title("KMeans Clusters - Train Data")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.xlim(-5, 7)
plt.ylim(-3, 5)
plt.grid(True)

# Adding legend for the KMeans plot
unique_clusters = np.unique(df_pca_train['cluster'])
cluster_colors = [scatter_clusters.cmap(scatter_clusters.norm(value)) for value in unique_clusters]
cluster_patches = [plt.plot([], [], marker="o", ms=10, ls="", mec=None, color=cluster_colors[i],
                            label="Cluster {:d}".format(unique_clusters[i]))[0] for i in range(len(unique_clusters))]
plt.legend(handles=cluster_patches, loc='best')
plt.show()

# Test data classes
plt.figure(figsize=(10, 6))
scatter_classes_test = plt.scatter(df_pca_test['PC1'], df_pca_test['PC2'], c=Y_test, cmap='plasma', alpha=0.6, edgecolor='k')
plt.title("Classes - Test Data")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.xlim(-5, 7)
plt.ylim(-3, 5)
plt.grid(True)
classes = ['did not claim his insurance', 'claimed his insurance']
colors = [scatter_classes_test.cmap(scatter_classes_test.norm(value)) for value in [0, 1]]
patches = [plt.plot([], [], marker="o", ms=10, ls="", mec=None, color=colors[i], label="{:s}".format(classes[i]))[0]
               for i in range(len(classes))]
plt.legend(handles=patches, loc='best')
plt.show()

# KMeans Clusters - Test Data Plot
plt.figure(figsize=(10, 6))
scatter_clusters_test = plt.scatter(df_pca_test['PC1'], df_pca_test['PC2'], c=df_pca_test['cluster'], cmap='Accent', alpha=0.6, edgecolor='k')
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], marker='+', s=100, color='red')
plt.title("KMeans Clusters - Test Data")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.xlim(-5, 7)
plt.ylim(-3, 5)
plt.grid(True)

# Adding a legend for the KMeans test plot
unique_clusters_test = np.unique(df_pca_test['cluster'])
cluster_colors_test = [scatter_clusters_test.cmap(scatter_clusters_test.norm(value)) for value in unique_clusters_test]
cluster_patches_test = [plt.plot([], [], marker="o", ms=10, ls="", mec=None, color=cluster_colors_test[i],
                                 label="Cluster {:d}".format(unique_clusters_test[i]))[0] for i in range(len(unique_clusters_test))]
plt.legend(handles=cluster_patches_test, loc='best')
plt.show()


# -------------------------------------------- many classes ------------------------------------------------------------
# Define cluster options
cluster_options = [2, 4, 6, 8, 10, 12, 15, 20]

# Results list
results = []

# Perform K-means for each cluster option
for n_clusters in cluster_options:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(x_train_Kmeans)

    # Predictions
    y_prediction_train = kmeans.predict(x_train_Kmeans)
    y_prediction_test = kmeans.predict(x_test_Kmeans)

    # Relabel clusters to match ground truth
    y_prediction_train = relabel_clusters(Y_train, y_prediction_train)
    y_prediction_test = relabel_clusters(Y_test, y_prediction_test)

    # Calculate metrics
    f1_train = f1_score(Y_train, y_prediction_train, average='weighted')
    accuracy_train = accuracy_score(Y_train, y_prediction_train)

    f1_test = f1_score(Y_test, y_prediction_test, average='weighted')
    accuracy_test = accuracy_score(Y_test, y_prediction_test)

    # Append results
    results.append({
        'n_clusters': n_clusters,
        'F1 Score (Train)': f1_train,
        'Accuracy (Train)': accuracy_train,
        'F1 Score (Test)': f1_test,
        'Accuracy (Test)': accuracy_test
    })

# Convert results to DataFrame and display
results_df = pd.DataFrame(results)

# Sort the DataFrame by F1 Score (Test) in descending order
results_df = results_df.sort_values(by='F1 Score (Test)', ascending=False)

# Apply formatting
results_df['F1 Score (Train)'] = results_df['F1 Score (Train)'].apply(lambda x: f"{x:.4f}")
results_df['F1 Score (Test)'] = results_df['F1 Score (Test)'].apply(lambda x: f"{x:.4f}")
results_df['Accuracy (Train)'] = results_df['Accuracy (Train)'].apply(lambda x: f"{x * 100:.2f}%")
results_df['Accuracy (Test)'] = results_df['Accuracy (Test)'].apply(lambda x: f"{x * 100:.2f}%")

pd.set_option('display.expand_frame_repr', False)  # Prevent line wrapping for wide tables
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns

print("\nK-means clustering results:")
print(results_df)


# Extract the best number of clusters from the sorted results
best_n_clusters = int(results_df.iloc[0]['n_clusters'])

# Fit KMeans with the best number of clusters
best_kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
best_kmeans.fit(x_train_Kmeans)

# Predictions for train and test sets
train_clusters = best_kmeans.predict(x_train_Kmeans)
test_clusters = best_kmeans.predict(x_test_Kmeans)

# Reduce the data to 2D for visualization (PCA already applied)
train_pca_clusters = pd.DataFrame(X_train_pca, columns=['PC1', 'PC2'])
train_pca_clusters['cluster'] = train_clusters

test_pca_clusters = pd.DataFrame(X_test_pca, columns=['PC1', 'PC2'])
test_pca_clusters['cluster'] = test_clusters

# Extract cluster centers for plotting
centers_pca = pca.transform(best_kmeans.cluster_centers_)

# Plot train and test clusters
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Plot for training data
scatter_train = axs[0].scatter(
    train_pca_clusters['PC1'], train_pca_clusters['PC2'],
    c=train_pca_clusters['cluster'], cmap='Accent', alpha=0.6, edgecolor='k'
)
axs[0].scatter(centers_pca[:, 0], centers_pca[:, 1], marker='+', s=500, color='gold',edgecolors='black',
    linewidth=2, label='Centers')
axs[0].set_title(f"KMeans Clusters - Train Data (n_clusters={best_n_clusters})")
axs[0].set_xlabel("PC1")
axs[0].set_ylabel("PC2")
axs[0].grid(True)

# Plot for testing data
scatter_test = axs[1].scatter(
    test_pca_clusters['PC1'], test_pca_clusters['PC2'],
    c=test_pca_clusters['cluster'], cmap='Accent', alpha=0.6, edgecolor='k'
)
axs[1].scatter(centers_pca[:, 0], centers_pca[:, 1], marker='+', s=500, color='gold',edgecolors='black',
    linewidth=2, label='Centers')
axs[1].set_title(f"KMeans Clusters - Test Data (n_clusters={best_n_clusters})")
axs[1].set_xlabel("PC1")
axs[1].set_ylabel("PC2")
axs[1].grid(True)

# Add legends
axs[0].legend(*scatter_train.legend_elements(), title="Clusters", loc='best')
axs[1].legend(*scatter_test.legend_elements(), title="Clusters", loc='best')

plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- XGboost -----------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
x_train_XGB = X_train
x_test_XGB = X_test

# standardizing the data using StandardScaler
standard_scaler = StandardScaler()
x_train_XGB = standard_scaler.fit_transform(x_train_XGB)
x_test_XGB = standard_scaler.transform(x_test_XGB)

defult_model = XGBClassifier(random_state=42)
defult_model.fit(x_train_XGB, Y_train)

print("\ndefult XGB F1 score and Accuracy:")
# Train F1 Score
train_predictions = defult_model.predict(x_train_XGB)
f1_train = f1_score(y_true=Y_train, y_pred=train_predictions)
print(f"F1 Score train set: {f1_train:.4f}")
# Train Accuracy
train_accuracy = accuracy_score(Y_train, train_predictions)
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")

# Test F1 Score
test_predictions = defult_model.predict(x_test_XGB)
f1_test = f1_score(y_true=Y_test, y_pred=test_predictions)
print(f"F1 Score test set: {f1_test:.4f}")
# Test Accuracy
test_accuracy = accuracy_score(Y_test, test_predictions)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# param_grid = {
#     'max_depth': [6, 7, 8, 9],  # Focus around max_depth=7 from DT
#     'learning_rate': [0.03, 0.05, 0.1, 0.3],  # Fine-tune around a small learning rate
#     'n_estimators': [100, 150, 250],  # Keep a reasonable range for boosting rounds
#     'subsample': [0.6, 0.8, 1.0],  # High subsampling for robust training
#     #'colsample_bytree': [0.6, 0.8, 0.9],  # Match max_features logic (14 features used)
#     'min_child_weight': [5, 10, 15],  # Based on min_samples_leaf=9 from DT
#     #'gamma': [0, 0.05, 0.2],  # Low regularization for splits
#     'reg_alpha': [0, 0.05, 0.1, 0.5],  # Minimal L1 regularization for feature sparsity
#     #'reg_lambda': [0.5, 2, 5]  # Moderate L2 regularization for weight control
# }
#
# # Generate all combinations of parameters
# param_combinations = list(itertools.product(
#     param_grid['max_depth'],
#     param_grid['learning_rate'],
#     param_grid['n_estimators'],
#     param_grid['subsample'],
#     #param_grid['colsample_bytree'],
#     #param_grid['gamma'],
#     param_grid['min_child_weight'],
#     param_grid['reg_alpha']))
#     #param_grid['reg_lambda']))
#
# # Initialize variables to track results
# results = []
#
# # Loop through all combinations
# for combination in param_combinations:
#     (max_depth, learning_rate, n_estimators, subsample, min_child_weight, reg_alpha) = combination
#      #colsample_bytree, gamma, min_child_weight, reg_alpha, reg_lambda) = combination
#
#     # Initialize and train the model
#     model = XGBClassifier(
#         max_depth=max_depth,
#         learning_rate=learning_rate,
#         n_estimators=n_estimators,
#         subsample=subsample,
#         #colsample_bytree=colsample_bytree,
#         #gamma=gamma,
#         min_child_weight=min_child_weight,
#         reg_alpha=reg_alpha,
#         #reg_lambda=reg_lambda,
#         scale_pos_weight=2.2,  # Adjusted for class imbalance
#         random_state=42
#     )
#
#     # Train the model on the entire training set
#     model.fit(x_train_XGB, Y_train)
#     train_predictions = model.predict(x_train_XGB)
#     f1 = f1_score(Y_train, train_predictions)
#
#     # Store results
#     results.append({
#         'max_depth': max_depth,
#         'learning_rate': learning_rate,
#         'n_estimators': n_estimators,
#         'subsample': subsample,
#         #'colsample_bytree': colsample_bytree,
#         #'gamma': gamma,
#         'min_child_weight': min_child_weight,
#         'reg_alpha': reg_alpha,
#         #'reg_lambda': reg_lambda,
#         'f1_score': f1})
#
# # Save results to a CSV file
# results_df = pd.DataFrame(results)
# results_df.to_csv('grid_search_XGB_results.csv', index=False)
#
# # Load the results from the CSV file
# XGB_result = pd.read_csv('grid_search_XGB_results.csv')
# # Find the row with the highest F1 score
# best_model_row = XGB_result.loc[results_df['f1_score'].idxmax()]
#
# # Display the best parameters and F1 score
# print("\nBest Parameters:")
# print(best_model_row.drop('f1_score'))
# print(f"\nBest F1 Score: {best_model_row['f1_score']:.4f}")

best_model = XGBClassifier(
        max_depth=8,
        learning_rate=0.3,
        n_estimators=100,
        subsample=0.6,
        #colsample_bytree=0.9,
        #gamma=0.00,
        min_child_weight=10,
        reg_alpha=0.5,
        #reg_lambda=1.00,
        scale_pos_weight=2.2,  # Adjusted for class imbalance
        random_state=42)

best_model.fit(x_train_XGB, Y_train)

print("\nBest XGB F1 score and Accuracy after Hyperparameter-tuning:")
# Train F1 Score
train_predictions = best_model.predict(x_train_XGB)
f1_train = f1_score(y_true=Y_train, y_pred=train_predictions)
print(f"F1 Score train set: {f1_train:.4f}")
# Train Accuracy
train_accuracy = accuracy_score(Y_train, train_predictions)
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")

# Test F1 Score
test_predictions = best_model.predict(x_test_XGB)
f1_test = f1_score(y_true=Y_test, y_pred=test_predictions)
print(f"F1 Score test set: {f1_test:.4f}")
# Test Accuracy
test_accuracy = accuracy_score(Y_test, test_predictions)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")



# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- final forecast ----------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
x_train_final = X_train
x_test_final = X_test

# standardizing the data using StandardScaler
standard_scaler = StandardScaler()
x_train_final = standard_scaler.fit_transform(x_train_final)
x_test_final = standard_scaler.transform(x_test_final)

# final NN
best_params = {
    'activation': 'logistic',
    'alpha': 0.01,
    'batch_size': 32,
    'hidden_layer_sizes': ((50,)),
    'learning_rate_init': 0.001,
    'max_iter': 500,
    'solver': 'adam',
    'random_state': 42}

final_model = MLPClassifier(**best_params)
final_model.fit(x_train_final, Y_train)

print("\nFinal model (NN) F1 score and Accuracy:")
# Evaluate on the training set
train_predictions = final_model.predict(x_train_final)
train_f1 = f1_score(y_true=Y_train, y_pred=train_predictions, average='weighted')
print(f"F1 Score train set: {train_f1:.4f}")
train_accuracy = accuracy_score(Y_train, train_predictions)
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")

# Evaluate on the test set
test_predictions = final_model.predict(x_test_final)
test_f1 = f1_score(y_true=Y_test, y_pred=test_predictions, average='weighted')
print(f"F1 Score test set: {test_f1:.4f}")
test_accuracy = accuracy_score(Y_test, test_predictions)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Compute confusion matrix for the training set
train_conf_matrix_final = confusion_matrix(Y_train, train_predictions)
test_conf_matrix_final = confusion_matrix(Y_test, test_predictions)

plt.figure(figsize=(12, 6))

# Test set confusion matrix
sns.heatmap(test_conf_matrix_final, annot=True, fmt="d", cmap='Blues', cbar=False,
            xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title("Confusion Matrix - Test Set")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

x_test_predict = pd.read_csv("X_test.csv")
# print(x_test_predict.head())
# print(x_test_predict.info())

# -------------------------------------------- change the x-test df ----------------------------------------------------
# drop VEHICLE_TYPE
x_test_predict = x_test_predict.drop('VEHICLE_TYPE', axis=1)

# Driving Experience change
x_test_predict['DRIVING_EXPERIENCE'] = x_test_predict['DRIVING_EXPERIENCE'].where(
    x_test_predict['AGE'] - x_test_predict['DRIVING_EXPERIENCE'] >= 16, x_test_predict['AGE'] - 16)
df_driving_experience_Xtest = x_test_predict['DRIVING_EXPERIENCE']

# Credit Score change
data_notNull_Xtest = x_test_predict[x_test_predict['CREDIT_SCORE'].notnull()]
data_missing_Xtest = x_test_predict[x_test_predict['CREDIT_SCORE'].isnull()]
income_stats_Xtest = data_notNull_Xtest.groupby('INCOME')['CREDIT_SCORE'].agg(['mean', 'std'])

def fill_with_distribution(row):
    income_category = row['INCOME']
    # Get the mean and std for the income category
    mean = income_stats_Xtest.loc[income_category, 'mean']
    std = income_stats_Xtest.loc[income_category, 'std']
    return np.random.normal(mean, std)

data_missing_Xtest.loc[:, 'CREDIT_SCORE'] = data_missing_Xtest.apply(fill_with_distribution, axis=1)
x_test_predict = pd.concat([data_notNull_Xtest, data_missing_Xtest])

# annual mileage change
data_notNull_Xtest = x_test_predict[x_test_predict['ANNUAL_MILEAGE'].notnull()]
data_missing_Xtest = x_test_predict[x_test_predict['ANNUAL_MILEAGE'].isnull()]

mean_mileage = data_notNull_Xtest['ANNUAL_MILEAGE'].mean()
std_mileage = data_notNull_Xtest['ANNUAL_MILEAGE'].std()

def fill_with_overall_distribution(row):
    return np.random.normal(mean_mileage, std_mileage)

data_missing_Xtest.loc[:, 'ANNUAL_MILEAGE'] = data_missing_Xtest.apply(fill_with_overall_distribution, axis=1)
x_test_predict = pd.concat([data_notNull_Xtest, data_missing_Xtest])
x_test_predict['ANNUAL_MILEAGE'] = x_test_predict['ANNUAL_MILEAGE'].round(-3)

# output_file = "test_after_changes.csv"
# x_test_predict.to_csv(output_file, index=False)

# categorization
x_test_predict['AGE'] = x_test_predict['AGE'].apply(lambda x: 'young' if 16 <= x < 32 else ('adult' if 32 <= x < 67 else 'old'))
x_test_predict['ANNUAL_MILEAGE'] = x_test_predict['ANNUAL_MILEAGE'].apply(lambda x: 'low' if x <= 10000 else ('middle' if x <= 15000 else 'many'))
x_test_predict['DRIVING_EXPERIENCE'] = x_test_predict['DRIVING_EXPERIENCE'].apply(lambda x: 'low' if x <= 5 else ('middle' if x <= 15 else 'high'))
x_test_predict['SPEEDING_VIOLATIONS'] = x_test_predict['SPEEDING_VIOLATIONS'].apply(lambda x: 'none' if x <= 0 else ('few' if x <= 5 else 'many'))
x_test_predict['PAST_ACCIDENTS'] = x_test_predict['PAST_ACCIDENTS'].apply(lambda x: 'none' if x <= 0 else ('few' if x <= 3 else 'many'))

mean = np.mean(x_test_predict['CREDIT_SCORE'])
std_dev = np.std(x_test_predict['CREDIT_SCORE'])

# Define thresholds based on standard deviations
low_threshold = mean - std_dev  # μ - σ
high_threshold = mean + std_dev  # μ + σ

# Categorize data based on thresholds
def categorize(score):
    if score < low_threshold: return 'Low'
    elif low_threshold <= score <= high_threshold: return 'Medium'
    else: return 'High'

x_test_predict['CREDIT_SCORE'] = pd.Series(x_test_predict['CREDIT_SCORE']).apply(categorize)
x_test_predict['GENDER'] = x_test_predict['GENDER'].map({'male': 0, 'female': 1})
x_test_predict['EDUCATION'] = x_test_predict['EDUCATION'].map({'high school': 1, 'university': 2, 'none': 3})
x_test_predict['INCOME'] = x_test_predict['INCOME'].map({'poverty': 1, 'working class': 2, 'middle class': 3, 'upper class' : 4})
x_test_predict['CREDIT_SCORE'] = x_test_predict['CREDIT_SCORE'].map({'Low': 1, 'Medium': 2, 'High': 3})
x_test_predict['VEHICLE_YEAR'] = x_test_predict['VEHICLE_YEAR'].map({'before 2015': 1, 'after 2015': 2})
x_test_predict['POSTAL_CODE'] = x_test_predict['POSTAL_CODE'].map({10238: 1, 32765: 2, 92101: 3, 21217 : 4})
x_test_predict['ANNUAL_MILEAGE'] = x_test_predict['ANNUAL_MILEAGE'].map({'low': 1, 'middle': 2, 'many': 3})
x_test_predict['AGE'] = x_test_predict['AGE'].map({'young': 1, 'adult': 2, 'old': 3})
x_test_predict['DRIVING_EXPERIENCE'] = x_test_predict['DRIVING_EXPERIENCE'].map({'low': 1, 'middle': 2, 'high': 3})
x_test_predict['SPEEDING_VIOLATIONS'] = x_test_predict['SPEEDING_VIOLATIONS'].map({'none': 1, 'few': 2, 'many': 3})
x_test_predict['PAST_ACCIDENTS'] = x_test_predict['PAST_ACCIDENTS'].map({'none': 1, 'few': 2, 'many': 3})

# -------------------------------------------- Make predictions --------------------------------------------------------
# arrenge by order of ID the rows before prediction
x_test_predict = x_test_predict.sort_values(by='ID', ascending=True)
# drop ID
x_test_predict = x_test_predict.drop('ID', axis=1)

# standardizing the data using StandardScaler
standard_scaler = StandardScaler()
x_test_predict = standard_scaler.fit_transform(x_test_predict)

y_test_predictions = final_model.predict(x_test_predict)
results_df = pd.DataFrame({'Predictions': y_test_predictions})

output_file = "CI_G7_ytest_NN.csv"
results_df.to_csv(output_file, index=False)
