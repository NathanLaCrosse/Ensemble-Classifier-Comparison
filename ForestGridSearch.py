import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("Crop_Recommendation.csv")

# Features - Nitrogen, Phosphorus, Potassium, Temperature, pH, rainfall
X = df.iloc[:,:-1].copy().to_numpy()

# Label - Most optimal crop
y = df.iloc[:,-1].copy().to_numpy()

# Create a random forest object to search on
clf = RandomForestClassifier(n_estimators=75, max_depth=3,
                             oob_score=True, verbose=1)

# Perform a grid search
params = {'max_depth':range(5, 20)}
grid_search = GridSearchCV(clf, params, cv=5)
grid_search.fit(X, y)

# Print out the results of the grid search
results = pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score', 'rank_test_score']]
print(results)
print("Best Params are: ", grid_search.best_params_)

# The grid search results had a lot of variance - there were two zones where
# the max depth seemed optimal -> 10-12 and 16-18, I chose 11 (a smaller depth)