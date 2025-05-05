import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("Crop_Recommendation.csv")

# Features - Nitrogen, Phosphorus, Potassium, Temperature, pH, rainfall
X = df.iloc[:,:-1].copy().to_numpy()

# Label - Most optimal crop
y = df.iloc[:,-1].copy().to_numpy()

# Create a dummy svc classifier to change during the search
clf = LinearSVC(C=1.0)
bClf = BaggingClassifier(clf, n_estimators=50, oob_score=True, verbose=2)

# Perform a grid search
params = {'estimator__C':range(1,11)}
grid_search = GridSearchCV(bClf, params, cv=5)
grid_search.fit(X, y)

# Print out the results
results = pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score', 'rank_test_score']]
print(results)
print("Best Params are: ", grid_search.best_params_)

# Best C appears to be ~7
# There is also high variance in our results here, likely due to the nature
# of the ensemble setup, with each classifier only training on a subset of the data.