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

# Create the best estimator
clf = LinearSVC(C=7)
bClf = BaggingClassifier(clf, n_estimators=50, oob_score=True, verbose=2)
bClf.fit(X, y)

# Print result scores
print("Best Estimator Results:")
print(f"OOB score: {bClf.oob_score_}")
print(f"Score: {bClf.score(X, y)}")

# Generate and view the classifier's confusion matrix
cm = confusion_matrix(y, bClf.predict(X))
disp = ConfusionMatrixDisplay(cm, display_labels=bClf.classes_)
fig, ax = plt.subplots(figsize=(8,8))
disp.plot(ax=ax, xticks_rotation=90)
plt.title("Linear Support Vector Ensemble Results")

# Save and plot the confusion matrix
plt.tight_layout()
plt.savefig("LSV_Results.png")
plt.show()

