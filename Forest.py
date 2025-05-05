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

clf = RandomForestClassifier(n_estimators=75, max_depth=11,
                             oob_score=True, verbose=1)
clf.fit(X, y)

# Print out the results of the best classifier
print("Best Estimator Results:")
print(f"OOB score: {clf.oob_score_}")
print(f"Train score: {clf.score(X, y)}")

# Show the confusion matrix
cm = confusion_matrix(y, clf.predict(X))
disp = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
fig, ax = plt.subplots(figsize=(8,8))
disp.plot(ax=ax, xticks_rotation=90)
plt.title("Random Forest Results")

# Save and show figure

plt.tight_layout()
plt.savefig("Forest_Results.png")
plt.show()

# Bar plot for feature importance
importance = pd.DataFrame(clf.feature_importances_, index=df.columns[:-1])
importance.plot.bar()
plt.title("Feature Importance for the Random Forest")
plt.legend("", frameon=False)

# Save and show figure
plt.tight_layout()
plt.savefig("Forest_Importance.png")
plt.show()