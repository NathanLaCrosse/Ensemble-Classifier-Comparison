# Ensemble Classifier Comparison - Crop Selection
In this repository, we will be comparing two different machine learning ensemble algorithms and their performance on the Crop Recommendation Dataset.

Dataset Link: https://www.kaggle.com/datasets/varshitanalluri/crop-recommendation-dataset/data

The data in this dataset consists of regional variables and the type of crop best suited to those enviornmental conditions. For our models, they will 
be trained on the following data points: Soil Nitrogen Content, Soil Phosphorus Content, Soil Potassium Content, Temperature, Humidity, PH Value, Rainfall.

Let's start with a linear support vector machine classifier ensemble. This model consists of 50 linear support vector classifiers, which are creating some sort of
linear seperation of the dataset in a high dimensional space. Due to the ensemble nature, each of these classifiers gets a random subset of the data to fit on. Due to this
randomness, there is a high variance in the performance of the model. This makes performing a grid search for the optimal hyperparameter C difficult, so the value C=7 was
chosen as it was a midpoint in the result's range.

Let's look at the resulting confusion matrix when the model tested on the entire dataset:
![LSV_Results](https://github.com/user-attachments/assets/13c8f49b-3e9c-4cca-a90a-c5eae6e3a15b)

In general, the model does quite well in classifying the average data point. There are a few areas where it messes up though - mainly in classifying areas where jute should
be planted. This is likely due to the seperation between classes not being linear in the high dimensional space these data points lie in. 

Switching gears, let's see how a random forest (random tree ensemble) performs in classifying the data. The random trees used in this ensemble are even simpler than the linear
support vector machines, with each random tree essentially being a sequence of if statements that reduces the data's gini impurity. For more reading, reference https://en.wikipedia.org/wiki/Decision_tree_learning.
After performing a grid search based on the tree's depth, a value of max_depth=11 was chosen as it was a midpoint in the optimal range (there was variance in the results).

Here is the resulting confusion matrix after the random forest evaluated the entire dataset:

![Forest_Results](https://github.com/user-attachments/assets/7a6a5ff5-8a76-4b9b-a260-74082b72f90b)

This model performs much better than the linear support vector machine ensemble, with the model only getting confused when classying rice - and only missing 3 datapoints
at that. This difference in performance is likely due to the geometries present in the dataset's high dimensional space. The linear support vector machines can only create
linear seperations of the data, meanwhile the random forest can create nonlinear seperation of the data. This difference in how the two models seperate the data leads to
the random forest classifier performing better by every metric.

Another feature of the random forests is that they can be used to measure feature importance, essentially how important each variable is in the model's decision making
process. Here are the results from our model:

![Forest_Importance](https://github.com/user-attachments/assets/bec09da6-360c-4b0e-900e-9c412121fa2d)

So, the two most important variables in determining what plant should be planted are rainfall and humidity, which should make sense. Certain plants have evolved to live
in tropical regions and others evolved in drier, contential regions.

