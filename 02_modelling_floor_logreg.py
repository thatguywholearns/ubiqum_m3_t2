#-----------------------------Import libraries-------------------------------#

# Import libraries
import os
import pandas as pd

# Import algorithms
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Set working directy, !set manually in top right corner
path = os.getcwd()
os.chdir(path) 

# Import data sets
train_x = pd.read_csv(filepath_or_buffer = "./generated_data/train/train_feat_l2_pca.csv")
train_y = pd.read_csv(filepath_or_buffer = "./generated_data/train/train_y.csv")
val_x = pd.read_csv(filepath_or_buffer = "./generated_data/validation/val_feat_l2_pca.csv")
val_y = pd.read_csv(filepath_or_buffer = "./generated_data/validation/val_y.csv")

#--------------------------------Model floor----------------------------------#

## We have 100% on floor, thus we use it as a feature
#train_x["buildingid"] = train_y["buildingid"]
#val_x["buildingid"] = val_y["buildingid"]

## We normalize the buildingid according to the normalization used for the signal
# ?? do we?

# Encode floors as categories
train_y["floor"] = pd.Categorical(train_y["floor"])
val_y["floor"] = pd.Categorical(val_y["floor"])

# Drop all labels beside floor and convert to numpy array, because logreg throws error otherwise
train_y = pd.DataFrame.to_numpy(train_y["floor"])
val_y = pd.DataFrame.to_numpy(val_y["floor"])

# Scale features
scaler = min_max_scaler()
train_x = scaler.fit_transform(train_x)
val_x = scaler.fit_transform(val_x)

# Create one-vs-rest logistic regression object
clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='newton-cg', penalty = "l2")

# Fit model
MulitLogReg = clf.fit(train_x, train_y)

# Make predcitions on validation set
pred = MulitLogReg.predict(val_x)

# Calculate performance metrics
clf_perf = classification_report(val_y, pred)
clf_accu = accuracy_score(val_y, pred)
conf_matrix = confusion_matrix(val_y, pred)
