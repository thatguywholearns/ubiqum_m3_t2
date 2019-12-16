# Ideas
# For one type of phone, calculate the distance for each individual WAP
# Calibrate for one phone model and then standardize signal for this one phone
# Refitting and updating parameters https://scikit-learn.org/stable/tutorial/basic/tutorial.html
# Merge train and validation and resample yourself/use CV
# Reverse prediction, use long, lat, floor and building to predict WAP signal strength

# Import libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set working directy, !set manually in top right corner
path = os.getcwd()
os.chdir(path) 

# Import data 
train = pd.read_csv(filepath_or_buffer = "./data/trainingData.csv")
val = pd.read_csv(filepath_or_buffer = "./data/validationData.csv")

# Check structure data
train.head()
train.describe()
train.info()

# Make col names lower case
train.columns = map(str.lower, train.columns)
val.columns = map(str.lower, val.columns)
names = list(train.columns)

# Check most dependant variables
long = train.loc[:,"longitude"]
lat = train.loc[:,"latitude"]
floor = train.loc[:,"floor"]
buildingid = train.loc[:,"buildingid"]
spaceid = train.loc[:,"spaceid"]
relativeposition = train.loc[:,"relativeposition"]

# Check distribution with histogram of most important variable
# !! Make histogram according to datatype, i.e. make one for categorical variables like id
plt.hist(long, rwidth=0.95)
plt.hist(lat, rwidth=0.95)
plt.hist(floor, rwidth=0.95)
plt.hist(buildingid, rwidth=0.95)
plt.hist(spaceid, rwidth=0.95)
plt.hist(relativeposition, rwidth=0.95)

# Check scatterplot long vs lat
plt.scatter(long, lat)

# Split in train and val data in features and Y data frames #519
train_feat = train.iloc[:, 0:520]
train_y = train.iloc[:, 520:len(train.columns)]
val_feat = val.iloc[:, 0:520]
val_y = val.iloc[:, 520:len(train.columns)]

# Add column indicating val or training data and row bind both label data frames
train_y["validation"] = 0
val_y["validation"] = 1
join_y = train_y.append(val_y)
train_feat["validation"] = 0
val_feat["validation"] = 1
join_feat = train_feat.append(val_y)

# visualize signal position with scatter plot in 3D
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xs= join_y.loc[:, "longitude"],
           ys= join_y.loc[:, "latitude"] ,
           zs= join_y.loc[:, "floor"],
           zdir= "z",
           c= join_y.loc[:, "floor"],
           cmap ="viridis")

# Save to csv
train_feat.to_csv("./generated_data/train_feat.csv", index=False)
train_y.to_csv("./generated_data/train_y.csv", index=False)
val_feat.to_csv("./generated_data/val_feat.csv", index=False)
val_y.to_csv("./generated_data/val_y.csv", index=False)
join_feat.to_csv("./generated_data/join_feat.csv", index=False)
join_y.to_csv("./generated_data/join_y.csv", index=False)




