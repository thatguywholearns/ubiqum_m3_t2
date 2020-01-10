# Import libraries
import os
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Set working directy, !set manually in top right corner
path = os.getcwd()
os.chdir(path) 

# Import data 
train = pd.read_csv(filepath_or_buffer = "./data/trainingData.csv")
val = pd.read_csv(filepath_or_buffer = "./data/validationData.csv")

# Select 80% of the validation data and add to training to enrich training data
val_80 = val.sample(frac = 0.80)
train = train.append(val_80)

# Remove that 80% from the validation data
val = pd.concat([val, val_80]).drop_duplicates(keep=False)

# Check structure data
train.head()
train.describe()
train.info()

# Make col names lower case
train.columns = map(str.lower, train.columns)
val.columns = map(str.lower, val.columns)
names = list(train.columns)

## Check most dependant variables
#long = train.loc[:,"longitude"]
#lat = train.loc[:,"latitude"]
#floor = train.loc[:,"floor"]
#buildingid = train.loc[:,"buildingid"]
#spaceid = train.loc[:,"spaceid"]
#relativeposition = train.loc[:,"relativeposition"]
#
## Check distribution with histogram of most important variable
## !! Make histogram according to datatype, i.e. make one for categorical variables like id
#plt.hist(long, rwidth=0.95)
#plt.hist(lat, rwidth=0.95)
#plt.hist(floor, rwidth=0.95)
#plt.hist(buildingid, rwidth=0.95)
#plt.hist(spaceid, rwidth=0.95)
#plt.hist(relativeposition, rwidth=0.95)

# Check scatterplot long vs lat
#plt.scatter(long, lat)

# Split in train and val data in features and Y data frames #519
train_feat = train.iloc[:, 0:520]
train_y = train.iloc[:, 520:len(train.columns)]
val_feat = val.iloc[:, 0:520]
val_y = val.iloc[:, 520:len(train.columns)]

# Check for zero values 
train_feat.eq(0).any().any()
val_feat.eq(0).any().any()

# 0 would be an abnormal powerful signal We replace it by the maximum meaningful signal -15)
train_feat = train_feat.replace(0, -15)

#Add column indicating val or training data and row bind both label data frames
train_y["validation"] = 0
val_y["validation"] = 1
join_y = train_y.append(val_y)
train_feat["validation"] = 0
val_feat["validation"] = 1

# Drop validation columns
train_feat = train_feat.drop("validation", axis = 1)
val_feat = val_feat.drop("validation", axis = 1)

# Save to csv
train_feat.to_csv("./generated_data/train/train_feat.csv", index=False)
train_y.to_csv("./generated_data/train/train_y.csv", index=False)
val_feat.to_csv("./generated_data/validation/val_feat.csv", index=False)
val_y.to_csv("./generated_data/validation/val_y.csv", index=False)

#-----------------------------Handeling 100 values-----------------------------#

# 1
# Replace 100's by large negative value
train_feat_repl = train_feat.replace([100], -105)
val_feat_repl = val_feat.replace([100], -105)

# Save to csv
train_feat_repl.to_csv("./generated_data/train/train_feat_repl.csv", index=False)
val_feat_repl.to_csv("./generated_data/validation/val_feat_repl.csv", index=False)

# 2
# Create a function that makes wifi signal strength relative for each instance
def as_percent(a_df):
    df_as_percent = abs(a_df)
    df_as_percent = df_as_percent.apply(lambda x: 1/x)
    df_as_percent = df_as_percent.replace([0.01], 0)
    df_as_percent = df_as_percent.apply(lambda df_as_percent: df_as_percent/df_as_percent.sum(), axis=1)
    return df_as_percent

train_feat_perc = as_percent(train_feat)
val_feat_perc = as_percent(val_feat)

# Save to csv
train_feat_perc.to_csv("./generated_data/train/train_feat_perc.csv", index=False)
val_feat_perc.to_csv("./generated_data/validation/val_feat_perc.csv", index=False)

# 3
# Create a function that L1 normalizes the wifi signal strength
def norm_l2(a_df):
    df_l2 = a_df.apply(lambda x: x + 105)
    df_l2 = df_l2.replace([205], 0)
    normalizer = Normalizer()
    df_l2 = normalizer.fit_transform(df_l2)
    df_l2 = pd.DataFrame(df_l2, columns = a_df.columns)
    return df_l2

train_feat_l2 = norm_l2(train_feat)
val_feat_l2 = norm_l2(val_feat)

# Save to csv
train_feat_l2.to_csv("./generated_data/train/train_feat_l2.csv", index=False)
val_feat_l2.to_csv("./generated_data/validation/val_feat_l2.csv", index=False)

# 4
# Create a function that L1 normalizes the wifi signal strength
def norm_l1(a_df):
    df_l1 = a_df.apply(lambda x: x + 105)
    df_l1 = df_l1.replace([205], 0)
    normalizer = Normalizer(norm="l1")
    df_l1 = normalizer.fit_transform(df_l1)
    df_l1 = pd.DataFrame(df_l1, columns = a_df.columns)
    return df_l1


train_feat_l1 = norm_l1(train_feat)
val_feat_l1 = norm_l1(val_feat)

# Save to csv
train_feat_l1.to_csv("./generated_data/train/train_feat_l1.csv", index=False)
val_feat_l1.to_csv("./generated_data/validation/val_feat_l1.csv", index=False)

# 5
# Replace 100's by NULL
train_feat_null = train_feat.replace([100], None)   
val_feat_null = val_feat.replace([100], None)

# Save to csv
train_feat_null.to_csv("./generated_data/train/train_feat_null.csv", index=False)
val_feat_null.to_csv("./generated_data/validation/val_feat_null.csv", index=False)

#---------------------------------Shifted Data--------------------------------#

# Humanize the data set by shifting all values to the positive side of the x-axis
train_feat_shifted = train_feat_repl + 105
val_feat_shifted = val_feat_repl + 105

train_feat_shifted_l2 = norm_l2(train_feat_shifted)
val_feat_shifted_l2 = norm_l2(val_feat_shifted)

# Save to csv
train_feat_l1.to_csv("./generated_data/train/train_feat_l1.csv", index=False)
val_feat_l1.to_csv("./generated_data/validation/val_feat_l1.csv", index=False)
