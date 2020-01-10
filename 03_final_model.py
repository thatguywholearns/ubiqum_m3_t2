#-----------------------------Import libraries--------------------------------#

# Import libraries
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from math import sqrt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import cohen_kappa_score


#-----------------------------Set-up environment------------------------------#

# Set working directy, !set manually in top right corner in Spyder
path = os.getcwd()
os.chdir(path) 

#---------------------------------Load data-----------------------------------#

# Import data 
train = pd.read_csv(filepath_or_buffer = "./data/trainingData.csv")
val = pd.read_csv(filepath_or_buffer = "./data/validationData.csv")

# Make col names lower case
train.columns = map(str.lower, train.columns)
val.columns = map(str.lower, val.columns)

# Select 80% of the validation data and add to training to enrich training data
val_80 = val.sample(frac = 0.80, random_state = 0)
train = train.append(val_80)

# Remove that 80% from the validation data
val = pd.concat([val, val_80]).drop_duplicates(keep=False)

# Reset index for both data frames
train.reset_index(drop = True, inplace = True)
val.reset_index(drop = True, inplace = True)

# Split in train and val data in features and Y data frames #519
train_feat = train.iloc[:, 0:520]
train_y = train.iloc[:, 520:len(train.columns)]
val_feat = val.iloc[:, 0:520]
val_y = val.iloc[:, 520:len(train.columns)]

#------------------------------Pre processing---------------------------------#

# Value of 0 would be an abnormal powerful signal We replace it by the maximum meaningful signal -15)
train_feat = train_feat.replace(0, -15)

# There are several factors i.e. phone, model, time of mesearument, height that
# make instance less comparable. We define several functions that would make the 
# instances comparable

# 1
# Create a function that L2 normalizes the wifi signal strength
def norm_l2(a_df):
    df_l2 = a_df.apply(lambda x: x + 105)
    df_l2 = df_l2.replace([205], 0)
    normalizer = Normalizer()
    df_l2 = normalizer.fit_transform(df_l2)
    df_l2 = pd.DataFrame(df_l2, columns = a_df.columns)
    return df_l2

# 2
# Create a function that L1 normalizes the wifi signal strength
def norm_l1(a_df):
    df_l1 = a_df.apply(lambda x: x + 105)
    df_l1 = df_l1.replace([205], 0)
    normalizer = Normalizer(norm="l1")
    df_l1 = normalizer.fit_transform(df_l1)
    df_l1 = pd.DataFrame(df_l1, columns = a_df.columns)
    return df_l1

# 3
# Create a function that makes wifi signal strength relative for each instance
def as_percent(a_df):
    df_as_percent = abs(a_df)
    df_as_percent = df_as_percent.apply(lambda x: 1/x)
    df_as_percent = df_as_percent.replace([0.01], 0)
    df_as_percent = df_as_percent.apply(lambda df_as_percent: df_as_percent/df_as_percent.sum(), axis=1)
    return df_as_percent

#----------------------------Predict buildingid-------------------------------#

# Apply function to normalize both data sets
train_x = norm_l2(train_feat)
val_x = norm_l2(val_feat)

# Random Forest

# Initiate classifier
rf = RandomForestClassifier(n_estimators=200)

# Fit model
rf.fit(train_x, train_y["buildingid"])

# Predict values for validation data
pred_buildingid_rf = rf.predict(val_x)
     
# Estimate accuracy metrics
accu_buildingid_rf = accuracy_score(val_y["buildingid"], pred_buildingid_rf)
kapp_buildingid_rf = cohen_kappa_score(val_y["buildingid"], pred_buildingid_rf)
conf_matrix_buildingid_rf = confusion_matrix(val_y["buildingid"], pred_buildingid_rf)

# Write to summary data frame
perf_buildingid_rf = [accu_buildingid_rf, kapp_buildingid_rf]


# knn 

# Initiate classifier
knn = KNeighborsClassifier()

# Fit model
knn.fit(train_x, train_y["buildingid"])

# Predict values for validation data
pred_buildingid_knn = knn.predict(val_x)
     
# Perfromance metrics
accu_buildingid_knn = accuracy_score(val_y["buildingid"], pred_buildingid_knn)
kapp_buildingid_knn = cohen_kappa_score(val_y["buildingid"], pred_buildingid_knn)
conf_matrix_buildingid_knn = confusion_matrix(val_y["buildingid"], pred_buildingid_knn)

# Write to summary data frame
perf_buildingid_knn = [accu_buildingid_knn, kapp_buildingid_knn]


# Support vector machine

# Initiate classifier
svm = SVC(kernel='linear', C = 1.0)

# Fit model
svm.fit(train_x, train_y["buildingid"])

# Predict values for validation data
pred_buildingid_svm = svm.predict(val_x)

# Perfromance metrics
accu_buildingid_svm = accuracy_score(val_y["buildingid"], pred_buildingid_svm)
kapp_buildingid_svm = cohen_kappa_score(val_y["buildingid"], pred_buildingid_svm)
conf_matrix_buildingid_svm = confusion_matrix(val_y["buildingid"], pred_buildingid_svm)

# Write to summary data frame
perf_buildingid_svm = [accu_buildingid_svm, kapp_buildingid_svm]


# Assign prediction you want to keep for buildingid
pred_buildingid = pred_buildingid_svm

#------------------------------Pre processing---------------------------------#

## Left cascading model out of modeling for floor because results where worse

## Add buildingid as feature
#val_feat["buildingid"] = pred_buildingid
#train_feat = train_feat.join(train_y["buildingid"])
#
## L2 normalize the 
#train_x = norm_l2(train_feat)
#val_x = norm_l2(val_feat)

#----------------------------Predict floor------------------------------------#

# Random Forest

# Initiate classifier
rf = RandomForestClassifier(n_estimators=200)

# Fit model
rf.fit(train_x, train_y["floor"])

# Predict values for validation data
pred_floor_rf = rf.predict(val_x)
     
# Estimate accuracy metrics
accu_floor_rf = accuracy_score(val_y["floor"], pred_floor_rf)
kapp_floor_rf = cohen_kappa_score(val_y["floor"], pred_floor_rf)
conf_matrix_floor_rf = confusion_matrix(val_y["floor"], pred_floor_rf)

# Write to summary data frame
perf_floor_rf = [accu_floor_rf, kapp_floor_rf]


# knn 

# Initiate classifier
knn = KNeighborsClassifier()

# Fit model
knn.fit(train_x, train_y["floor"])

# Predict values for validation data
pred_floor_knn = knn.predict(val_x)
     
# Perfromance metrics
accu_floor_knn = accuracy_score(val_y["floor"], pred_floor_knn)
kapp_floor_knn = cohen_kappa_score(val_y["floor"], pred_floor_knn)
conf_matrix_floor_knn = confusion_matrix(val_y["floor"], pred_floor_knn)

# Write to summary data frame
perf_floor_knn = [accu_floor_knn, kapp_floor_knn]


# Support vector machine

# Initiate classifier
svm = SVC(kernel='linear', C = 1.0)

# Fit model
svm.fit(train_x, train_y["floor"])

# Predict values for validation data
pred_floor_svm = svm.predict(val_x)

# Perfromance metrics
accu_floor_svm = accuracy_score(val_y["floor"], pred_floor_svm)
kapp_floor_svm = cohen_kappa_score(val_y["floor"], pred_floor_svm)
conf_matrix_floor_svm = confusion_matrix(val_y["floor"], pred_floor_svm)

# Write to summary data frame
perf_floor_svm = [accu_floor_svm, kapp_floor_svm]


# Assign prediction you want to keep for floor
pred_floor = pred_floor_rf

#--------------------------Predict latitude-----------------------------------#

# Add buildingid as feature
val_feat["buildingid_pred"] = pred_buildingid
train_feat = train_feat.join(train_y["buildingid"])

# Add floor as feature
val_feat["floor_pred"] = pred_floor
train_feat = train_feat.join(train_y["floor"])

# Random forest

# Apply function to normalize both data sets
train_x = norm_l2(train_feat)
val_x = norm_l2(val_feat)

## Use PCA tp reduce dimensionality of the data set
#pca = PCA(n_components = 200)
#pca.fit(train_feat)
#train_x = pca.transform(train_x)
#val_x = pca.transform(val_x)

# Initiate regressor
rf = RandomForestRegressor(n_estimators=200)

# Fit model
rf.fit(train_x, train_y["latitude"])

# Predict values for validation data
pred_latitude_rf = rf.predict(val_x)

# Perfromance metrics     
mse_lat_rf = mean_squared_error(val_y["latitude"], pred_latitude_rf)
mea_lat_rf = mean_absolute_error(val_y["latitude"], pred_latitude_rf)
r_sqrd_lat_rf = r2_score(val_y["latitude"], pred_latitude_rf)

# Write to summary data frame
perf_latitude_rf = [mea_lat_rf, mse_lat_rf]


# knn

# Drop new features because they have been L2 normalized and want to scale them first for knn
train_feat = train_feat.drop(["buildingid", "floor"], axis = 1)
val_feat = val_feat.drop(["buildingid_pred", "floor_pred"], axis = 1)

# Add buildingid as feature
train_feat = train_feat.join(train_y["buildingid"])
val_feat["buildingid_pred"] = pred_buildingid

# Add floor as feature
train_feat = train_feat.join(train_y["floor"])
val_feat["floor_pred"] = pred_floor

# Scale non wap features to 0 and 1 scale
scaler = MinMaxScaler()
train_feat_non_wap = scaler.fit_transform(train_feat.loc[:, ["buildingid", "floor"]])
val_feat_non_wap = scaler.fit_transform(val_feat.loc[:, ["buildingid_pred", "floor_pred"]])

# Transform to data frames
train_feat_non_wap = pd.DataFrame(train_feat_non_wap, columns = ["buildingid",  "floor"])
val_feat_non_wap = pd.DataFrame(val_feat_non_wap, columns = ["buildingid_pred",  "floor_pred"])

# Drop new features because and replace them by their scaled version
train_feat = train_feat.drop(["buildingid", "floor"], axis = 1)
val_feat = val_feat.drop(["buildingid_pred", "floor_pred"], axis = 1)

train_feat = train_feat.join(train_feat_non_wap)
val_feat = val_feat.join(val_feat_non_wap)

# Run a for loop for testing the performance off different compenents

#for i in range(1, len(train_feat.columns)):
#    # Apply function to normalize both data sets
#    train_x = norm_l2(train_feat)
#    val_x = norm_l2(val_feat)
#       
#    # Use PCA tp reduce dimensionality of the data set
#    pca = PCA(n_components = i)
#    pca.fit(train_x)
#    train_x = pca.transform(train_x)
#    val_x = pca.transform(val_x)
#    
#    # Initiate regressor
#    knn = KNeighborsRegressor()
#    
#    # Fit model
#    knn.fit(train_x, train_y["latitude"])
#    
#    # Predict values for validation data
#    pred_latitude_knn = knn.predict(val_x)
#    
#    # Perfromance metrics     
#    mse_lat_knn = mean_squared_error(val_y["latitude"], pred_latitude_knn)
#    mea_lat_knn = mean_absolute_error(val_y["latitude"], pred_latitude_knn)
#    print(i, " ", mea_lat_knn)
#    r_sqrd_lat_knn = r2_score(val_y["latitude"], pred_latitude_knn)
#    
#    # Write to summary data frame
#    perf_lat_knn = [mea_lat_knn, mse_lat_knn]

# Apply function to normalize both data sets
train_x = norm_l2(train_feat)
val_x = norm_l2(val_feat)
   
# Use PCA tp reduce dimensionality of the data set
pca = PCA(n_components = 57)
pca.fit(train_x)
train_x = pca.transform(train_x)
val_x = pca.transform(val_x)

# Initiate regressor
knn = KNeighborsRegressor()

# Fit model
knn.fit(train_x, train_y["latitude"])

# Predict values for validation data
pred_latitude_knn = knn.predict(val_x)

# Perfromance metrics     
mse_lat_knn = mean_squared_error(val_y["latitude"], pred_latitude_knn)
mea_lat_knn = mean_absolute_error(val_y["latitude"], pred_latitude_knn)
r_sqrd_lat_knn = r2_score(val_y["latitude"], pred_latitude_knn)

# Write to summary data frame
perf_lat_knn = [mea_lat_knn, mse_lat_knn]


# Support vector machine

# Initiate regressor
svm_reg = SVR(kernel='rbf')

# Fit model
svm_reg.fit(train_x, train_y["latitude"])

# Predict values for validation data
pred_latitude_svm = svm_reg.predict(val_x)

# Perfromance metrics
mse_latitude_svm = mean_squared_error(val_y["latitude"], pred_latitude_svm)
mae_latitude_svm = mean_absolute_error(val_y["latitude"], pred_latitude_svm)
r_sqrd_latitude_svm = r2_score(val_y["latitude"], pred_latitude_svm)

# Write to summary data frame
perf_latitude_svm = [mae_latitude_svm, mse_latitude_svm]


# Assign prediction you want to keep for floor
pred_latitude = pred_latitude_knn

#-------------------------Predict longitude-----------------------------------#

# Random forest

# Drop new features because they have been minmax scaled
train_feat = train_feat.drop(["buildingid", "floor"], axis = 1)
val_feat = val_feat.drop(["buildingid_pred", "floor_pred"], axis = 1)

# Add buildingid as feature
train_feat = train_feat.join(train_y["buildingid"])
val_feat["buildingid_pred"] = pred_buildingid

# Add floor as feature
train_feat = train_feat.join(train_y["floor"])
val_feat["floor_pred"] = pred_floor

# Apply function to normalize both data sets
train_x = norm_l2(train_feat)
val_x = norm_l2(val_feat)

# Initiate regressor
rf = RandomForestRegressor(n_estimators=200)

# Fit model
rf.fit(train_x, train_y["longitude"])

# Predict values for validation data
pred_longitude_rf = rf.predict(val_x)

# Perfromance metrics     
mse_long_rf = mean_squared_error(val_y["longitude"], pred_longitude_rf)
mea_long_rf = mean_absolute_error(val_y["longitude"], pred_longitude_rf)
r_sqrd_long_rf = r2_score(val_y["longitude"], pred_longitude_rf)

# Write to summary data frame
perf_longitude_rf = [mea_long_rf, mse_long_rf]


# knn

# Drop new features because they have been L2 normalized and want to scale them first for knn
train_feat = train_feat.drop(["buildingid", "floor"], axis = 1)
val_feat = val_feat.drop(["buildingid_pred", "floor_pred"], axis = 1)

# Add buildingid as feature
train_feat = train_feat.join(train_y["buildingid"])
val_feat["buildingid_pred"] = pred_buildingid

# Add floor as feature
train_feat = train_feat.join(train_y["floor"])
val_feat["floor_pred"] = pred_floor

# Scale non wap features to 0 and 1 scale
scaler = MinMaxScaler()
train_feat_non_wap = scaler.fit_transform(train_feat.loc[:, ["buildingid", "floor"]])
val_feat_non_wap = scaler.fit_transform(val_feat.loc[:, ["buildingid_pred", "floor_pred"]])

# Transform to data frames
train_feat_non_wap = pd.DataFrame(train_feat_non_wap, columns = ["buildingid",  "floor"])
val_feat_non_wap = pd.DataFrame(val_feat_non_wap, columns = ["buildingid_pred",  "floor_pred"])

# Drop new features because and replace them by their scaled version
train_feat = train_feat.drop(["buildingid", "floor"], axis = 1)
val_feat = val_feat.drop(["buildingid_pred", "floor_pred"], axis = 1)

train_feat = train_feat.join(train_feat_non_wap)
val_feat = val_feat.join(val_feat_non_wap)

# Run a for loop for testing the performance off different compenents

#for i in range(1, len(train_feat.columns)):
#    # Apply function to normalize both data sets
#    train_x = norm_l2(train_feat)
#    val_x = norm_l2(val_feat)
#       
#    # Use PCA tp reduce dimensionality of the data set
#    pca = PCA(n_components = i)
#    pca.fit(train_x)
#    train_x = pca.transform(train_x)
#    val_x = pca.transform(val_x)
#    
#    # Initiate regressor
#    knn = KNeighborsRegressor()
#    
#    # Fit model
#    knn.fit(train_x, train_y["longitude"])
#    
#    # Predict values for validation data
#    pred_longitude_knn = knn.predict(val_x)
#    
#    # Perfromance metrics     
#    mse_long_knn = mean_squared_error(val_y["longitude"], pred_longitude_knn)
#    mea_long_knn = mean_absolute_error(val_y["longitude"], pred_longitude_knn)
#    print(i, " ", mea_long_knn)
#    r_sqrd_long_knn = r2_score(val_y["longitude"], pred_longitude_knn)


# Apply function to normalize both data sets
train_x = norm_l2(train_feat)
val_x = norm_l2(val_feat)
   
# Use PCA tp reduce dimensionality of the data set
pca = PCA(n_components = 57)
pca.fit(train_x)
train_x = pca.transform(train_x)
val_x = pca.transform(val_x)

# Initiate regressor
knn = KNeighborsRegressor()

# Fit model
knn.fit(train_x, train_y["longitude"])

# Predict values for validation data
pred_longitude_knn = knn.predict(val_x)

# Perfromance metrics     
mse_long_knn = mean_squared_error(val_y["longitude"], pred_longitude_knn)
mea_long_knn = mean_absolute_error(val_y["longitude"], pred_longitude_knn)
r_sqrd_long_knn = r2_score(val_y["longitude"], pred_longitude_knn)

# Write to summary data frame
perf_long_knn = [mea_long_knn, mse_long_knn]


# Support vector machine

# Initiate regressor
svm_reg = SVR(kernel='rbf')

# Fit model
svm_reg.fit(train_x, train_y["longitude"])

# Predict values for validation data
pred_latitude_svm = svm_reg.predict(val_x)

# Perfromance metrics
mse_long_svm = mean_squared_error(val_y["longitude"], pred_latitude_svm)
mae_long_svm = mean_absolute_error(val_y["longitude"], pred_latitude_svm)
r_sqrd_long_svm = r2_score(val_y["longitude"], pred_latitude_svm)

# Write to summary data frame
perf_latitude_svm = [mae_latitude_svm, mse_latitude_svm]


# Assign prediction you want to keep for floor
pred_longitude = pred_longitude_knn

#------------------------Calculate competition score--------------------------#

# Calculate the Euclidian distance between prediction and actual value
distance_squared = (val_y.loc[:, "latitude"] - pred_latitude)**2 + (val_y.loc[:, "longitude"] - pred_longitude)**2
distance_root = distance_squared.apply(lambda x: sqrt(x))

# Calculate penalyty for misclassifying floor
floor = abs(pred_floor - val_y.loc[:, "floor"]) * 4

# Calculate penalyty for misclassifying building
building = pred_buildingid != val_y.loc[:, "buildingid"]
building = building.apply(lambda x: int(x)) * 50

# Calculate final score using distance, floor, and building
distance_75 = np.percentile(distance_root + floor + building, 75)

