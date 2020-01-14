#-----------------------------Import libraries--------------------------------#

# Import libraries
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from scipy import mean
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.utils import resample
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

plt.style.use('ggplot')

#---------------------------------Load data-----------------------------------#

# Import data 
train = pd.read_csv(filepath_or_buffer = "./data/trainingData.csv")
val = pd.read_csv(filepath_or_buffer = "./data/validationData.csv")

# Make col names lower case
# ?? How can this be done with an apply function
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

# Extra functions
# Calculate 95% confidence interval for mean
def confidence_interval_mean(a_list, confidence_level = 0.95):   
    n = len(a_list)
    m = mean(a_list)
    std_err = sem(a_list)
    h = std_err * t.ppf((1 + confidence_level) / 2, n - 1)
    start = m - h
    end = m + h
    return [start, end]

#----------------------------Predict buildingid-------------------------------#

# Define number of iterations for bootstrapping
n_iterations = 2


# Random forest

# Create variables that will store boostrap results for random forest
buildingid_rf_accu = []
buildingid_rf_kappa = []
buildingid_rf_confmat = []
buildingid_rf_pred_aggr = pd.DataFrame()

for i in range(n_iterations):
    # Apply function to normalize both data sets
    train_x = norm_l2(train_feat)
    val_x = norm_l2(val_feat)
    # Resample train and validation data for bootstrapping
    train_x = resample(train_x, n_samples = int(len(train_x)*0.60), random_state = i)
    train_y = resample(train_y, n_samples = int(len(train_y)*0.60), random_state = i)
    val_x = resample(val_x, n_samples = int(len(val_x)*0.60), random_state = i)
    val_y = resample(val_y, n_samples = int(len(val_y)*0.60), random_state = i)
    # Initiate classifier
    rf = RandomForestClassifier(n_estimators=200)
    # Fit model
    rf.fit(train_x, train_y["buildingid"])
    # Predict values for validation data
    prediction = rf.predict(val_x)
    # Convert to data frame
    prediction = pd.DataFrame(prediction)
    # Save predictions
    buildingid_rf_pred_aggr = pd.concat([buildingid_rf_pred_aggr, prediction], axis = 1)                
    # Calculate accuracy
    buildingid_rf_accu.append(accuracy_score(val_y["buildingid"], prediction))
    # Calculate kappa
    buildingid_rf_kappa = cohen_kappa_score(val_y["buildingid"], prediction)
    # Calculate confusion matrix
    buildingid_rf_confmat = confusion_matrix(val_y["buildingid"], prediction)
    # Reset train_y and val_y
    train_y = train.iloc[:, 520:len(train.columns)]
    val_y = val.iloc[:, 520:len(train.columns)]

# Vizualize results
plt.hist(buildingid_rf_accu, bins = int(math.sqrt(len(buildingid_rf_accu))))

# Calculate confidence interval
confidence_buildingid_rf_acc =  confidence_interval_mean(buildingid_rf_accu)


# K nearest neighbours

# Create variables that will store boostrap results for random forest
buildingid_knn_accu = []
buildingid_knn_kappa = []
buildingid_knn_confmat = []
buildingid_knn_pred_aggr = pd.DataFrame()

for i in range(n_iterations):
    # Apply function to normalize both data sets
    train_x = norm_l2(train_feat)
    val_x = norm_l2(val_feat)
    # Resample train and validation data for bootstrapping
    train_x = resample(train_x, n_samples = int(len(train_x)*0.60), random_state = i)
    train_y = resample(train_y, n_samples = int(len(train_y)*0.60), random_state = i)
    val_x = resample(val_x, n_samples = int(len(val_x)*0.60), random_state = i)
    val_y = resample(val_y, n_samples = int(len(val_y)*0.60), random_state = i)
    # Initiate classifier
    knn = KNeighborsClassifier()
    # Fit model
    knn.fit(train_x, train_y["buildingid"])
    # Predict values for validation data
    prediction = knn.predict(val_x)
    # Convert to data frame
    prediction = pd.DataFrame(prediction)
    # Save predictions
    buildingid_knn_pred_aggr = pd.concat([buildingid_knn_pred_aggr, prediction], axis = 1)                
    # Calculate accuracy
    buildingid_knn_accu.append(accuracy_score(val_y["buildingid"], prediction))
    # Calculate kappa
    buildingid_knn_kappa = cohen_kappa_score(val_y["buildingid"], prediction)
    # Calculate confusion matrix
    buildingid_knn_confmat = confusion_matrix(val_y["buildingid"], prediction)
    # Reset train_y and val_y
    train_y = train.iloc[:, 520:len(train.columns)]
    val_y = val.iloc[:, 520:len(train.columns)]

# Vizualize results
plt.hist(buildingid_knn_accu, bins = int(math.sqrt(len(buildingid_knn_accu))))

# Calculate confidence interval
confidence_buildingid_knn_acc = confidence_interval_mean(buildingid_knn_accu)

 
# Support vector machine

# Create variables that will store boostrap results for random forest
buildingid_svm_accu = []
buildingid_svm_kappa = []
buildingid_svm_confmat = []
buildingid_svm_pred_aggr = pd.DataFrame()

for i in range(n_iterations):
    # Apply function to normalize both data sets
    train_x = norm_l2(train_feat)
    val_x = norm_l2(val_feat)
    # Resample train and validation data for bootstrapping
    train_x = resample(train_x, n_samples = int(len(train_x)*0.60), random_state = i)
    train_y = resample(train_y, n_samples = int(len(train_y)*0.60), random_state = i)
    val_x = resample(val_x, n_samples = int(len(val_x)*0.60), random_state = i)
    val_y = resample(val_y, n_samples = int(len(val_y)*0.60), random_state = i)
    # Initiate classifier
    svm = SVC(kernel='linear', C = 1.0)
    # Fit model
    svm.fit(train_x, train_y["buildingid"])
    # Predict values for validation data
    prediction = svm.predict(val_x)
    # Convert to data frame
    prediction = pd.DataFrame(prediction)
    # Save predictions
    buildingid_svm_pred_aggr = pd.concat([buildingid_svm_pred_aggr, prediction], axis = 1)                
    # Calculate accuracy
    buildingid_svm_accu.append(accuracy_score(val_y["buildingid"], prediction))
    # Calculate kappa
    buildingid_svm_kappa = cohen_kappa_score(val_y["buildingid"], prediction)
    # Calculate confusion matrix
    buildingid_svm_confmat = confusion_matrix(val_y["buildingid"], prediction)
    # Reset train_y and val_y
    train_y = train.iloc[:, 520:len(train.columns)]
    val_y = val.iloc[:, 520:len(train.columns)]

# Vizualize results
plt.hist(buildingid_svm_accu, bins = int(math.sqrt(len(buildingid_svm_accu))))

# Calculate confidence interval
confidence_buildingid_svm_acc = confidence_interval_mean(buildingid_svm_accu)


# Compare test predictions different models 
# ?? How to add axis?
list_model_perf_building = [buildingid_rf_accu, buildingid_knn_accu, buildingid_svm_accu]
my_xticks = ["Random Forrest", "Knn", "Svm"]
plt.xticks( np.array([1, 2, 3]), my_xticks)
plt.boxplot(list_model_perf_building)


# Save resluts
pd.DataFrame(list_model_perf_building).to_csv("./generated_data/list_model_perf_building.csv")

# Assign prediction you want to keep for buildingid

# Apply function to normalize both data sets
train_x = norm_l2(train_feat)
val_x = norm_l2(val_feat)
# Initiate classifier
knn = KNeighborsClassifier()
# Fit model
knn.fit(train_x, train_y["buildingid"])
# Predict values for validation data
pred_buildingid = knn.predict(val_x)

#----------------------------Predict floor------------------------------------#

# Random forest

# Create variables that will store boostrap results for random forest
floor_rf_accu = []
floor_rf_kappa = []
floor_rf_confmat = []
floor_rf_pred_aggr = pd.DataFrame()

for i in range(n_iterations):
    # Apply function to normalize both data sets
    train_x = norm_l2(train_feat)
    val_x = norm_l2(val_feat)
    # Resample train and validation data for bootstrapping
    train_x = resample(train_x, n_samples = int(len(train_x)*0.60), random_state = i)
    train_y = resample(train_y, n_samples = int(len(train_y)*0.60), random_state = i)
    val_x = resample(val_x, n_samples = int(len(val_x)*0.60), random_state = i)
    val_y = resample(val_y, n_samples = int(len(val_y)*0.60), random_state = i)
    # Initiate classifier
    rf = RandomForestClassifier(n_estimators=200)
    # Fit model
    rf.fit(train_x, train_y["floor"])
    # Predict values for validation data
    prediction = rf.predict(val_x)
    # Convert to data frame
    prediction = pd.DataFrame(prediction)
    # Save predictions
    floor_rf_pred_aggr = pd.concat([floor_rf_pred_aggr, prediction], axis = 1)                
    # Calculate accuracy
    floor_rf_accu.append(accuracy_score(val_y["floor"], prediction))
    # Calculate kappa
    floor_rf_kappa = cohen_kappa_score(val_y["floor"], prediction)
    # Calculate confusion matrix
    floor_rf_confmat = confusion_matrix(val_y["floor"], prediction)
    # Reset train_y and val_y
    train_y = train.iloc[:, 520:len(train.columns)]
    val_y = val.iloc[:, 520:len(train.columns)]

# Vizualize results
plt.hist(floor_rf_accu, bins = int(math.sqrt(len(floor_rf_accu))))

# Calculate confidence interval
confidence_floor_rf_acc =  confidence_interval_mean(floor_rf_accu)


# K nearest neighbours

# Create variables that will store boostrap results for random forest
floor_knn_accu = []
floor_knn_kappa = []
floor_knn_confmat = []
floor_knn_pred_aggr = pd.DataFrame()

for i in range(n_iterations):
    # Apply function to normalize both data sets
    train_x = norm_l2(train_feat)
    val_x = norm_l2(val_feat)
    # Resample train and validation data for bootstrapping
    train_x = resample(train_x, n_samples = int(len(train_x)*0.60), random_state = i)
    train_y = resample(train_y, n_samples = int(len(train_y)*0.60), random_state = i)
    val_x = resample(val_x, n_samples = int(len(val_x)*0.60), random_state = i)
    val_y = resample(val_y, n_samples = int(len(val_y)*0.60), random_state = i)
    # Initiate classifier
    knn = KNeighborsClassifier()
    # Fit model
    knn.fit(train_x, train_y["floor"])
    # Predict values for validation data
    prediction = knn.predict(val_x)
    # Convert to data frame
    prediction = pd.DataFrame(prediction)
    # Save predictions
    floor_knn_pred_aggr = pd.concat([floor_knn_pred_aggr, prediction], axis = 1)                
    # Calculate accuracy
    floor_knn_accu.append(accuracy_score(val_y["floor"], prediction))
    # Calculate kappa
    floor_knn_kappa = cohen_kappa_score(val_y["floor"], prediction)
    # Calculate confusion matrix
    floor_knn_confmat = confusion_matrix(val_y["floor"], prediction)
    # Reset train_y and val_y
    train_y = train.iloc[:, 520:len(train.columns)]
    val_y = val.iloc[:, 520:len(train.columns)]

# Vizualize results
plt.hist(floor_knn_accu, bins = int(math.sqrt(len(floor_knn_accu))))

# Calculate confidence interval
confidence_floor_knn_acc = confidence_interval_mean(floor_knn_accu)

 
# Support vector machine

# Create variables that will store boostrap results for random forest
floor_svm_accu = []
floor_svm_kappa = []
floor_svm_confmat = []
floor_svm_pred_aggr = pd.DataFrame()

for i in range(n_iterations):
    # Apply function to normalize both data sets
    train_x = norm_l2(train_feat)
    val_x = norm_l2(val_feat)
    # Resample train and validation data for bootstrapping
    train_x = resample(train_x, n_samples = int(len(train_x)*0.60), random_state = i)
    train_y = resample(train_y, n_samples = int(len(train_y)*0.60), random_state = i)
    val_x = resample(val_x, n_samples = int(len(val_x)*0.60), random_state = i)
    val_y = resample(val_y, n_samples = int(len(val_y)*0.60), random_state = i)
    # Initiate classifier
    svm = SVC(kernel='linear', C = 1.0)
    # Fit model
    svm.fit(train_x, train_y["floor"])
    # Predict values for validation data
    prediction = svm.predict(val_x)
    # Convert to data frame
    prediction = pd.DataFrame(prediction)
    # Save predictions
    floor_svm_pred_aggr = pd.concat([floor_svm_pred_aggr, prediction], axis = 1)                
    # Calculate accuracy
    floor_svm_accu.append(accuracy_score(val_y["floor"], prediction))
    # Calculate kappa
    floor_svm_kappa = cohen_kappa_score(val_y["floor"], prediction)
    # Calculate confusion matrix
    floor_svm_confmat = confusion_matrix(val_y["floor"], prediction)
    # Reset train_y and val_y and val_y
    train_y = train.iloc[:, 520:len(train.columns)]
    val_y = val.iloc[:, 520:len(train.columns)]

# Vizualize results
plt.hist(floor_svm_accu, bins = int(math.sqrt(len(floor_svm_accu))))

# Calculate confidence interval
confidence_floor_svm_acc = confidence_interval_mean(floor_svm_accu)


# Compare test predictions different models 
# ?? How to add axis?
list_model_perf_floor = [floor_rf_accu, floor_knn_accu, floor_svm_accu]
plt.boxplot(list_model_perf_floor)

# Save resluts
pd.DataFrame(list_model_perf_floor).to_csv("./generated_data/list_model_perf_floor.csv")


# Assign prediction you want to keep for floor

# Apply function to normalize both data sets
train_x = norm_l2(train_feat)
val_x = norm_l2(val_feat)
# Initiate classifier
knn = KNeighborsClassifier()
# Fit model
knn.fit(train_x, train_y["floor"])
# Predict values for validation data
pred_floor = knn.predict(val_x)

#--------------------------Predict latitude-----------------------------------#

# Add buildingid as feature
val_feat["buildingid_pred"] = pred_buildingid
train_feat = train_feat.join(train_y["buildingid"])

# Add floor as feature
val_feat["floor_pred"] = pred_floor
train_feat = train_feat.join(train_y["floor"])

# Random forest

# Create variables that will store boostrap results for random forest
lat_rf_mse = []
lat_rf_mae = []
lat_rf_r2 = []
lat_rf_pred_aggr = pd.DataFrame()

for i in range(n_iterations):
    print("Lat rf iter", i)
    # Apply function to normalize both data sets
    train_x = norm_l2(train_feat)
    val_x = norm_l2(val_feat)
    # Resample train and validation data for bootstrapping
    train_x = resample(train_x, n_samples = int(len(train_x)*0.60), random_state = i)
    train_y = resample(train_y, n_samples = int(len(train_y)*0.60), random_state = i)
    val_x = resample(val_x, n_samples = int(len(val_x)*0.60), random_state = i)
    val_y = resample(val_y, n_samples = int(len(val_y)*0.60), random_state = i)
    # Initiate classifier
    rf = RandomForestRegressor(n_estimators=100)
    # Fit model
    rf.fit(train_x, train_y["latitude"])
    # Predict values for validation data
    prediction = rf.predict(val_x)
    # Convert to data frame
    prediction = pd.DataFrame(prediction)
    # Save predictions
    lat_rf_pred_aggr = pd.concat([lat_rf_pred_aggr, prediction], axis = 1)                
    # Calculate accuracy
    lat_rf_mse.append(mean_squared_error(val_y["latitude"], prediction))
    # Calculate kappa
    lat_rf_mae.append(mean_absolute_error(val_y["latitude"], prediction))
    # Calculate confusion matrix
    lat_rf_r2.append(r2_score(val_y["latitude"], prediction))
    # Reset train_y and val_y
    train_y = train.iloc[:, 520:len(train.columns)]
    val_y = val.iloc[:, 520:len(train.columns)]

# Vizualize results
plt.hist(lat_rf_mae, bins = int(math.sqrt(len(lat_rf_mae))))

# Calculate confidence interval
confidence_floor_rf_acc =  confidence_interval_mean(lat_rf_mae)


# knn

# Create variables that will store boostrap results for random forest
lat_knn_mse = []
lat_knn_mae = []
lat_knn_r2 = []
lat_knn_pred_aggr = pd.DataFrame()

for i in range(n_iterations):
    
    print("Lat knn iter", i)

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
    
    # Drop new features because they are in old scale and replace them by their scaled version
    train_feat = train_feat.drop(["buildingid", "floor"], axis = 1)
    val_feat = val_feat.drop(["buildingid_pred", "floor_pred"], axis = 1)
    
    train_feat = train_feat.join(train_feat_non_wap)
    val_feat = val_feat.join(val_feat_non_wap)
        
    ## Run a for loop for testing the performance off different compenents
    #lat_knn_all_pca_mae = []
    #
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
    #    mae_lat_knn = mean_absolute_error(val_y["latitude"], pred_latitude_knn)
    #    lat_knn_all_pca_mae.append(mae_lat_knn)
    #    r_sqrd_lat_knn = r2_score(val_y["latitude"], pred_latitude_knn)
    #    
    ## Get optimal number of principal components
    #lat_principal_comp = np.argmin(lat_knn_all_pca_mae) + 1
    #    
    ## Convert to Data frame
    #lat_knn_all_pca_mae = pd.DataFrame(lat_knn_all_pca_mae)
    #
    ## Plot results and save plot
    #fig_lat_knn_all_pca_mae = plt.plot(lat_knn_all_pca_mae)
    #plt.title("latitude predictions - Principal Components")
    #plt.xlabel('Principal components')
    #plt.ylabel('MAE')
    #plt.axvline(57)
    #plt.text(64, 18, '57 components', rotation=90, )
    #plt.savefig("./plots/fig_lat_knn_all_pca_mae.png")
    #
    ## Save results to csv
    #lat_knn_all_pca_mae.to_csv("./generated_data/lat_knn_all_pca_accu.csv", index=False)
    
    # Apply function to normalize both data sets
    train_x = norm_l2(train_feat)
    val_x = norm_l2(val_feat)

    # Resample train and validation data for bootstrapping
    train_x = resample(train_x, n_samples = int(len(train_x)*0.60), random_state = i)
    train_y = resample(train_y, n_samples = int(len(train_y)*0.60), random_state = i)
    val_x = resample(val_x, n_samples = int(len(val_x)*0.60), random_state = i)
    val_y = resample(val_y, n_samples = int(len(val_y)*0.60), random_state = i)
       
    # Use PCA tp reduce dimensionality of the data set
    pca = PCA(n_components = 57)
    pca.fit(train_x)
    train_x = pca.transform(train_x)
    val_x = pca.transform(val_x)
    
    # Initiate classifier
    knn = KNeighborsRegressor()
    # Fit model
    knn.fit(train_x, train_y["latitude"])
    # Predict values for validation data
    prediction = knn.predict(val_x)
    # Convert to data frame
    prediction = pd.DataFrame(prediction)
    # Save predictions
    lat_knn_pred_aggr = pd.concat([lat_knn_pred_aggr, prediction], axis = 1)                
    # Calculate mse
    lat_knn_mse.append(mean_squared_error(val_y["latitude"], prediction))
    # Calculate kappa
    lat_knn_mae.append(mean_absolute_error(val_y["latitude"], prediction))
    # Calculate confusion matrix
    lat_knn_r2.append(r2_score(val_y["latitude"], prediction))
    # Reset train_y and val_y
    train_y = train.iloc[:, 520:len(train.columns)]
    val_y = val.iloc[:, 520:len(train.columns)]
    
# Save results knn with PCA
    
# Vizualize results
plt.hist(lat_knn_mae, bins = int(math.sqrt(len(lat_knn_mae))))

# Calculate confidence interval
confidence_lat_knn_mae =  confidence_interval_mean(lat_knn_mae)


    
# Support Vector Machine

# Create variables that will store boostrap results for random forest
lat_svm_mse = []
lat_svm_mae = []
lat_svm_r2 = []
lat_svm_pred_aggr = pd.DataFrame()

for i in range(n_iterations):
    print("Lat svm iter", i)
    # Apply function to normalize both data sets
    train_x = norm_l2(train_feat)
    val_x = norm_l2(val_feat)
    # Resample train and validation data for bootstrapping
    train_x = resample(train_x, n_samples = int(len(train_x)*0.60), random_state = i)
    train_y = resample(train_y, n_samples = int(len(train_y)*0.60), random_state = i)
    val_x = resample(val_x, n_samples = int(len(val_x)*0.60), random_state = i)
    val_y = resample(val_y, n_samples = int(len(val_y)*0.60), random_state = i)
    # Initiate classifier
    rf = SVR(kernel='rbf')
    # Fit model
    rf.fit(train_x, train_y["latitude"])
    # Predict values for validation data
    prediction = rf.predict(val_x)
    # Convert to data frame
    prediction = pd.DataFrame(prediction)
    # Save predictions
    lat_svm_pred_aggr = pd.concat([lat_svm_pred_aggr, prediction], axis = 1)                
    # Calculate accuracy
    lat_svm_mse.append(mean_squared_error(val_y["latitude"], prediction))
    # Calculate kappa
    lat_svm_mae.append(mean_absolute_error(val_y["latitude"], prediction))
    # Calculate confusion matrix
    lat_svm_r2.append(r2_score(val_y["latitude"], prediction))
    # Reset train_y and val_y
    train_y = train.iloc[:, 520:len(train.columns)]
    val_y = val.iloc[:, 520:len(train.columns)]

# Vizualize results
plt.hist(lat_svm_mae, bins = int(math.sqrt(len(lat_svm_mae))))

# Calculate confidence interval
confidence_floor_svm_acc =  confidence_interval_mean(lat_svm_mae)   


# Compare test predictions different models 
# ?? How to add axis?
list_model_perf_lat = [lat_rf_mae, lat_knn_mae]
plt.boxplot(list_model_perf_lat)

# Save resluts
pd.DataFrame(list_model_perf_lat).to_csv("./generated_data/list_model_perf_lat.csv")


# Assign prediction you want to keep for latitude

# Apply function to normalize both data sets
train_x = norm_l2(train_feat)
val_x = norm_l2(val_feat)
# Use PCA tp reduce dimensionality of the data set
pca = PCA(n_components = 57)
pca.fit(train_x)
train_x = pca.transform(train_x)
val_x = pca.transform(val_x)
# Initiate classifier
knn = KNeighborsRegressor()
# Fit model
knn.fit(train_x, train_y["latitude"])
# Predict values for validation data
pred_latitude = knn.predict(val_x)

#-------------------------Predict longitude-----------------------------------#

# Random forest

# Add buildingid as feature
val_feat["buildingid_pred"] = pred_buildingid
train_feat = train_feat.join(train_y["buildingid"])

# Add floor as feature
val_feat["floor_pred"] = pred_floor
train_feat = train_feat.join(train_y["floor"])

# Random forest

# Create variables that will store boostrap results for random forest
long_rf_mse = []
long_rf_mae = []
long_rf_r2 = []
long_rf_pred_aggr = pd.DataFrame()

for i in range(n_iterations):
    print("Long rf iter", i)
    # Apply function to normalize both data sets
    train_x = norm_l2(train_feat)
    val_x = norm_l2(val_feat)
    # Resample train and validation data for bootstrapping
    train_x = resample(train_x, n_samples = int(len(train_x)*0.60), random_state = i)
    train_y = resample(train_y, n_samples = int(len(train_y)*0.60), random_state = i)
    val_x = resample(val_x, n_samples = int(len(val_x)*0.60), random_state = i)
    val_y = resample(val_y, n_samples = int(len(val_y)*0.60), random_state = i)
    # Initiate classifier
    rf = RandomForestRegressor(n_estimators=100)
    # Fit model
    rf.fit(train_x, train_y["longitude"])
    # Predict values for validation data
    prediction = rf.predict(val_x)
    # Convert to data frame
    prediction = pd.DataFrame(prediction)
    # Save predictions
    long_rf_pred_aggr = pd.concat([long_rf_pred_aggr, prediction], axis = 1)                
    # Calculate accuracy
    long_rf_mse.append(mean_squared_error(val_y["longitude"], prediction))
    # Calculate kappa
    long_rf_mae.append(mean_absolute_error(val_y["longitude"], prediction))
    # Calculate confusion matrix
    long_rf_r2.append(r2_score(val_y["longitude"], prediction))
    # Reset train_y and val_y
    train_y = train.iloc[:, 520:len(train.columns)]
    val_y = val.iloc[:, 520:len(train.columns)]

# Vizualize results
plt.hist(long_rf_mae, bins = int(math.sqrt(len(long_rf_mae))))

# Calculate confidence interval
confidence_floor_rf_acc =  confidence_interval_mean(long_rf_mae)


# knn

# Create variables that will store boostrap results for random forest
long_knn_mse = []
long_knn_mae = []
long_knn_r2 = []
long_knn_pred_aggr = pd.DataFrame()

for i in range(n_iterations):

    print("Long knn iter", i)

    
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
    
    # Drop new features because they are in old scale and replace them by their scaled version
    train_feat = train_feat.drop(["buildingid", "floor"], axis = 1)
    val_feat = val_feat.drop(["buildingid_pred", "floor_pred"], axis = 1)
    
    train_feat = train_feat.join(train_feat_non_wap)
    val_feat = val_feat.join(val_feat_non_wap)
        
    ## Run a for loop for testing the performance off different compenents
    #long_knn_all_pca_mae = []
    #
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
    #    mae_long_knn = mean_absolute_error(val_y["longitude"], pred_longitude_knn)
    #    long_knn_all_pca_mae.append(mae_long_knn)
    #    r_sqrd_long_knn = r2_score(val_y["longitude"], pred_longitude_knn)
    #    
    ## Get optimal number of principal components
    #long_principal_comp = np.argmin(long_knn_all_pca_mae) + 1
    #    
    ## Convert to Data frame
    #long_knn_all_pca_mae = pd.DataFrame(long_knn_all_pca_mae)
    #
    ## Plot results and save plot
    #fig_long_knn_all_pca_mae = plt.plot(long_knn_all_pca_mae)
    #plt.title("longitude predictions - Principal Components")
    #plt.xlabel('Principal components')
    #plt.ylabel('MAE')
    #plt.axvline(57)
    #plt.text(64, 18, '57 components', rotation=90, )
    #plt.savefig("./plots/fig_long_knn_all_pca_mae.png")
    #
    ## Save results to csv
    #long_knn_all_pca_mae.to_csv("./generated_data/long_knn_all_pca_accu.csv", index=False)
    
    # Apply function to normalize both data sets
    train_x = norm_l2(train_feat)
    val_x = norm_l2(val_feat)

    # Resample train and validation data for bootstrapping
    train_x = resample(train_x, n_samples = int(len(train_x)*0.60), random_state = i)
    train_y = resample(train_y, n_samples = int(len(train_y)*0.60), random_state = i)
    val_x = resample(val_x, n_samples = int(len(val_x)*0.60), random_state = i)
    val_y = resample(val_y, n_samples = int(len(val_y)*0.60), random_state = i)
       
    # Use PCA tp reduce dimensionality of the data set
    pca = PCA(n_components = 57)
    pca.fit(train_x)
    train_x = pca.transform(train_x)
    val_x = pca.transform(val_x)
    
    # Initiate classifier
    knn = KNeighborsRegressor()
    # Fit model
    knn.fit(train_x, train_y["longitude"])
    # Predict values for validation data
    prediction = knn.predict(val_x)
    # Convert to data frame
    prediction = pd.DataFrame(prediction)
    # Save predictions
    long_knn_pred_aggr = pd.concat([long_knn_pred_aggr, prediction], axis = 1)                
    # Calculate mse
    long_knn_mse.append(mean_squared_error(val_y["longitude"], prediction))
    # Calculate kappa
    long_knn_mae.append(mean_absolute_error(val_y["longitude"], prediction))
    # Calculate confusion matrix
    long_knn_r2.append(r2_score(val_y["longitude"], prediction))
    # Reset train_y and val_y
    train_y = train.iloc[:, 520:len(train.columns)]
    val_y = val.iloc[:, 520:len(train.columns)]
    
    
plt.hist(long_knn_mae, bins = int(math.sqrt(len(long_knn_mae))))

# Calculate confidence interval
confidence_long_knn_mae =  confidence_interval_mean(long_knn_mae)


# Support Vector Machine

# Create variables that will store boostrap results for random forest
long_svm_mse = []
long_svm_mae = []
long_svm_r2 = []
long_svm_pred_aggr = pd.DataFrame()

for i in range(n_iterations):
    print("Long svm iter", i)
    # Apply function to normalize both data sets
    train_x = norm_l2(train_feat)
    val_x = norm_l2(val_feat)
    # Resample train and validation data for bootstrapping
    train_x = resample(train_x, n_samples = int(len(train_x)*0.60), random_state = i)
    train_y = resample(train_y, n_samples = int(len(train_y)*0.60), random_state = i)
    val_x = resample(val_x, n_samples = int(len(val_x)*0.60), random_state = i)
    val_y = resample(val_y, n_samples = int(len(val_y)*0.60), random_state = i)
    # Initiate classifier
    rf = SVR(kernel='linear', C = 1.0)
    # Fit model
    rf.fit(train_x, train_y["longitude"])
    # Predict values for validation data
    prediction = rf.predict(val_x)
    # Convert to data frame
    prediction = pd.DataFrame(prediction)
    # Save predictions
    long_svm_pred_aggr = pd.concat([long_svm_pred_aggr, prediction], axis = 1)                
    # Calculate accuracy
    long_svm_mse.append(mean_squared_error(val_y["longitude"], prediction))
    # Calculate kappa
    long_svm_mae.append(mean_absolute_error(val_y["longitude"], prediction))
    # Calculate confusion matrix
    long_svm_r2.append(r2_score(val_y["longitude"], prediction))
    # Reset train_y and val_y
    train_y = train.iloc[:, 520:len(train.columns)]
    val_y = val.iloc[:, 520:len(train.columns)]

# Vizualize results
plt.hist(long_svm_mae, bins = int(math.sqrt(len(long_svm_mae))))

# Calculate confidence interval
confidence_floor_svm_acc =  confidence_interval_mean(long_svm_mae)   

# Compare test predictions different models 
# ?? How to add axis?
list_model_perf_long = [long_rf_mae, long_knn_mae]
plt.boxplot(list_model_perf_long)

# Save resluts
pd.DataFrame(list_model_perf_long).to_csv("./generated_data/list_model_perf_long.csv")

    
# Assign prediction you want to keep for longitude

# Apply function to normalize both data sets
train_x = norm_l2(train_feat)
val_x = norm_l2(val_feat)
# Use PCA tp reduce dimensionality of the data set
pca = PCA(n_components = 57)
pca.fit(train_x)
train_x = pca.transform(train_x)
val_x = pca.transform(val_x)
# Initiate classifier
knn = KNeighborsRegressor()
# Fit model
knn.fit(train_x, train_y["longitude"])
# Predict values for validation data
pred_longitude = knn.predict(val_x)

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

