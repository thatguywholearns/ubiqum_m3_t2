# Import libraries
import os
import pandas as pd
import seaborn as sn

# Import algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import cohen_kappa_score

# Set working directy, !set manually in top right corner
path = os.getcwd()
os.chdir(path) 

# Read csv and drop first to avoid reading index
train_x = pd.read_csv(filepath_or_buffer = "./generated_data/train_feat.csv")
train_y = pd.read_csv(filepath_or_buffer = "./generated_data/train_y.csv")
val_x = pd.read_csv(filepath_or_buffer = "./generated_data/val_feat.csv")
val_y = pd.read_csv(filepath_or_buffer = "./generated_data/val_y.csv")

# Create function that selects output variables
def get_y_var(df, a_list):
    return df[a_list]

# Initialize data frame and library that will be used 
accu = {}
kapp = {}
conf_matrix = {}
perf_clf = [accu, kapp, conf_matrix]
mse = {}
mea = {}
r_sqrd = {}
perf_reg = [mse, mea, r_sqrd]

# Select all variables that need prediction
pred_var = ["longitude", 
            "latitude", 
            "floor", 
            "buildingid"]

# Select all variables that need regression
reg_labels = ["longitude", 
              "latitude"]

# Loop over the variable you want to predict
for i in pred_var:
    # If i is label that requires regression, run regression else classification
    if i in reg_labels:
        # Select label you want to predict from train
        train_y_reg = get_y_var(train_y, i)
        # Select label you want to predict from validation
        val_y_reg = get_y_var(val_y, i)        
        # Initiate classifier
        rf = KNeighborsRegressor()
        # Fit model
        rf.fit(train_x, train_y_reg)
        # Predict values for validation data
        pred = rf.predict(val_x) 
        # Estimate mse, mea, r^2 and write to library for relevant i
        mse[i] = mean_squared_error(val_y_reg, pred)
        mea[i] = mean_absolute_error(val_y_reg, pred)
        r_sqrd[i] = r2_score(val_y_reg, pred)

    else:
        # Select label you want to predict from train
        train_y_clf = get_y_var(train_y, i)
        # Select label you want to predict from validation
        val_y_clf = get_y_var(val_y, i)        
        # Initiate classifier
        rf = KNeighborsClassifier()
        # Fit model
        rf.fit(train_x, train_y_clf)
        # Predict values for validation data
        pred = rf.predict(val_x) 
        # Estimate mse, mea, r^2 and write to library for relevant i
        accu[i] = accuracy_score(val_y_clf, pred)
        kapp[i] = cohen_kappa_score(val_y_clf, pred)
        conf_matrix[i] = confusion_matrix(val_y_clf, pred,)

perf_reg = pd.DataFrame(perf_reg, index=["mse", "mea", "r_sqrd"])
perf_clf = pd.DataFrame(perf_clf, index=["accu", "kapp", "conf_matrix"])

# Check Seaborn issue

# Train and fit classification models on clf data frames
#for i in range(len(train_y_clf.columns)):
#    # Initiate classifier
#    rf = RandomForestClassifier()
#    # Fit model
#    rf.fit(train_x, train_y_clf.iloc[:, i])
#    # Predict values for validation data
#    pred = rf.predict(val_x)
#    # Estimate accuracy, kappa and write to library for relevant i
#    accu[val_y_clf.columns[i]] = accuracy_score(val_y_clf.iloc[:, i], pred)
#    kapp[val_y_clf.columns[i]] = cohen_kappa_score(val_y_clf.iloc[:, i], pred)
#    conf_matrix = confusion_matrix(val_y_clf.iloc[:, i], pred)
#    sn.heatmap(conf_matrix, annot=True)
#    # Normalize 
#    conf_matrix_norm = conf_matrix.astype("float") / conf_matrix.sum(axis=1)
#    # Plot confusion matrix
#    sn.heatmap(conf_matrix_norm, annot=True)
