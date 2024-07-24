import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#Filename [CHANGE THIS TO YOUR FILENAME FOR SUSY]
filename='./SUSY.csv'

#Read in SUSY File. We will only work with subset of data for demonstration purposes.

features=['SUSY','lepton 1 pT', 'lepton 1 eta', 'lepton 1 phi', 'lepton 2 pT', 'lepton 2 eta', 'lepton 2 phi', 
 'missing energy magnitude', 'missing energy phi', 'MET_rel', 'axial MET', 'M_R', 'M_TR_2', 'R', 'MT2', 
 'S_R', 'M_Delta_R', 'dPhi_r_b', 'cos(theta_r1)']

low_features=['lepton 1 pT', 'lepton 1 eta', 'lepton 1 phi', 'lepton 2 pT', 'lepton 2 eta', 'lepton 2 phi', 
 'missing energy magnitude', 'missing energy phi']

high_features=['MET_rel', 'axial MET', 'M_R', 'M_TR_2', 'R', 'MT2','S_R', 'M_Delta_R', 'dPhi_r_b', 'cos(theta_r1)']

#Number of datapoints to work with
N = 100000
print("Size of dataset : %i"%N)
df = pd.read_csv(filename, header=None,nrows=N,engine='python')
df.columns=features
y = df['SUSY'].values
X = df[[col for col in df.columns if col!="SUSY"]]

#Make datasets using only the 8 low-level features and 10 high-level features
X_low=X[low_features]
X_high=X[high_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=0)
X_low_train, X_low_test, y_low_train, y_low_test = train_test_split(X_low, y, test_size=.1, random_state=0)
X_high_train, X_high_test, y_high_train, y_high_test = train_test_split(X_high, y, test_size=.1, random_state=0)

from sklearn.metrics import roc_auc_score
import time
import xgboost as xgb
import warnings

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

print("Training on %i examples with %i features"%X_train.shape)

#Use default parameters and train on full dataset
XGBclassifier = xgb.sklearn.XGBClassifier(nthread=-1, seed=1, n_estimators=1000)
#Train and time classifier
start_time = time.time()
XGBclassifier.fit(X_train, y_train)
run_time = time.time() - start_time

#Make Predictions
print("Predicting on %i examples with %i features\n"%X_test.shape)
y_pred= XGBclassifier.predict(X_test)

#Print Results
print("Model Accuracy with all features: {:.2f}%".format(100*XGBclassifier.score(X_test, y_test)))
print("The AUC score with all features is {:.2f}".format(roc_auc_score(y_test,y_pred)))
print("Run time with all features: {:.2f} sec\n\n".format(run_time))


#Rerun with just low-level kinematic features with default parameters

print("Training on %i examples with %i features"%X_low_train.shape)
XGBclassifier_low = xgb.sklearn.XGBClassifier(nthread=-1, seed=1)
#Train and time classifier
start_time = time.time()
XGBclassifier_low.fit(X_low_train, y_low_train)
run_time = time.time() - start_time

#Make Predictions
print("Predicting on %i examples with %i features\n"%X_low_test.shape)
y_low_pred = XGBclassifier_low.predict(X_low_test)

#Print Results
print("Model Accuracy with just low-level kinematic features: {:.2f}%".format(100*XGBclassifier_low.score(X_low_test, y_low_test)))
print("The low-level AUC score is {:.2f}".format(roc_auc_score(y_test,y_low_pred)))
print("Run time with low-level features: {:.2f} sec\n\n".format(run_time))


#Rerun with just high-level kinematic features with default parameters

print("Training on %i examples with %i features\n"%X_high_train.shape)
XGBclassifier_high = xgb.sklearn.XGBClassifier(nthread=-1, seed=1)
#Train and time classifier
start_time = time.time()
XGBclassifier_high.fit(X_high_train, y_high_train)
run_time = time.time() - start_time

print("Training on %i examples with %i features"%X_high_test.shape)
#Make Predictions
y_high_pred = XGBclassifier_high.predict(X_high_test)

#Print Results
print("Model Accuracy with just high-level features: {:.2f}%".format(100*XGBclassifier_low.score(X_low_test, y_low_test)))
print("The high-level AUC score is {:.2f}".format(roc_auc_score(y_test,y_high_pred)))
print("Run time with high-level features: {:.2f} sec\n\n".format(run_time))

#import ml_style as style
import matplotlib as mpl
#mpl.rcParams.update(style.style)
import matplotlib.pyplot as plt

fig=plt.figure()
xgb.plot_importance(XGBclassifier, ax=plt.gca())
fig.subplots_adjust(left=0.4)  #
#fig.savefig('SUSYXGBoost1.pdf')

fig=plt.figure()
xgb.plot_importance(XGBclassifier_low, ax=plt.gca())
fig.subplots_adjust(left=0.4)
#fig.savefig('SUSYXGBoost2.pdf')
fig=plt.figure()
xgb.plot_importance(XGBclassifier_high, ax=plt.gca())
fig.subplots_adjust(left=0.4)
fig.savefig('SUSYXGBoost3.pdf')


from sklearn.metrics import roc_curve 

fpr, tpr, _ = roc_curve(y_test, y_pred)
fpr_low, tpr_low, _ = roc_curve(y_test, y_low_pred)
fpr_high, tpr_high, _ = roc_curve(y_test, y_high_pred)
plt.figure(1)
plt.plot(tpr, 1-fpr, label='Full')
plt.plot(tpr_low, 1-fpr_low, label='Low')
plt.plot(tpr_high, 1-fpr_high, label='High')
plt.legend(loc=1)
plt.xlabel('Signal efficiency')
plt.ylabel('Background Rejection')
plt.savefig("SUSY_roc_XGBoost.pdf")