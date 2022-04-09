import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, average_precision_score

from imblearn import over_sampling

from model import LRModel

# Load dataset
path_data = '/mnt/sharedfolder/data/'
cc_fraud = pd.read_csv(f'{path_data}creditcard.csv')

# Drop duplicates
cc_fraud.drop_duplicates(inplace=True)

# Clip the outliers
cc_fraud_f = cc_fraud.copy()

for col in cc_fraud.columns[1:-1]:
    Q1 = cc_fraud[col].quantile(0.25)
    Q3 = cc_fraud[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    cc_fraud_f[col] = cc_fraud[col].clip(lower_bound, upper_bound)

# Dealing with class imbalance using SMOTE method
features = list(cc_fraud.columns[:-1])
x = cc_fraud_f[features]
y = cc_fraud_f.Class

smote = over_sampling.SMOTE()

# Fit data for balancing
x_smote, y_smote = smote.fit_resample(x, y)

# convert to numpy
x = x_smote.to_numpy()
y = y_smote.to_numpy()

# Setting-up model
model = LRModel()

# Setting up scaler
model.scaler_fit(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 2)

# Apply transform to both the training set and the test set.
x_train = model.scaler.transform(x_train)
x_test = model.scaler.transform(x_test)


# Fit model
model.clf.fit(x_train, y_train)

# Evaluation
y_pred = model.clf.predict(x_test)
x_test_scores = model.clf.decision_function(x_test)

print('----------------------------------------------')
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Recall: {recall_score(y_test, y_pred)}')
print(f'Precision: {precision_score(y_test, y_pred)}')
print(f'F1: {f1_score(y_test, y_pred)}')
print(f'Average Precision (AUPRC): {average_precision_score(y_test, x_test_scores)}')
print('----------------------------------------------')

# Saving pickles of scaler and model
model.pickle_model()
model.pickle_scaler()