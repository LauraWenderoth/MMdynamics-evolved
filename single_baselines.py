import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score, accuracy_score,balanced_accuracy_score
def eval_all_classifier(X_train,X_test,y_train,y_test):
    logistic_model = LogisticRegression(solver='saga')#
    logistic_model.fit(X_train, y_train)
    print('eval logistic_model')
    evaluate(logistic_model,[X_test, y_test])

    knn_model = KNeighborsClassifier()  #
    knn_model.fit(X_train, y_train)
    print('eval knn_model')
    evaluate(knn_model, [X_test, y_test])

    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    print('eval svm_model')
    evaluate(svm_model, [X_test, y_test])

    # Train and evaluate Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    print('eval rf_model')
    evaluate(rf_model, [X_test, y_test])

def evaluate(model,test_set):
    predictions = model.predict( test_set[0])
    y_true = test_set[1]

    f1 = f1_score(y_true, predictions, average='weighted')
    f1_macro = f1_score(y_true, predictions, average='macro')
    recall = recall_score(y_true, predictions, average='weighted')
    precision = precision_score(y_true, predictions, average='weighted')
    accuracy = accuracy_score(y_true, predictions)
    balanced_accuracy = balanced_accuracy_score(y_true, predictions)
    print(f"F1 Score: {f1},F1 Score macro: {f1_macro}, Recall: {recall}, Precision: {precision}, Accuracy: {accuracy}, balanced Acc {balanced_accuracy}")

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, predictions)
    return f1,f1_macro,recall,precision,accuracy,balanced_accuracy, conf_matrix

root_folder =Path("/home/lw754/R255/data/singlecell")

# Read CSV files into pandas DataFrames
df_meta = pd.read_csv(root_folder/'meta_data.csv')
df_protein = pd.read_csv(root_folder/'protein_data.csv')
df_rna = pd.read_csv(root_folder/'rna_rand_data_train.csv')

df_rna_merged = pd.merge(df_meta, df_rna, on='cell_id', how='inner')
df_protein_merged = pd.merge(df_meta, df_protein, on='cell_id', how='inner')
df_merged = pd.merge(df_protein_merged, df_protein, on='cell_id', how='inner')

#train test split
test_donor = 31800
df_rna_merged_test = df_rna_merged[df_rna_merged['donor'] == test_donor].copy()
df_rna_merged_train = df_rna_merged[df_rna_merged['donor'] != test_donor].copy()

columns_to_drop = ['cell_id', 'day', 'donor', 'technology']
df_protein_merged.drop(columns=columns_to_drop, inplace=True)
df_rna_merged.drop(columns=columns_to_drop, inplace=True)
df_merged.drop(columns=columns_to_drop, inplace=True)

df_rna_merged_train.drop(columns=columns_to_drop, inplace=True)
df_rna_merged_test.drop(columns=columns_to_drop, inplace=True)
dict_classes = {'BP':0, 'EryP':1, 'MoP':2, 'NeuP':3}
###
y_test = df_rna_merged_test['cell_type'].map(dict_classes).values
y_train = df_rna_merged_train['cell_type'].map(dict_classes).values
X_test = df_rna_merged_test.drop(columns=['cell_type'], inplace=False)
X_train = df_rna_merged_train.drop(columns=['cell_type'], inplace=False)

print('##### RNA #####')
eval_all_classifier(X_train, X_test, y_train, y_test)

### train lr
y = df_merged['cell_type'].map(dict_classes).values
X = df_merged.drop(columns=['cell_type'], inplace=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('##### Both #####')
eval_all_classifier(X_train, X_test, y_train, y_test)

y = df_rna_merged['cell_type'].map(dict_classes).values
X = df_rna_merged.drop(columns=['cell_type'], inplace=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('##### RNA #####')
eval_all_classifier(X_train, X_test, y_train, y_test)

y = df_protein_merged['cell_type'].map(dict_classes).values
X = df_protein_merged.drop(columns=['cell_type'], inplace=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('##### Protein #####')
eval_all_classifier(X_train, X_test, y_train, y_test)





