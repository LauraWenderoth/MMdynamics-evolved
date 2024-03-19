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
from functools import reduce
import tensorflow as tf
from tensorflow.keras import layers, models

def eval_all_classifier(X_train,X_test,y_train,y_test):
    results = {}

    X_train_nn, X_val, y_train_nn, y_val = train_test_split(X_train, y_train, test_size = 0.20, random_state = 42)
    neural_model = models.Sequential([
        layers.Flatten(input_shape=X_train.shape[1:]),  # Flatten input
        layers.Dense(128, activation='relu'),  # Fully connected layer with 128 neurons and ReLU activation
        layers.Dense(4, activation='softmax')  # Output layer with 4 neurons for classification
    ])
    neural_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    neural_model.fit(X_train_nn, y_train_nn, epochs=10, validation_data=(X_val, y_val))
    print('eval neural')
    result,cfm = evaluate(neural_model, [X_test, y_test])
    results['NN'] = result

    logistic_model = LogisticRegression(solver='saga')#
    logistic_model.fit(X_train, y_train)
    print('eval logistic_model')
    result,cfm = evaluate(logistic_model,[X_test, y_test])
    results['LR'] = result

    knn_model = KNeighborsClassifier()  #
    knn_model.fit(X_train, y_train)
    print('eval knn_model')
    result,cfm = evaluate(knn_model, [X_test, y_test])
    results['KNN'] = result

    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    print('eval svm_model')
    result,cfm = evaluate(svm_model, [X_test, y_test])
    results['SVM'] = result

    # Train and evaluate Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    print('eval rf_model')
    result,cfm = evaluate(rf_model, [X_test, y_test])
    results['RF'] = result
    return results

def evaluate(model,test_set):
    predictions = model.predict( test_set[0])
    if len(predictions.shape) == 2:
        probas = tf.nn.softmax(predictions).numpy()
        predictions = np.argmax(probas, axis=-1)

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
    results = {'f1':f1,'f1_macro':f1_macro,'recall':recall,'precision':precision,'accuracy':accuracy,'balanced_accuracy':balanced_accuracy}
    return results, conf_matrix

def prepare_data(meta_df,dfs,donor = 31800,concatinate=False,columns_to_drop = ['cell_id', 'day', 'donor', 'technology'],target='cell_type',dict_classes = {'BP': 0, 'EryP': 1, 'MoP': 2, 'NeuP': 3}):
    merged_dfs = []
    for df in dfs:
        merged_df = pd.merge(meta_df, df, on='cell_id', how='inner')
        merged_df = merged_df.drop_duplicates(subset='cell_id', keep='first')
        merged_dfs.append(merged_df)
    if concatinate:
        concatenated_df = pd.merge(merged_dfs[0], merged_dfs[1], on=columns_to_drop+[target], how='inner')
        for df in merged_dfs[2:]:
            concatenated_df = pd.merge(concatenated_df, df, on='cell_id', how='inner')
        merged_dfs.append(concatenated_df)
    dfs = []
    for df in merged_dfs:
        test = df[df['donor'] == donor].copy()
        train = df[df['donor'] != donor].copy()
        test.drop(columns=columns_to_drop, inplace=True)
        train.drop(columns=columns_to_drop, inplace=True)
        y_test = test[target].map(dict_classes).values
        y_train = train[target].map(dict_classes).values
        X_test = test.drop(columns=[target], inplace=False)
        X_train = train.drop(columns=[target], inplace=False)
        dfs.append({'X_train': X_train, 'X_test': X_test,'y_train': y_train, 'y_test': y_test})

    return dfs

def transform_to_df(models_dict):
    data = []
    # Iterate over each model and its corresponding results
    for model, result_data in models_dict.items():
        # Append the model name as the first element of the data row
        row_data = [model]
        # Append the evaluation metrics for the model
        for metric, value in result_data.items():
            row_data.append(value)
        # Append the row data to the main data list
        data.append(row_data)

    # Create a DataFrame from the data
    df_results = pd.DataFrame(data, columns=['Model', 'f1', 'f1_macro', 'recall', 'precision', 'accuracy',
                                             'balanced_accuracy'])
    df_results.set_index('Model', inplace=True)


    return df_results

if __name__ == "__main__":
    root_folder = Path("/home/lw754/R255/data/singlecell")
    results_dir = Path("/home/lw754/R255/results/singlecell")

    df_meta = pd.read_csv(root_folder / 'meta_data_train.csv')
    df_protein = pd.read_csv(root_folder / 'protein_data_train.csv')
    df_rna = pd.read_csv(root_folder / 'rna_rand_data_train.csv')

    modality_names = ['protein','rna','both']
    modalities = {key: [] for key in modality_names}
    modality_results = {key: [] for key in modality_names}
    donors = np.unique(df_meta['donor'].values)

    for donor in donors:
        dfs = prepare_data(df_meta, [df_protein,df_rna], donor = donor, concatinate=True)
        for idx,key in enumerate(modalities.keys()):
            modalities[key].append(dfs[idx])


    for key in modalities.keys():
        print(f'### {key} ###')
        results = []
        for df_dict in modalities[key]:
            X_train, X_test, y_train, y_test = df_dict['X_train'], df_dict['X_test'], df_dict['y_train'], df_dict['y_test']
            print('#################')
            result = eval_all_classifier(X_train, X_test, y_train, y_test)
            results.append(transform_to_df(result))
        mean_df = pd.concat(results).groupby(level=0).mean()

        # Calculate the standard deviation across the three DataFrames
        std_df = pd.concat(results).groupby(level=0).std()
        print(mean_df.to_latex())
        print(std_df.to_latex())
        mean_df.to_csv(results_dir/f'{key}_mean.csv')
        std_df.to_csv(results_dir / f'{key}_std.csv')


    print('Finished')
