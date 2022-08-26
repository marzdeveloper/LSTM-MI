import csv
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing import sequence

# todo set params
csv_dataset_path = "drive/MyDrive/Cyber Security/dga_domains_full.csv" #drive/MyDrive/Cyber Security/dga_domains_full.csv
out_path = "drive/MyDrive" #drive/MyDrive
nfolds=10

def get_data():
    """Read data from file (Traning, testing and validation) to process"""
    data= []
    with open(csv_dataset_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    binary_labels = [x[0] for x in data]
    X = [x[2] for x in data]
    labels = [x[1] for x in data]
    return X, binary_labels, labels

def to_csv(X_train, X_test, y_train, y_test, fold):
    out_file_train = open(out_path + '/dga_domains_train_fold_' + str(fold) + '.csv', "w")
    out_file_test = open(out_path + '/dga_domains_test_fold_' + str(fold) + '.csv', "w")
    csvwriter_train = csv.writer(out_file_train, delimiter=";")
    csvwriter_test = csv.writer(out_file_test, delimiter=";")
    for i in range (0, len(y_train)):
        row = []
        row.append(y_train[i])
        for j in range (0, len(X_train[i])):
            row.append(X_train[i][j])
        csvwriter_train.writerow(row)
    for i in range (0, len(y_test)):
        row = []
        row.append(y_test[i])
        for j in range (0, len(X_test[i])):
            row.append(X_test[i][j])
        csvwriter_test.writerow(row)

def preprocess(X, binary_labels, labels):
    valid_chars = {x: idx + 1 for idx, x in enumerate(set(''.join(X)))}
    max_features = len(valid_chars) + 1
    maxlen = np.max([len(x) for x in X])

    # Convert characters to int and pad
    X = [[valid_chars[y] for y in x] for x in X]
    X = sequence.pad_sequences(X, maxlen=maxlen)

    # Convert labels to 0-1 for binary class
    y_binary = np.array([0 if x == 'legit' else 1 for x in binary_labels])
    # Convert labels to 0-(number of classes - 1) for multi class
    valid_class = {i: indx for indx, i in enumerate(set(labels))}
    y = [valid_class[x] for x in labels]
    y = np.array(y)
    return X, y_binary, y, max_features, maxlen

def main(nfolds=nfolds):
    # Read data to process
    X, binary_labels, labels = get_data()
    print("Reading data...")

    # Preprocessing stage
    X, y_binary, y, max_features, maxlen = preprocess(X, binary_labels, labels)
    print("Preprocessing...")
    print("Max features: " + str(max_features))
    print("Max len: " + str(maxlen))

    # Divide the dataset into training + holdout and testing with folds
    sss = StratifiedKFold(n_splits=nfolds, random_state=0)

    fold = 0
    for train, test in sss.split(X, y_binary, y):
        print("Writing fold " + str(fold + 1) + " to csv...")
        fold += 1
        X_train, X_test, y_train, y_test = X[train], X[test], y_binary[train], y_binary[test]
        to_csv(X_train, X_test, y_train, y_test, fold)
    print("Files created")

if __name__ == "__main__":
    main()
