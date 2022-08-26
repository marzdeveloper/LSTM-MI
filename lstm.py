"""Train and test LSTM classifier"""
import numpy as np
import os
import random
import csv
import collections
import math
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score, classification_report,accuracy_score, f1_score, confusion_matrix
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable

# todo set params
folder_dataset_path = "drive/MyDrive" #drive/MyDrive
dataset_name = "dga_domains"
start_fold = 1
end_fold = 10 #10
out_path = "drive/MyDrive" #drive/MyDrive
test_name = "dga_domains"
max_epoch = 20 #20
batch_size = 128 #128
patience = 20
min_delta = 0.0


def get_data(fold):
	"""Read data from file (train, test) to process"""
	X_train = []
	X_test = []
	y_train = []
	y_test = []
	in_train_file = open(folder_dataset_path + "/" + dataset_name + "_train_fold_" + str(fold) + ".csv")
	in_test_file = open(folder_dataset_path + "/" + dataset_name + "_test_fold_" + str(fold) + ".csv")
	csvreader = csv.reader(in_train_file, delimiter=";")
	for row in csvreader:
		y_train.append(int(row[0]))
		line = []
		for i in range(1, len(row)):
			line.append(int(row[i]))
		X_train.append(line)
	csvreader = csv.reader(in_test_file, delimiter=";")
	for row in csvreader:
		y_test.append(int(row[0]))
		line = []
		for i in range(1, len(row)):
			line.append(int(row[i]))
		X_test.append(line)
	X_train = np.array(X_train)
	X_test = np.array(X_test)
	y_train = np.array(y_train)
	y_test = np.array(y_test)
	return X_train, X_test, y_train, y_test

def build_binary_model(max_features, maxlen):
    """Build LSTM model for two-class classification"""
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer='rmsprop', metrics=['accuracy'])

    return model

def create_class_weight(labels_dict,mu):
    """Create weight based on the number of domain name in the dataset"""
    total = np.sum(labels_dict.values())
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.pow(total/float(labels_dict[key]),mu)
	class_weight[key] = score	

    return class_weight

def classifaction_report_csv(report,precision,recall,f1_score,accuracy,cm,fold):
    """Generate the report to data processing"""
    with open(out_path + '/LSTM-MI_' + test_name + '_results.csv', 'a') as f:
        report_data = []
        lines = report.split('\n')
        row = {}
        row['class'] =  "fold %u" % (fold)
        report_data.append(row)
        for line in lines[2:44]:
            row = {}
            line = " ".join(line.split())
            row_data = line.split(' ')
            if(len(row_data)>2):
                if(row_data[0]!='avg'):
                    row['class'] = row_data[0]
                    row['precision'] = float(row_data[1])
                    row['recall'] = float(row_data[2])
                    row['f1_score'] = float(row_data[3])
                    row['support'] = row_data[4]
                    report_data.append(row)
                else:
                    row['class'] = row_data[0]+row_data[1]+row_data[2]
                    row['precision'] = float(row_data[3])
                    row['recall'] = float(row_data[4])
                    row['f1_score'] = float(row_data[5])
                    row['support'] = row_data[6]
                    report_data.append(row)
        row = {}
        row['class'] = 'macro'
        row['precision'] = float(precision)
        row['recall'] = float(recall)
        row['f1_score'] = float(f1_score)
        row['support'] = 0
        report_data.append(row)
        row = {}
        row['class'] = 'accuracy'
        row['f1_score'] = float(accuracy)
        report_data.append(row)
        row = {}
        row['class'] = 'confusion matrix'
        row['precision'] = 'dga'
        row['f1_score'] = 'legit'
        report_data.append(row)
        row = {}
        row['class'] = 'legit'
        row['precision'] = int(cm[0][1])
        row['f1_score'] = int(cm[0][0])
        report_data.append(row)
        row = {}
        row['class'] = 'dga'
        row['precision'] = int(cm[1][1])
        row['f1_score'] = int(cm[1][0])
        report_data.append(row)
        dataframe = pd.DataFrame.from_dict(report_data)
        dataframe.to_csv(f, index = False)

def run(max_epoch=max_epoch, start_fold=start_fold, end_fold=end_fold, batch_size=batch_size, patience=patience, min_delta=min_delta):
    """Run train/test on logistic regression model"""
    for fold in range (start_fold, end_fold + 1):
        X_train, X_test, y_train, y_test = get_data(fold)
        
        print "Fold %u/%u" % (fold, end_fold) 

        #Build the model for two-class classification stage
        model = build_binary_model(40, 73) #max features, max len
        
        print "Training the model for two-class classification stage..."
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=0)
        for train, test in sss1.split(X_train, y_train):
            X_train, X_holdout, y_train, y_holdout = X_train[train], X_train[test], y_train[train], y_train[test]
        
        #Create weight for two-class classification stage
        labels_dict=collections.Counter(y_train)
        class_weight = create_class_weight(labels_dict,0.1)
        best_auc = 0.0
        acc_val = []
        acc_train = []
        loss_val = []
        loss_train = []
        stop_train = 0
        
        for ep in range(max_epoch):
            if stop_train < patience:
                print "Epoch %u/%u" % (ep+1, max_epoch)
                history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1, validation_data=(X_holdout, y_holdout), class_weight=class_weight)
                t_loss = history.history['loss'][0]
                t_acc = history.history['accuracy'][0]
                v_loss = history.history['val_loss'][0]
                v_acc = history.history['val_accuracy'][0]
                #Get the model with highest accuracy
                if v_acc >= best_auc + min_delta:
                    stop_train = 0
                else:
                    stop_train += 1    
                if v_acc > best_auc:
                    best_model = model
                    best_auc = v_acc
                #Getting all data
                acc_val.append(v_acc)
                acc_train.append(t_acc)
                loss_val.append(v_loss)
                loss_train.append(t_loss)
		#Save the model for two-class classification     
        #Serialize model to JSON
        model_json = best_model.to_json()
        name_file = out_path + "/LSTM-MI_" + test_name + "_model_fold_" + str(fold) + ".json"
        name_file2 = out_path + "/LSTM-MI_" + test_name + "_model_fold_" + str(fold) + ".h5"
        with open(name_file, "w") as json_file:
            json_file.write(model_json)
        #Serialize weights to HDF5
            best_model.save_weights(name_file2)
        print("Saved two-class model to disk\n")
        
        x_epochs = []
        for i in range(0,len(acc_train)):
        	x_epochs.append(i+1)	

        fig1 = plt.figure(1)
        plt.title('Loss')
        plt.plot(loss_val, 'r', label='Validation Loss')
        plt.plot(loss_train, 'b', label='Training Loss')
        plt.legend(loc="upper right")
        x = list(range(0, len(loss_train), 1))
        plt.xlim(right=len(loss_train))
        plt.xticks(x, x_epochs)
        plt.grid(True)
        fig1.savefig(out_path + "/LSTM-MI_" + test_name + "_loss_fold_" + str(fold) + ".png")
        plt.show()
        plt.close(fig1)

        fig2 = plt.figure(2)
        plt.title('Accuracy')
        plt.plot(acc_val, 'r', label='Validation Accuracy')
        plt.plot(acc_train, 'b', label='Training Accuracy')
        plt.legend(loc="lower right")
        x = list(range(0, len(acc_train), 1))
        plt.xlim(right=len(acc_train))
        plt.xticks(x, x_epochs)
        plt.grid(True)
        fig2.savefig(out_path + "/LSTM-MI_" + test_name + "_accuracy_fold_" + str(fold) + ".png")
        plt.show()
        plt.close(fig2)

        y_pred = best_model.predict_proba(X_test)
        y_result = [0 if(x<=0.5) else 1 for x in y_pred]
        #End of two-class classification stage

        #Calculate the final result
        class_names = ["legit", "dga"]

        score = f1_score(y_test, y_result, average="macro")
        precision = precision_score(y_test, y_result, average="macro")
        recall = recall_score(y_test, y_result, average="macro")
        report = classification_report(y_test, y_result, target_names=class_names, digits=4)
        acc = accuracy_score(y_test, y_result)
        print '\nClassification report:\n', report
        print 'F1 score:', score
        print 'Recall:', recall
        print 'Precision:', precision
        print 'Acc:', acc
        cm = confusion_matrix(y_test, y_result)
        print("\nConfusion matrix:")
        print cm
        print("")
        classifaction_report_csv(report,precision,recall,score,acc, cm, fold)
        
   
if __name__ == "__main__":
    run()
