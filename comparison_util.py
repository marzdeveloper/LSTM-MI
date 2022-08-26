__author__ = 'Daniele Marzetti'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def get_avg_confusion_matrix():
    #Inserire nella cartella in cui c'è questo script le matrici di confusione di un detemrinato algoritmo
    tp = []
    tn = []
    fp = []
    fn = []

    for i in range(1,11):
        df = pd.read_csv("confusion_matrix"+str(i)+".csv", header=None, index_col= None)
        tn.append(df.loc[0, 0])
        tp.append(df.loc[1, 1])
        fn.append(df.loc[1, 0])
        fp.append(df.loc[0, 1])

    avg_tp = np.array(tp).mean()
    avg_tn = np.array(tn).mean()
    avg_fp = np.array(fp).mean()
    avg_fn = np.array(fn).mean()

    print("avg tp " + str(avg_tp))
    print("avg tn " + str(avg_tn))
    print("avg fp " + str(avg_fp))
    print("avg fn " + str(avg_fn))


def plot():
    # Inserire nella cartella solo i csv avg_metrics.csv dei vari algoritmi
    # Tale funzione permette di ottenere un grafico per confrontare le metriche medie e la std dei vari algoritmi
    for i in ('precision', 'recall', 'f1-score', 'accuracy'):
        df1 = pd.DataFrame(columns=['mean', 'std'])
        for j, file in enumerate(os.listdir()):
            if file.endswith(".csv"):
                df = pd.read_csv(file, index_col=0)
                df1.loc[file.split('_')[0]] = df[i]

        plt.figure(figsize=(15, 8))
        sns.barplot("mean", "index", data=df1.reset_index(), palette="Set2", orient="h", **{'xerr': df1['std']})
        plt.xlabel("Mean " + i)
        plt.ylabel("Algorithm")
        plt.title("Cross validation " + i)
        #plt.show()
        plt.savefig(i + '.png')


def metrics_avg():
    # Da usare nei csv di elmo e bilbo per ottenere la media e la std di ogni metrica rispetto ai dga
    metrics = pd.read_csv("bilbo.csv", delimiter=',', index_col=0)
    avg_metrics = pd.DataFrame([metrics.loc['precision']['dga'].mean(), metrics.loc['precision']['dga'].std()], columns=['precision'], index=['mean', 'std'])
    avg_metrics = pd.concat([avg_metrics, pd.DataFrame([metrics.loc['recall']['dga'].mean(), metrics.loc['recall']['dga'].std()], columns=['recall'], index=avg_metrics.index)], axis =1)
    avg_metrics = pd.concat([avg_metrics,
                             pd.DataFrame([metrics.loc['f1-score']['dga'].mean(), metrics.loc['f1-score']['dga'].std()],
                                          columns=['f1-score'], index=avg_metrics.index)], axis=1)
    avg_metrics = pd.concat([avg_metrics,
                             pd.DataFrame([metrics.loc['precision']['accuracy'].mean(), metrics.loc['precision']['accuracy'].std()],
                                          columns=['accuracy'], index=avg_metrics.index)], axis=1)
    avg_metrics.to_csv("bilbo_avg_metrics.csv")


def metrics_avg1():
    # Da usare nei csv di lstm-mi e ngrams per ottenere la media e la std di ogni metrica rispetto ai dga
    metrics = pd.read_csv("ngrams.csv", delimiter=';')
    avg_metrics = pd.DataFrame([metrics['Precision'].mean(), metrics['Precision'].std()], columns=['precision'], index=['mean', 'std'])
    avg_metrics = pd.concat([avg_metrics, pd.DataFrame([metrics['Recall'].mean(), metrics['Recall'].std()], columns=['recall'], index=avg_metrics.index)], axis =1)
    avg_metrics = pd.concat([avg_metrics,
                             pd.DataFrame([metrics['F1-score'].mean(), metrics['F1-score'].std()],
                                          columns=['f1-score'], index=avg_metrics.index)], axis=1)
    avg_metrics = pd.concat([avg_metrics,
                             pd.DataFrame([metrics['Accuracy'].mean(), metrics['Accuracy'].std()],
                                          columns=['accuracy'], index=avg_metrics.index)], axis=1)
    avg_metrics.to_csv("ngrams_avg_metrics.csv")