# Required libraries
import os
import re
import numpy as np
import pandas as pd
import json
import joblib
from datetime import datetime, timedelta, date
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import sklearn.metrics as skm
from sklearn import preprocessing
from sklearn import metrics
from matplotlib import pyplot as plt
from scipy import signal
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.models import load_model
import boto3

# Adding path to system for importing custom modules
sys.path.append("")
sys.path.append("")

from ectopicpreprocessing import set_window_size, resample_ecg, auto_gen_bed_times, get_ecg, getECG, get_R_peak_windowed, zero_mean_normalise


# Functions for processing ECG
def resample_ecg(ECG, ogFS):
    fs = 240
    N = int((len(ECG) / ogFS) * fs)
    return SIGNAL.resample(ECG, N)

def predict(ecg):
    label_mappings = {'0': 'NOTEB', '1': 'VEB', '2': 'SVEB'}
    resampled_ecg = resample_ecg(ecg, ogFS=128)
    sig_df, sig_norm_df, rpeaks, N = get_R_peak_windowed(resampled_ecg, fs=128)
    X_test = np.expand_dims(sig_norm_df.iloc[:, :127], 2)
    pred_label_prob = model.predict(X_test)
    predicted_labels = np.argmax(pred_label_prob, axis=1)
    max_probabilities = np.max(pred_label_prob, axis=1)
    predicted_label = [label_mappings.get(str(label), str(label)) for label in predicted_labels]
    return max_probabilities, predicted_label

def get_prelabels(df):
    ecg = df.ecg_ii.apply(lambda x: 2000 if x > 2000 else -2000 if x < -2000 else x)
    max_prob, predlabel = predict(ecg)
    qrs_inds = MHTD(ecg, fs=240)
    rpeaks = qrs_inds.astype(int)
    return max_prob, predlabel, rpeaks

# AWS S3 client setup
s3_client = boto3.client('s3')
ECG_path = 's3://ectopicbeat-model/Cohort-Ectopic-gp1/'
manifest = []

def save_jsonl(list_of_jsons, filename):
    object_strings = [json.dumps(obj) for obj in list_of_jsons]
    file_string = "\n".join(object_strings)
    with open(filename, 'w') as file:
        file.writelines(file_string)

# Read from file and process data
if os.path.exists('/home/ec2-user/SageMaker/Ectopic beat model/Ground Truth Labelling/'):
    with open('CohortEctopic_gp1.csv', 'r') as file:
        lines = file.readlines()

    for l in lines:
        d = json.loads(l)
        df = pd.DataFrame.from_dict(s3.get_json(d['source-ref']))
        prob_max, predlabels, rpeaks = get_prelabels(df)
        probs = [round(i,2) for i in prob_max]
        dg = {
            'source-ref': os.path.join(ECG_path, f"{d['bed']}_{d['patientid']}_{d['start']}_{d['end']}.json"),
            'bed': d['bed'],
            'patientid': d['patientid'],
            'start': d['start'],
            'end': d['end'],
            'rpeaks': rpeaks.tolist(),
            'pre_labels': predlabels,
            'pre_probs': str(probs)
        }
        manifest.append(dg)

    save_jsonl(manifest, "manifest-ectopic-gp1.jsonl")
    print(manifest)
else:
    print("Ground Truth Labelling directory not found.")
