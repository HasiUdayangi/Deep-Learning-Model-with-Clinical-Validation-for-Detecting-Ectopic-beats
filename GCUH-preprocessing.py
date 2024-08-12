#========================== IMPORTS =========================#
import datetime as dt
import pandas as pd
import numpy as np
import sys
import tensorflow
import collections
import matplotlib.pyplot as plt
import os
from IPython.display import clear_output
#import resampy

from utils import S3, LabelStore
from datetime import datetime, timedelta, date
from utils import LabelStore, LabelReader, Athena_Query, S3
from utils.waveform_viewer2 import Waveform_Extract, Waveform_Helper
s3 = S3()
from scipy import signal as SIGNAL

#========================== Functions =========================#
def set_window_size(L=1):
    """
    This function returns window lengh
    param L:
    """
    L = L
    return L

def resample_ecg(ECG, ogFS):
    fs = 240
    N = int((len(ECG) / ogFS) * fs)
    SIGNAL.resample(ECG, N)
    return ECG

def resample(ECG):
    newFS = 128 # target sampling rate in Hz
    ogFS = 240
    resampled_ecg = resampy.resample(ECG, ogFS, newFS)
    return resampled_ecg

def auto_gen_bed_times(patientid, starttime_offset=0, len_of_extract=1000, bed_occ=0):
    we = Waveform_Extract(patientid)
    pat_bed_info = we.get_all_bed_times()    
    
    start = str(pat_bed_info['fromtime'][bed_occ])
    end = str(pat_bed_info['totime'][bed_occ])
    BED = pat_bed_info['bedname'][bed_occ]
    bed = BED.strip().upper()
    if bed[3] == '0':
        bed = bed[0:3] + " " + bed[4:5]

    datestring_format = '%Y-%m-%d %H:%M:%S'
    st = datetime.strptime(str(start), datestring_format)
    a_timedelta = st - datetime(1970, 1, 1)
    seconds = a_timedelta.total_seconds()+starttime_offset
    ECG_start = str(datetime.fromtimestamp(seconds))
    ECG_end = str(datetime.fromtimestamp((seconds+len_of_extract)))
    
    return (bed, ECG_start, ECG_end)

def get_ecg(patientid, ECG_cols=['ecg_ii'], starttime_offset=0, len_of_extract=1000, bed_occ=0):
    bed, start, end = auto_gen_bed_times(patientid, starttime_offset, len_of_extract, bed_occ)
    we = Waveform_Extract(patientid)
    we.set_extract_time(start, end)
    df = we.get_ecg(cols=ECG_cols)
    
    ECG = df['ecg_ii'].to_numpy()
    
    print('ECG Retrieved!')

    return ECG

def getECG(patientid, start, end, ECG_cols=['ecg_ii']):
    we = Waveform_Extract(patientid)
    we.set_extract_time(start,end)
    df = we.get_ecg(cols= ECG_cols)
    
    ECG = df['ecg_ii'].to_numpy()
    
    print("ECG Retrieved")
    
    return ECG

def get_R_peak_windowed(ECG, fs, L=1):
    
    qrs_inds = MHTD(ECG, fs)
    qrs_inds = qrs_inds.astype(int)

    W = fs * L
    # Crop data that won't give a full segment
    qrs_inds = qrs_inds[qrs_inds > int(W / 2)]
    qrs_inds = qrs_inds[qrs_inds < (len(ECG) - int(W / 2))]
    N = len(qrs_inds)
    sig = np.zeros((N, W))
    
    sig_data = []
    
    for i in range(N):
        st = qrs_inds[i] - int(W / 2)
        en = qrs_inds[i] + int(W / 2)
        sig[i, :] = ECG[st:en]
        
    
    sig_data.append(sig)
    sig_df = pd.DataFrame(np.concatenate(sig_data))
    
    sig_norm_data = []
    for i in range(N):
        st = qrs_inds[i] - int(W / 2)
        en = qrs_inds[i] + int(W / 2)
        sig[i, :] = ECG[st:en]
        sig_norm = zero_mean_normalise(sig)

    sig_norm_data.append(sig_norm)
    sig_norm_df = pd.DataFrame(np.concatenate(sig_norm_data))
    return sig_df, sig_norm_df, qrs_inds, N

def zero_mean_normalise(signl):
    """
    returns normalized signal
    """
    for i in range(len(signl)):
        signl[i, :] -= np.mean(signl[i, :])  # Zero mean the signal
        signl[i, :] /= np.max(signl[i, :])  # Normalise

    return signl
