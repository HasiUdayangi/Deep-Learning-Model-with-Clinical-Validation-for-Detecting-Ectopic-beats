'''
This file will validate a cohort of ECG time slots from a cohort which should be in the format of a csv or json. 


----------------------------------------------------


'''
cohort_path = ''

# Define the name of the new manifest file. 
manifest_key = ''

ECG_Folder = ""
'''
-----------------------------------------------------
'''

import sys
sys.path.append("/home/ec2-user/SageMaker/")
import os
import numpy as np
import pandas as pd
import json
import datetime
from collections import Counter
from utils import S3
from utils.waveform_viewer2 import Waveform_Extract
from utils.waveform_viewer2 import Waveform_Chart
from utils.waveform_viewer2.waveform_helper import Waveform_Helper
import datetime
import re
import pathlib

s3=S3()

def checkLen(dfe):
    """
    Check length of the ECG and alter start and end time if need be. If lead 2 for the ECG is missing then drop patient bedtime.
    """
    lenm = len(np.array(dfe['ecg_ii']))
    for ecg in dfe.columns[3:]:
        if len(np.array(dfe[ecg])) < 7200:
            timeflag = 1
            lenm = len(np.array(dfe[ecg]))
            break
        else:
            timeflag = 0
    return timeflag, lenm

def checkLeadII(dfe):
    """Checks LeadII of ECG and if it is disconnected then it flags the timeslot as unsuitable. If lead II is acceptable then killflag is not triggered.

    Args:
        dfe (pd.DataFrame): DataFrame which contains the leads of the ECG timeslot. 
    """
    if any(count > 2400 for count in Counter(np.array(dfe['ecg_ii'])).values()) and any(x in dfe['ecg_ii'] for x in [32768]):
        killflag = 1
        print("\nPatient Bedtime has too much missing data on ECG lead II. Dropping from label manifest...\n")
    else:
        killflag = 0
    return killflag

def storeECGtoS3(df: pd.DataFrame, s3_path):
    """
    Method stores ECG leads to json file in S3 at given path with transform for displaying ECG at correct amplitude for labelling interface.
    """
    data = {}
    for ecg in df.columns[3:]:
        data[ecg] = np.around((df[ecg]*2.44/1000).astype(np.double),4).tolist()
    s3.put_json(data,s3_path)
    
def prelabelSetup(df: pd.DataFrame, s3_path):
    data = {}
    for ecg in df.columns[3:]:
        data[ecg] = np.around((df[ecg]).astype(np.double),4).tolist()
    s3.put_json(data,s3_path)
    
def run_json(lines, ECG_bucket, prelabel_bucket, prelabel_manifest_path): # Test set
    
    s3=S3()
    timeFormat = '%Y-%m-%d %H:%M:%S'

    manifest = []
    for l in lines:
        d = json.loads(l)
        start = datetime.datetime.strptime(d['start'],timeFormat).strftime(timeFormat)
        end = (datetime.datetime.strptime(d['start'],timeFormat) + datetime.timedelta(seconds=30)).strftime(timeFormat)
        bed = d['bed']
        pid = d['patientid']
        ref = f'{bed}_{pid}_{start}_{end}.json'
        s3_path = os.path.join(ECG_bucket, ref)
        s3_path_prelabel = os.path.join(prelabel_ECG_path, ref)
        
        try:
        
            we = Waveform_Extract(pid)
            we.set_extract_time(start, end)
            dfe = we.get_ecg(cols=['ecg_ii']) 

            timeflag, lenm = checkLen(dfe)

            '''
            The timeflag trigger will change the start and end time by the amount that is missing from the end. 
            '''
            if timeflag == 1:
                print("timeflag triggered\n")
                sec = (7200 -  lenm)/fs
                secend = lenm/fs
                start = (datetime.datetime.strptime(d['start'],timeFormat) - datetime.timedelta(seconds=sec)).strftime(timeFormat)
                end = (datetime.datetime.strptime(d['start'],timeFormat) + datetime.timedelta(seconds=secend)).strftime(timeFormat)
                ref = f'{bed}_{pid}_{start}_{end}.json'
                s3_path = os.path.join(ECG_bucket, ref)
                s3_path_prelabel = os.path.join(prelabel_ECG_path, ref)
                we = Waveform_Extract(pid) 
                we.set_extract_time(start, end)
                dfe = we.get_ecg(cols=['ecg_ii'])

            killflag =  checkLeadII(dfe)

            if killflag == 0:
                storeECGtoS3(dfe, s3_path)
                prelabelSetup(dfe, s3_path_prelabel)

                dg = {}
                dg['source-ref'] = s3_path
                dg['bed'] = bed
                dg['patientid'] = pid
                dg['start'] = start
                dg['end'] = end
                manifest.append(dg)
                
                dp = {}
                dp['source-ref'] = s3_path_prelabel
                dp['bed'] = bed
                dp['patientid'] = pid
                dp['start'] = start
                dp['end'] = end
                
                with open(prelabel_manifest_path, 'a') as fl:
                    fl.write(json.dumps(dp))
                    fl.write('\n')
                
        except:
            pass

    return(manifest)

def run_csv(d, ECG_bucket, prelabel_bucket, prelabel_manifest_path): 
    
    s3=S3()
    timeFormat = '%Y-%m-%d %H:%M:%S'

    manifest = []
    manifest_prelabel = []
    for i in range(len(d)):
        
        start = datetime.datetime.strptime(d['start'].iloc[i],timeFormat).strftime(timeFormat)
        end = (datetime.datetime.strptime(d['start'].iloc[i],timeFormat) + datetime.timedelta(seconds=30)).strftime(timeFormat)
        bed = d['bed'].iloc[i]
        pid = d['patientid'].iloc[i]
        ref = f'{bed}_{pid}_{start}_{end}.json'
        s3_path = os.path.join(ECG_bucket, ref)
        s3_path_prelabel = os.path.join(prelabel_ECG_path, ref)
        
        try:
            we = Waveform_Extract(pid)
            we.set_extract_time(start, end)
            dfe = we.get_ecg(cols=['ecg_ii']) 
            
            timeflag, lenm = checkLen(dfe)

            '''
            The timeflag trigger will change the start and end time by the amount that is missing from the end. 
            '''
            if timeflag == 1:
                print("timeflag triggered\n")
                sec = (7200 -  lenm)/fs
                secend = lenm/fs
                start = (datetime.datetime.strptime(d['start'].iloc[i],timeFormat) - datetime.timedelta(seconds=sec)).strftime(timeFormat)
                end = (datetime.datetime.strptime(d['start'].iloc[i],timeFormat) + datetime.timedelta(seconds=secend)).strftime(timeFormat)
                ref = f'{bed}_{pid}_{start}_{end}.json'
                s3_path = os.path.join(ECG_bucket, ref)
                s3_path_prelabel = os.path.join(prelabel_ECG_path, ref)
                we = Waveform_Extract(pid) 
                we.set_extract_time(start, end)
                dfe = we.get_ecg(cols=['ecg_ii'])

            killflag =  checkLeadII(dfe)

            if killflag == 0:
                storeECGtoS3(dfe, s3_path)
                prelabelSetup(dfe, s3_path_prelabel)

                dg = {}
                dg['source-ref'] = s3_path
                dg['bed'] = bed
                dg['patientid'] = pid
                dg['start'] = start
                dg['end'] = end
                manifest.append(dg)
                
                dp = {}
                dp['source-ref'] = s3_path_prelabel
                dp['bed'] = bed
                dp['patientid'] = pid
                dp['start'] = start
                dp['end'] = end
                
                
                with open(prelabel_manifest_path, 'a') as fl:
                    fl.write(json.dumps(dp))
                    fl.write('\n')
        except:
            pass
    return(manifest)

manifest_bucket = ''
manifest_path = os.path.join(manifest_bucket, manifest_key)

# Define the name of folder to store ECG values to be referenced. 
ECG_Folder = "" 

ECG_path = os.path.join("", ECG_Folder)

pwd = os.path.abspath(os.getcwd())

prelabel_ECG_path = ''
prelabel_manifest_path = os.path.join(pwd, 'prelabel_inference-' + manifest_key)

if cohort_path.endswith('.csv'):
    df = pd.read_csv(cohort_path, sep=',')
    file = open(prelabel_manifest_path, 'w')
    file.close()
    
    manifest = run_csv(df, ECG_path, pwd, prelabel_manifest_path)
    s3.put_jsonl(manifest, manifest_path)
    print("Task completed successfully")
    

elif cohort_path.endswith('.json' or '.manifest' or '.jsonl'):
    file = open(cohort_path, 'r')
    lines = file.readlines()
    file.close()
    
    manifest = run_json(lines, ECG_bucket, pwd, prelabel_manifest_path)
    s3.put_jsonl(manifest, manifest_path)
    print("Task completed successfully")
