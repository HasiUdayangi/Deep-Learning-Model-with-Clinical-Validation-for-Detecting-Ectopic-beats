
"""
Clinician Validation and Annotation Processing Script

This script performs the following tasks:
1. Downloads and loads the output manifest from S3.
2. Retrieves worker annotation files from S3 and processes them.
3. Creates a CSV file from a prelabel inference manifest.
4. Checks for client errors in the manifest.
5. Extracts nested annotation labels and processes beat types.
6. Loads cohort data and retrieves ECG files from S3 based on the CSV.
7. Uses a (pre-trained) model to predict labels and evaluates the predictions.

"""

import json
import csv
import os
import glob
from collections import Counter
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta, MO

import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Custom utilities
from utils import Athena_Query, s3, LabelStore
from utils.sql_query import SqlQuery
from utils.waveform_viewer2 import Waveform_Extract, Waveform_Helper

# Global constants and S3 configuration
BUCKET = ''
EXP_NAME = ''
JOB_NAME = ''
OUTPUT_MANIFEST_KEY = f"{EXP_NAME}/{JOB_NAME}/"
WORKER_ANNOTATION_PREFIX = f"{EXP_NAME}/{JOB_NAME}/"
ECG_FOLDER = "

# Initialize S3 client and helper objects
s3_client = boto3.client('s3')
wh = Waveform_Helper()
athena = Athena_Query()


def download_file_from_s3(bucket, key, local_path):
    """Download a file from S3 to a local path."""
    try:
        s3_client.download_file(bucket, key, local_path)
        print(f"Downloaded {key} to {local_path}")
    except Exception as e:
        print(f"Error downloading {key}: {e}")


def load_manifest(manifest_path):
    """Load and parse the manifest file."""
    with open(manifest_path, "r") as f:
        manifest = [json.loads(line.strip()) for line in f.readlines()]
    return manifest


def download_worker_annotations(local_dir=""):
    """Download worker annotation files from S3 recursively to a local directory."""
    # Here you might use boto3 to list objects and download them.
    # For brevity, we assume the files have been downloaded externally.
    print("Assuming worker annotations are downloaded to:", local_dir)
    return glob.glob(os.path.join(local_dir, "**/*.json"), recursive=True)


def read_sample_worker_annotation(file_path):
    """Read and return JSON data from a sample worker annotation file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def create_patient_data_csv(input_json_path, output_csv_path):
    """Create a CSV file from the prelabel inference manifest JSON."""
    with open(input_json_path, 'r') as jf:
        json_data = [json.loads(line) for line in jf]
    
    with open(output_csv_path, 'w', newline='') as cf:
        csv_writer = csv.writer(cf)
        csv_writer.writerow(['bed', 'patientid', 'start', 'end'])
        for entry in json_data:
            csv_writer.writerow([entry.get('bed'), entry.get('patientid'),
                                 entry.get('start'), entry.get('end')])
    print("CSV file created successfully at", output_csv_path)


def check_client_errors(manifest):
    """Iterate through the manifest and print entries with ClientError."""
    for entry in manifest:
        metadata = entry.get("", {})
        failure_reason = metadata.get("failure-reason", "")
        if "ClientError" in failure_reason:
            print("ClientError found in entry:", entry.get("source-ref"))


# Dummy process_beats function – replace with your actual implementation.
def process_beats(beat_list, label):
    """Process a list of beat annotations and return processed labels.
    For demonstration, returns a list of tuples (start, end, label).
    """
    processed = []
    for beat in beat_list:
        # Assuming beat is a dict with keys 'start' and 'end'
        start = beat.get('start', 0)
        end = beat.get('end', 0)
        processed.append((start, end, label))
    return processed


def extract_annotation_labels(manifest_entry):
    """
    Extract nested annotation labels from a manifest entry.
    Assumes the annotation data is under:

    """
    try:
        annotation_data = manifest_entry[""][""][0][""]
        nested_json = json.loads(annotation_data["content"])
        labels_data = nested_json.get('labels', {})
        
        not_eb = labels_data.get('not_eb', [])
        supraventricular_eb = labels_data.get('supraventricular_eb', [])
        ventricular_eb = labels_data.get('ventricular_eb', [])
        
        not_eb_labels = process_beats(not_eb, 'NOTEB')
        supraventricular_labels = process_beats(supraventricular_eb, 'SVEB')
        ventricular_labels = process_beats(ventricular_eb, 'VEB')
        
        all_labels = not_eb_labels + supraventricular_labels + ventricular_labels
        all_labels.sort(key=lambda x: x[0])
        return all_labels
    except Exception as e:
        print("Error extracting annotation labels:", e)
        return []


def load_cohort_data(csv_path="filtered_data.csv"):
    """Load cohort data from a CSV file."""
    return pd.read_csv(csv_path)


def load_ecg_data_from_s3(cohort_csv, bucket=BUCKET, folder=ECG_FOLDER):
    """
    Reads the cohort CSV and retrieves ECG JSON files from S3.
    Returns a DataFrame with a column 'X_test' containing numpy arrays.
    """
    ecg_data = []
    df = pd.read_csv(cohort_csv)
    
    for _, row in df.iterrows():
        bed = row['bed']
        patientid = row['patientid']
        start = row['start']
        end = row['end']
        file_key = f"{folder}{bed}_{patientid}_{start}_{end}.json"
        try:
            response = s3_client.get_object(Bucket=BUCKET, Key=file_key)
            json_content = response['Body'].read().decode('utf-8')
            json_data = json.loads(json_content)
            ecg_list = json_data.get('ecg_ii', [])
            ecg = np.array(ecg_list, dtype=np.float64)
            ecg_data.append(ecg)
        except Exception as e:
            print(f"Error processing {file_key}: {e}")
    
    X_test_ecg = pd.DataFrame({'X_test': ecg_data})
    return X_test_ecg

def predict(ecg):
    """
    Predict labels for the given ECG data using a pre-trained model.
    This dummy implementation returns random probabilities.
    """
    # Replace this with your model inference code:
    import numpy as np
    pred_label_prob = np.random.rand(len(ecg), 3)  # dummy probabilities
    predicted_labels = np.argmax(pred_label_prob, axis=1)
    labels = ['NOTEB', 'VEB', 'SVEB']
    predicted_label = [labels[i] for i in predicted_labels]
    max_probabilities = np.max(pred_label_prob, axis=1)
    probs = [round(i, 2) for i in max_probabilities.tolist()]
    return pred_label_prob, predicted_label, probs



def main():
    # Step 1: Download and load the output manifest
    local_manifest = "output.manifest"
    download_file_from_s3(BUCKET, OUTPUT_MANIFEST_KEY, local_manifest)
    manifest = load_manifest(local_manifest)
    
    # Step 2: Download worker annotations (assumed downloaded)
    worker_files = download_worker_annotations()
    if worker_files:
        sample_annotation = read_sample_worker_annotation(worker_files[0])
        print("Sample worker annotation keys:", sample_annotation.keys())
    
    # Step 3: Create patient data CSV from prelabel inference manifest
    create_patient_data_csv(".json", "data.csv")
    
    # Step 4: Check manifest for client errors
    check_client_errors(manifest)
    
    # Step 5: Extract annotation labels from first manifest entry (if available)
    if manifest:
        labels = extract_annotation_labels(manifest[0])
        for label in labels:
            print(f"Beat from {label[0]} to {label[1]} is {label[2]}")
    
    # Step 6: Load cohort data
    cohort = load_cohort_data("filtered_data.csv")
    print("Cohort data loaded:", cohort.head())
    
    # Step 7: Load ECG data from S3 based on the CSV
    X_test_ecg = load_ecg_data_from_s3("filtered_data.csv")
    print("ECG data shape:", X_test_ecg.shape)
    
    # Step 8: Predict using the model (dummy prediction here)
    # For demonstration, assuming X_df is prepared from X_test_ecg
    X_df = X_test_ecg['X_test']  # In practice, you'll preprocess as needed
    pred_label_prob, predicted_label, probs = predict(X_df)
    print("Predicted labels:", predicted_label)
    
    # Step 9: Evaluate predictions (using dummy GP_labels here)
    # Replace GP_labels with actual ground truth labels when available.
    GP_labels = np.random.randint(0, 3, size=len(predicted_label))
    evaluate_predictions(GP_labels, predicted_label)
    
    # Example: Count predicted labels
    print("Label counts:", Counter(predicted_label))


if __name__ == "__main__":
    main()
