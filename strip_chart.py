import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import ast
from GCUH_preprocessing import set_window_size, resample_ecg, auto_gen_bed_times, get_ecg, getECG, get_R_peak_windowed, zero_mean_normalise

def construct_query(bed, start_timestamp, end_timestamp):
    """
    Construct the SQL query for fetching ECG data for a given patient.
    """
    year, month, day = start_timestamp.split(' ')[0].split('-')
    query = f"""
        SELECT timestamp, "values" FROM "waveform"."waveform-new" where
        hospital = '' and
        unit = 'ICU' and
        room = '-' and
        bed = '{bed}' and
        year = {year} and
        month = {int(month)} and
        day = {int(day)} and
        observation_type = 'ECG' and
        observation_subtype = 'II' 
        and timestamp > TIMESTAMP '{start_timestamp}' and timestamp < TIMESTAMP '{end_timestamp}' order by timestamp;
    """
    return query

def string_to_float_list(string):
    return [float(x) for x in ast.literal_eval(string)]


def process_ecg_data(ecg_data, model):
    """
    Process ECG data and predict labels.
    """
    output_list = []
    rr_list = []
    ecg = ecg_data["values"].to_numpy()
    ecg_converted = np.concatenate([string_to_float_list(item) for item in ecg])
    sig_df, sig_norm_df, qrs_inds, N = get_R_peak_windowed(ecg_converted, fs = 128)
    rr = np.diff(qrs_inds)
    X_test = np.expand_dims(sig_norm_df.iloc[:,:127], 2)
    pred_label_prob = model.predict(X_test)
    predicted_labels = np.argmax(pred_label_prob, axis = -1)
    max_probabilities = np.max(pred_label_prob, axis=-1)
    label= ['0', '1', '2']
    predicted_label = [label[i] for i in predicted_labels]
    output_dict = {'R_peaks': qrs_inds,
                'Label': predicted_label,
                    }

    output = pd.DataFrame(output_dict)
    output['Label'] = output['Label'].replace({'0': 'Other', '1': 'VEB', '2': 'SVEB'})
    output_list.append(output)
            
            
    rr_dict = {'rr': rr}
    rr_df = pd.DataFrame(rr_dict)
    rr_list.append(rr_df)
    
    if len(output_list) > 0:
        output_df = pd.concat(output_list, ignore_index=True)
    else:
        output_df = pd.DataFrame(columns=cols)
        
    if len(rr_list) > 0:
        rr_df = pd.concat(rr_list, ignore_index=True)
    else:
        rr_df = pd.DataFrame(columns=['rr'])
        
        
    clear_output(wait=True)
    print('Lable prediction completed')
    return output_df, rr_df, sig_norm_df

def get_plot_handles():
    other_patch = mpatches.Patch(color='#4DAF4A', label='Other')
    veb_patch = mpatches.Patch(color='#FF5733', label='VEB')
    sveb_patch = mpatches.Patch(color='#000000', label='SVEB')
    return [other_patch, veb_patch, sveb_patch]

def plot_strip_chart(output_df, save_path):
    fig, ax = plt.subplots(figsize=(20, 4))

    for j in range(1, len(output_df)):
        if output_df['Label'][j-1] == 'Other':
            colour = '#4DAF4A'  # Green to represent 'Other'
        elif output_df['Label'][j-1] == 'VEB':
            colour = '#FF5733'  # Red to represent 'VEB'
        else:
            colour = '#000000'  # Black to represent 'SVEB'

        ax.barh(0.5, output_df['R_peaks'][j] - output_df['R_peaks'][j-1], color=colour, edgecolor=colour, left=output_df['R_peaks'][j-1])

    ax.set_xlabel('Time (samples)')
    ax.legend(handles=get_plot_handles(), bbox_to_anchor=(1.0, 0.75))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    
def plot_individual_beats(output_df, sig_norm_df):
    sveb_beats = output_df[output_df['Label'] == 'SVEB']
    veb_beats = output_df[output_df['Label'] == 'VEB']

    # Total number of plots needed
    total_plots = len(sveb_beats) + len(veb_beats)

    # Create subplots
    fig, axes = plt.subplots(nrows=total_plots, ncols=1, figsize=(8, 5 * total_plots), sharex=True)

    if total_plots > 1:
        axes = axes.flatten()  # Ensure axes is always a 1D array for easy indexing

    # Function to plot each beat
    def plot_beat(ax, beat_index, label, title):
        waveform = sig_norm_df.iloc[beat_index]
        ax.plot(waveform)
        ax.set_title(f'{title} Beat at R_peak {beat_index}')
        ax.label_outer()  # Only show x-labels for the bottom subplot

    # Plot SVEB and VEB beats
    plot_index = 0
    for idx, row in sveb_beats.iterrows():
        plot_beat(axes[plot_index], idx, 'SVEB', 'SVEB')
        plot_index += 1

    for idx, row in veb_beats.iterrows():
        plot_beat(axes[plot_index], idx, 'VEB', 'VEB')
        plot_index += 1

    plt.tight_layout()
    plt.show()
    

def save_individual_beats(output_df, sig_norm_df, patient_id):
    output_folder = f'beats/{patient_id}'

    sveb_beats = output_df[output_df['Label'] == 'SVEB']
    veb_beats = output_df[output_df['Label'] == 'VEB']
    total_plots = len(sveb_beats) + len(veb_beats)

    plot_index = 0
    for label, beats in [('SVEB', sveb_beats), ('VEB', veb_beats)]:
        for idx, row in beats.iterrows():
            fig, ax = plt.subplots(figsize=(8, 5))
            waveform = sig_norm_df.iloc[idx]
            ax.plot(waveform)
            ax.set_title(f'{label} Beat at R_peak {idx}')
            plt.tight_layout()

            # Save the figure
            fig.savefig(f'{output_folder}/{label}_Beat_{idx}.png')
            plt.close(fig)
            plot_index += 1

def label_strip_chart_for_n_patients(df, model, num_patients):
    cols = ['patientid', 'R_peaks', 'label']
    output_list = []
    rr_list = []

    patient_ids = df['patientid'].unique()[:num_patients]

    for patientid in patient_ids:
        patient_df = df[df['patientid'] == patientid]
        start = patient_df.iloc[0]['time_start']
        end = patient_df.iloc[-1]['time_end']

        start_dt = datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
        end_dt = datetime.datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
        strip_duration = (end_dt - start_dt).total_seconds() / 60
        
        try:
            ecg = getECG(patientid, start, end, ECG_cols=['ecg_ii'])


            ecg = ecg.astype('float64')
            sig_df, sig_norm_df, qrs_inds, N = get_R_peak_windowed(ecg, fs = 128)

            rr = np.diff(qrs_inds)
            X_test = np.expand_dims(sig_norm_df.iloc[:,:127], 2)
            pred_label_prob = model.predict(X_test)
            predicted_labels = np.argmax(pred_label_prob, axis = -1)
            max_probabilities = np.max(pred_label_prob, axis=-1)
            label= ['0', '1', '2']
            predicted_label = [label[i] for i in predicted_labels]
            output_dict = {'patientid': patientid,
                           'R_peaks': qrs_inds,
                           'Label': predicted_label,
                           'Confidence':max_probabilities
                          }

            output = pd.DataFrame(output_dict)
            output['Label'] = output['Label'].replace({'0': 'Other', '1': 'VEB', '2': 'SVEB'})
            output_list.append(output)
            
            
            rr_dict = {'patientid': patientid, 'rr': rr}
            rr_df = pd.DataFrame(rr_dict)
            rr_list.append(rr_df)
            
        except:
            print(f"Error occurred for patient {patientid}. Moving to next patient.")
            continue
    
    if len(output_list) > 0:
        output_df = pd.concat(output_list, ignore_index=True)
    else:
        output_df = pd.DataFrame(columns=cols)
        
    if len(rr_list) > 0:
        rr_df = pd.concat(rr_list, ignore_index=True)
    else:
        rr_df = pd.DataFrame(columns=['patientid', 'rr'])
        
        
    clear_output(wait=True)
    print('Lable prediction completed')
    return output_df, rr_df



hospital = ''
unit = 'ICU'
room = '-'
bed = 'BED33'
start_timestamp = ''
end_timestamp = ''

query = construct_query(bed, start_timestamp, end_timestamp)
ecg_data = athena.query_as_pandas(query)

#process ECG data
output_df, rr_df, sig_norm_df = process_ecg_data(ecg_data, model)

# Plot the strip chart
plot_strip_chart(output_df, 'strip_chart_patient2.png')

