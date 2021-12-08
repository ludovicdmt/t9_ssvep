"""
Script to read and classify EEG data in real time.

Authors: udovic Darmet, Juan Jesus Torre Tresols 
Mail: ludovic.darmet@siae-supaero.fr; Juan-jesus.TORRE-TRESOLS@isae-supaero.fr
"""

import argparse
from math import inf
import os
import pickle

import sys

import numpy as np
import pandas as pd

from pylsl import (
    StreamInfo,
    StreamInlet,
    StreamOutlet,
    resolve_byprop,
    local_clock,
)
from subprocess import call

from TRCA import TRCA

try:
    import winsound  # For Windows only
except:
    pass

parser = argparse.ArgumentParser(description='Parameters for the experiment')
parser.add_argument('-e', '--epochlen', metavar='EpochLength', type=float,
                    default=2.0, help="Length of each data epoch used for "
                                      "classification. Default: %(default)s.")
parser.add_argument('-b', '--buffer', metavar='BufferLength', type=int,
                    default=4.0, help="Length of the data buffer in seconds. "
                                    "Default: %(default)s.")
parser.add_argument('-ds', '--datastream', metavar='DataStream', type=str,
                    default='SimulatedData', help="Name of the data stream to look for")
parser.add_argument('-ms', '--markerstream', metavar='MarkerStream', type=str,
                    default='MyMarkerStream', help="Name of the marker stream to look for")
parser.add_argument('-m', '--mode', metavar='SampleMode', type=str,
                    default='ms', choices=['samples', 'ms'],
                    help="Format for the event timestamps. Can be samples or miliseconds. "
                         "Default: %(default)s. Choices: %(choices)s")

args = parser.parse_args()

## Argparse Parameters
buffer_len = args.buffer # Length of the array that keeps the data stored from the stream (s)
epoch_len = args.epochlen  # Length of each data epoch used as observations
data_name = args.datastream
marker_name = args.markerstream
sampling_mode = args.mode


def beep(waveform=(79, 45, 32, 50, 99, 113, 126, 127), win_freq=740):
    """
    Play a beep sound.

    Cross-platform sound playing with standard library only, no sound
    file required.

    From https://gist.github.com/juancarlospaco/c295f6965ed056dd08da
    """
    wavefile = os.path.join(os.getcwd(), "beep.wav")
    if not os.path.isfile(wavefile) or not os.access(wavefile, os.R_OK):
        with open(wavefile, "w+") as wave_file:
            for sample in range(0, 300, 1):
                for wav in range(0, 8, 1):
                    wave_file.write(chr(waveform[wav]))
    if sys.platform.startswith("linux"):
        return call("chrt -i 0 aplay '{fyle}'".format(fyle=wavefile), shell=1)
    if sys.platform.startswith("darwin"):
        return call("afplay '{fyle}'".format(fyle=wavefile), shell=True)
    if sys.platform.startswith("win"):  # FIXME: This is Ugly.
        winsound.Beep(win_freq, 500)
        return



def get_label_dict(info, n_class):
    """
    Get label names from stream info

    Parameters
    ----------

    info: LSL info object
        LSL Info object containing the label names.

    n_class: int
        Number of classes.

    Returns
    -------

    label_dict: dict
        Dictionary containing label info. Keys are label IDs and values are the label
        digit associated to them.
    """

    labels = info.desc().child("events_labels").child_value()
    label_list = [label for label in labels.split(",")]  # Formatting

    label_dict = {freq: idx for idx, freq in enumerate(set(label_list))}


    return label_dict


def get_channel_names(info):
    """
    Get channel names from stream info

    Parameters
    ----------

    info: LSL info object
        LSL Info object containing the label names.

    Returns
    -------

    ch_names: list
        Names of each channel, corresponding to the rows of the data
    """

    n_chan = info.channel_count()

    ch = info.desc().child("channels").first_child()
    ch_names = [ch.child_value("label")]

    for _ in range(n_chan - 1):
        ch = ch.next_sibling()
        ch_names.append(ch.child_value("label"))

    return ch_names


def get_ch_idx(ch_names, ch_to_keep=None):
    """
    Return the indices of the desired elements of the list.

    Parameters
    ----------

    ch_names : list of str
        List containing the names of all the channels. These must
        be in the same order they are contained in the data array.

    ch_to_keep : list of str or None, default = None
        Channels to keep. If None, all channels are kept.

    Returns
    -------

    ch_idxs : list of int
        Indices of the channels to keep.
    """

    if not ch_to_keep:
        ch_to_keep = ch_names

    ch_dict = {ch: idx for idx, ch in enumerate(ch_names)}

    ch_idxs = [ch_dict.get(ch) for ch in ch_to_keep]

    return ch_idxs


def get_trial(data_inlet, marker_inlet, labels, buffer_len=4, return_timestamps=True):
    """
    Get trigger-related data from the LSL data stream. The function never returns until
    a valid trial is completed. If a new event marker is received before a trial is completed,
    previous data is discarded in favor of the new trial. If a trigger corresponding to the
    current trial is received while data collection is in progress, it is interpreted as a
    cancel signal. In this case, the trial is dropped and the function waits for a new trial
    to re-start the process.

    Parameters
    ----------

    data_inlet: LSL StreamInlet
        The LSL stream that sends EEG data

    marker_inlet: LSL Stream Inlet
        LSL stream in charge of the event markers

    labels: list of str
        IDs of the triggers corresponding to the beginning of the trial

    buffer_len: float, default=4.
        Length of the data buffer in seconds

    return_timestamps: bool, default=True
        If True, return the list with timestamps for all samples
        of the trial

    Returns
    -------

    trial: np.array of shape (n_channels, n_samples)
        EEG data corresponding to the trial

    label: int
        Label corresponding to the trial

    epoch_times: list of float
        List containing the timestamps associated with the
        sending time of each sample of the trial. Only returned
        if return_timestamps=True
    """

    # Data parameters
    n_chan = data_inlet.info().channel_count()
    sfreq = int(data_inlet.info().nominal_srate())

    # Buffer to deque incoming training data
    data_buffer = np.zeros((n_chan, sfreq * buffer_len)) - 10
    _, buffer_samples = data_buffer.shape

    # Buffer to deque incoming timestamps
    times_buffer = np.zeros((buffer_samples)) - 10

    # Target time initialization
    target_time = inf

    # How much samples to collect
    samp = 10

    got_marker = False

    while True:
        # Pull data in small chunks
        eeg_data, data_times = data_inlet.pull_chunk(timeout=1/(2*sfreq), max_samples=samp)
        

        if eeg_data:
            # Prepare the data
            eeg_array = np.array(eeg_data).T
            times_array = np.array(data_times)

            if sampling_mode == "ms":
                times_array = np.round(times_array, 3)  # In miliseconds

            # Deque data and times arrays
            data_buffer = np.hstack((data_buffer, eeg_array))
            data_buffer = data_buffer[..., -buffer_samples:]

            times_buffer = np.hstack([times_buffer, times_array])

            times_buffer = times_buffer[-buffer_samples:]


            # Check if there is an event marker
            if got_marker == False:
                marker, marker_time = marker_inlet.pull_sample(timeout=0.0)
                if marker:
                    label = marker[0].split(",")[0]
                    if label in labels:
                        marker_time = np.array(marker_time)
                        got_marker = True

            if got_marker:
                #print(times_array)
                # Get more samples per pull_chunk
                # as we are not waiting for a marker anymore
                samp = int(sfreq * epoch_len / 4.)

                # Store label and timestamp
                true_label = event_id[label]

                if sampling_mode == "ms":
                    marker_time = np.round(marker_time, 3)
                # Parameter to modify the calculation of target time
                if sampling_mode == "ms":
                    time_mod = 1  # Timestamps from LiveAmp come already in miliseconds
                elif sampling_mode == "samples":
                    time_mod = sfreq

                # Find your target time
                total_len = epoch_len * time_mod + delay # epoch_len is in s so we convert to ms
                target_time = np.round(marker_time + total_len, 3)  # In samples

            if times_buffer[-1] > target_time:

                print(f"Marker time for the beginning of the epoch is: {marker_time + delay}")
                print(f"Target timestamp for the end of this epoch is: {target_time}")
                print("")

                # Find the index of the first and last sample
                first_sample = np.where(
                    times_buffer >= marker_time + delay
                )[0][0]
                last_sample = int(first_sample + (sfreq * epoch_len))
                
                if last_sample < buffer_samples-1:
                    print('First sample', first_sample, 'last sample', last_sample)
                    while last_sample - first_sample != epoch_len * sfreq:
                        print("Another one")
                        last_sample += 1
                    # Keep only the channels we are interested in
                    data_buffer = data_buffer[ch_to_keep, :]
                    # Slice the thing
                    epoch = data_buffer[:, first_sample:last_sample]

                    epoch_times = times_buffer[first_sample:last_sample]
                    # Average re-referencing
                    ref_data = epoch.mean(0, keepdims=True)
                    epoch -= ref_data

                    # Baseline correction
                    mean = np.mean(epoch, axis=1, keepdims=True)
                    epoch -= mean

                    # Reset target time
                    target_time = inf

                    # Reset number of samples per pull_chunk
                    samp = 10
                    got_marker = False

                    if return_timestamps:
                        return epoch, true_label, epoch_times
                    else:
                        return epoch, true_label


## LSL streams
# Create outlet for clf signal
clf_info = StreamInfo(
    name="TRCAOutput",
    type="TRCA",
    channel_count=1,
    nominal_srate=500.0,
    channel_format="int8",
    source_id="coolestIDever1234",
)

clf_outlet = StreamOutlet(clf_info)

# First resolve a data stream
print("Looking for a data stream...")
data_streams = resolve_byprop("type", "EEG", timeout=5)

# If nothing is found, raise an error
if len(data_streams) == 0:
    raise (RuntimeError("Can't find EEG stream..."))
else:
    print("Data stream found!")

# Then resolve the marker stream
print("Looking for a marker stream...")
marker_streams = resolve_byprop("name", marker_name, timeout=120)

# If nothing is found, raise an error
if len(marker_streams) == 0:
    raise (RuntimeError("Can't find marker stream..."))
else:
    print("Marker stream found!")

# Get data inlet
data_inlet = StreamInlet(data_streams[0], max_buflen=10, max_chunklen=1, processing_flags=1) # max_buflen should be in s
marker_inlet = StreamInlet(marker_streams[0], max_chunklen=1, processing_flags=1)

# Get the stream info and description
marker_info = marker_inlet.info()
data_info = data_inlet.info()

description = marker_info.desc()

## Parameters
buffer_len = args.buffer  # Length of the array that keeps the dnamesata stored from the stream (s)
epoch_len = args.epochlen  # Length of each data epoch used as observations
sfreq = int(data_info.nominal_srate())
delay = int(sfreq * 0.135)  # In samples
if sampling_mode == "ms":
    delay *= 1 / sfreq  # In ms
n_chan = data_info.channel_count()
n_samples = int(sfreq * epoch_len)  # Number of samples per epoch

ch_slice = ['O1', 'O2', 'Oz', 'P3', 'P4', 'Pz', 'P7', 'P8']  # Channels to keep
ch_slice = ['13', '14', '15', '16', '17', '18', '19', '20']

## CLF parameters
n_classes = int(description.child("n_class").child_value())  # Number of classes
n_train = int(description.child("n_train").child_value())  # Calibration trials per class
labels = description.child("events_labels").child_value()  # List containing all the stim triggers
cues = description.child("cues_labels").child_value()  # List containing all the cue triggers
filename = description.child("filename").child_value()  # String containing participant and session number
amp = description.child("amp").child_value()  # String containing amplitude of the stimuli
cal_trials = n_train * n_classes

labels = [label for label in labels.split(",")]
print("Labels", labels)
cues = [cue for cue in cues.split(",")]

event_id = get_label_dict(marker_info, n_classes)
ch_names = get_channel_names(data_info)
print(ch_names)

ch_to_keep = get_ch_idx(ch_names, ch_slice)
print(f"\n Channels number to keep: {ch_to_keep} \n")

# Parameters for the LSL processing
data_buffer_len = 4  # In seconds

peaks = [float(key.split('_')[0]) for key in event_id.keys()]
if np.max(peaks) < 20:
    nfbands = 5
else: 
    nfbands = 2

if np.max(peaks) > 20:
    cond = '_high_'
else:
    cond = '_low_'
    
amp = '_amp' + amp + '_' 

## Load or create model
model_filename = os.path.join(os.getcwd(), filename + cond + amp +  "TRCA_calibration.sav")
caldata_filename = os.path.join(os.getcwd(), filename + cond + amp +  "calibration_data.npy")
trustscore_filename = os.path.join(os.getcwd(), filename + cond + amp +  "_scores.csv")

try:
    X_train, y_train = pickle.load(open(caldata_filename, "rb"))
    clf = TRCA(sfreq=sfreq*1.0, peaks=peaks, downsample=2, n_fbands=nfbands, method='original', regul='lwf', trustscore=False)
    clf.fit(X_train, y_train)
    model_loaded = True

    print(f"Using saved data - {model_filename}")
    print("")

except FileNotFoundError:
    clf = TRCA(sfreq=sfreq*1.0, peaks=peaks, downsample=2, n_fbands=nfbands, method='original', regul='lwf', trustscore=False)
    model_loaded = False

    print("Calibration file not found...")
    print("")

## Skip calibration if model was found
if not model_loaded:
    ## CALIBRATION
    print("")
    print("-" * 21)
    print("Starting calibration")
    print("-" * 21)
    print("")

    print(f"Expected number of classes: {n_classes}")
    print(f"Expected number of calibration trials (per class): {n_train}")
    print("")
    print(f"Expected number of calibration trials (total): {cal_trials}")
    print("")

    X_train = np.zeros((cal_trials, len(ch_to_keep), n_samples))
    y_train = []

    # Pause the execution to set up what you need. Unpause with Intro key
    # print("Ready to start calibration, press the 'Intro' key to start...\n")


    # Number of training trial
    training_idx = 0

    # Get time 0 to correct the timestamps
    t0 = local_clock()

    while training_idx < cal_trials:
        print(training_idx)
        # Get training trial
        cal_trial, true_label, epoch_times = get_trial(
            data_inlet, marker_inlet, labels, buffer_len=data_buffer_len
        )

        # Add the epoch to the training data with its corresponding label
        X_train[training_idx, :, :] = cal_trial
        y_train.append(true_label)

        print(f"Start and end of the epoch: {epoch_times[0]}, {epoch_times[-1]}")
        print(f"Correctly stored calibration trial n {training_idx + 1}")
        print("")

        target_time = 0
        training_idx += 1

    print(f"Calibration data recorded. Final shape of X_train: {X_train.shape}")
    pickle.dump((X_train, y_train), open(caldata_filename, "wb"))

    print("")
    print("-" * 21)
    print("Fitting training data...")
    print("-" * 21)
    print("")

    clf.fit(X_train, y_train)

    print("Data was fit. Calibration complete")
    print("")

    ## Save the model if calibration was done
    pickle.dump(clf, open(model_filename, "wb"))

    print("")
    print(f"Model saved in {model_filename}")

prediction = []
test_idx = 0

## TESTING
# Pause the execution to set up what you need. Unpause with Intro key
#print("Ready to start testing, press the 'Intro' key to start...\n")
outputs = {"y_pred": [], "y_true": []}
while True:
    # Get test trial
    X_test, true_label, epoch_times = get_trial(data_inlet, marker_inlet, labels, buffer_len=data_buffer_len)

    # Predict on your data and check if it is correct
    y_pred = clf.predict(X_test)
    for k,v in event_id.items():
        if v==y_pred[0]:
            pred = k.split('_')[-1]
    for k,v in event_id.items():
        if v==true_label:
            true = k.split('_')[-1]
    if pred == 'Back':
        pred = 10
    else:
        pred = int(pred)
    if true == 'Back':
        true = 10
    else:
        true = int(true)

    outputs['y_pred'].append(pred)
    outputs['y_true'].append(true)

    clf_outlet.push_sample([pred])

    print(f"Start and end of the epoch: {epoch_times[0]}, {epoch_times[-1]}")
    print("")

    if y_pred[0] == true_label:
        prediction.append(1)
        print("Correct prediction!")
        beep()

    else:
        last_test = X_test
        prediction.append(0)
        print("Booooh")
        beep(win_freq=440)

    print(f"Predicted label: {pred}, True_label: {true}")
    print("-" * 20)
    print("")

    target_time = 0
    test_idx += 1

    df = pd.DataFrame(outputs)
    df.to_csv(trustscore_filename, index=None)
    print(f"Clf score: {sum(prediction) / len(prediction)}")
