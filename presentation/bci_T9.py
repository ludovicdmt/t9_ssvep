"""
T9 type numerical keyboard using flickers for BCI-demonstration

Authors: Juan Jesus Torre Tresols and Ludovic Darmet
email: Juan-jesus.TORRE-TRESOLS@isae-supaero.fr
"""

import json
import os
import platform
import random
import numpy as np
import datetime
import argparse

from psychopy import visual, core, event, gui 
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop, local_clock

from utils_experiments import get_screen_settings, CheckerBoard

wave_type = 'sin'

def pause():
    """Pause execution until the 'c' key is pressed"""

    paused = True

    while paused:
        if event.getKeys('space'):
            paused = False
        elif event.getKeys('s'):
            return "Skip"



# Load config file
path = os.getcwd()


parser = argparse.ArgumentParser(description='Config file name')
parser.add_argument('-f', '--file', metavar='ConfigFile', type=str,
                    default='T9_config_control.json', help="Name of the config file for freq "
                                      "and amplitude. Default: %(default)s.")
                                      
args = parser.parse_args()
config_path = os.path.join(path, args.file)

with open(config_path, 'r') as config_file:
    params = json.load(config_file)

# Experimental params
size = params['size']
trial_n = params['trial_n']
cal_n = params['cal_n']
epoch_duration = params['epoch_duration']
iti_duration = params['iti_duration']
cue_duration = params['cue_duration']

# Stim params
# Stim params
if wave_type=='code':
    freqs = np.arange(0,11,1)
elif wave_type=='mseq':
    freqs = np.arange(0,11,1)
elif wave_type =='modulation':
    freqs = np.arange(0,11,1)
else:
    freqs = params['freqs']
positions = [tuple(position) for position in params['positions']]  # JSON does not like tuples...
phases = params['phases']
number_codes = params['flicker_codes']
amp = params['amplitude']

# Classification inlet
clf_stream = resolve_byprop("name", "TRCAOutput", timeout=5)
# If nothing is found, raise an error
if len(clf_stream) == 0:
    raise (RuntimeError("Can't find classification stream..."))
else:
    print("Marker stream found!")

# Get data inlet
clf_inlet = StreamInlet(clf_stream[0], max_chunklen=1, processing_flags=1)
y_pred, timestamp = clf_inlet.pull_sample(timeout=0.0)

# Store info about the experiment session
expName = 'T9' 
expInfo = {'participant': '', 'session': '001'}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = datetime.date.today()  # add a simple timestamp
filename =  u'%s_P%s_sess%s_%s' % (expName, expInfo['participant'] ,expInfo['session'], expInfo['date'])

# Marker stream
info = StreamInfo(name='MyMarkerStream', type='Markers', channel_count=1,
                  nominal_srate=0, channel_format='string', source_id='myuidw43536')
info.desc().append_child_value("n_train", f"{cal_n}")
info.desc().append_child_value("n_class", f"{len(number_codes)}")
classes = []
for idx, f in enumerate(freqs):
    classes.append(str(f)+ '_' + str(phases[idx]) + '_' + str(number_codes[idx]))

info.desc().append_child_value("events_labels", ','.join(classes))
info.desc().append_child_value("filename", filename)
info.desc().append_child_value("amp", str(amp))
outlet = StreamOutlet(info)


# Window parameters
system = platform.system()
width, height = get_screen_settings(system)

window = visual.Window([width, height], screen=1, color=[-1.000,-1.000,-1.000],\
blendMode='avg', useFBO=True, units="pix", fullscr=True)
refresh_rate = round(window.getActualFrameRate())

# Time conversion to frames
epoch_frames = int(epoch_duration * refresh_rate)
iti_frames = int(iti_duration * refresh_rate)
iti_frames_cal = int(0.8 * refresh_rate)
cue_frames = int(cue_duration * refresh_rate)

# Stim
calib_text_start = "Calibration will start. Please try avoid moving, blinking 182 or chewing during stimulation. \n \
 Follow the red cue (or the rabbit and eat the red pill). \n \n \
 Please press space when you are ready. \n \n \
 If you want to skip calibration please press S."

calib_text_end = "Calibration is over. The classification will start. \n \
     Follow the guideline on the left. Output will be displayed on the right \n \n Please press space when you are ready."

cal_start = visual.TextStim(window, text=calib_text_start)
cal_end = visual.TextStim(window, text=calib_text_end)


cue_size = size + 5
cue = visual.Rect(window, width=cue_size, height=cue_size,
                  pos=[0, 0], lineWidth=10, lineColor='red')

code_seq = visual.TextStim(win=window, text="", pos=(-650, 75), color=(+1., +1., +1.), height=50)

clf_seq = visual.TextStim(win=window, text="", pos=(650, 75), color=(+1., +1., +1.), height=50)

buttons = {f"{code}": visual.TextStim(win=window, text=code, pos=pos, color=(-1., -1., -1.), height=35)
           for pos, code in zip(positions, number_codes)}

flickers = {f"{code}": CheckerBoard(window=window, size=size, frequency=freq, phase=phase, amplitude=amp, 
                                    wave_type=wave_type, duration=epoch_duration, fps=refresh_rate,
                                    base_pos=pos)
            for freq, pos, phase, code in zip(freqs, positions, phases, number_codes)}

# Experiment structure

# Calibration
trial_list = []
symbols = list(np.array(range(0,10)).astype(str))
symbols.append('Back')
for _ in range(cal_n):
    sequence = random.sample(symbols, 11)
    trial_list.append(sequence)

# Presentation
trialClock = core.Clock()
cal_start.draw()
window.flip()
out = pause()

if out != "Skip":

    for idx_block, sequence in enumerate(trial_list):
        
        # Draw the number codes
        for button in buttons.values():
            button.autoDraw = False

        test_stream = resolve_byprop("name", "TRCAOutput", timeout=5)
        # If nothing is found, raise an error
        if len(test_stream) == 0:
            txt = f'Classification stream is dead. \n \n Press any key to quit the experiment.'
            visual.TextStim(window, text=txt).draw()
            window.flip()
            paused = True
            while paused:
                if event.getKeys():
                    paused = False
            core.quit()
        else:
            del test_stream

        txt = f'Block {idx_block+1} out of {len(trial_list)}. \n Please press space to continue.'
        visual.TextStim(window, text=txt).draw()
        window.flip()
        pause()

        # Draw the number codes
        for button in buttons.values():
            button.autoDraw = True

        # For each number in our sequence...
        for target in sequence:

            # Select target flicker
            target_flicker = flickers[str(target)]
            target_pos = (target_flicker.base_x, target_flicker.base_y)
            target_freq = target_flicker.freq
            target_phase = target_flicker.phase

            # ITI presentation
            for n in range(iti_frames_cal):
                for flicker in flickers.values():
                    flicker.draw2(frame=0, amp_override=1.)
                window.flip()

            # Cue presentation
            cue.pos = target_pos
            for frame in range(cue_frames):
                for flicker in flickers.values():
                    flicker.draw2(frame=0, amp_override=1.)

                # Draw the cue over the static flickers
                cue.draw()
                window.flip()

            # Flicker presentation
            marker_info = [f"{target_freq}_{target_phase}_{target}"]
            outlet.push_sample(marker_info)

            frames = 0
            t0 = trialClock.getTime()  # Retrieve time at trial onset

            for frame, n in enumerate(range(epoch_frames)):
                for flicker in flickers.values():
                    flicker.draw2(frame=frame)
                frames += 1
                window.flip()

            # At the end of the trial, calculate real duration and amount of frames
            t1 = trialClock.getTime()  # Time at end of trial
            elapsed = t1 - t0
            print(f"Time elapsed: {elapsed}")
            print(f"Total frames: {frames}")
            print("")

    for button in buttons.values():
            button.autoDraw = False
    cal_end.draw()
    window.flip()
    pause()

# Testing trials
trial_list = []
for _ in range(trial_n):
    sequence = random.sample(symbols, 4)
    trial_list.append(sequence)

# Presentation
trialClock = core.Clock()

for idx_block, sequence in enumerate(trial_list):
    # Draw the number codes
    for button in buttons.values():
        button.autoDraw = False
    txt = f'Block {idx_block+1} out of {len(trial_list)}. \n Please press space to continue. \n Press Escape or Q to quit.'
    visual.TextStim(window, text=txt).draw()
    window.flip()
    pause()

    for key in event.getKeys():
        if key in ['escape', 'q']:
            core.quit()

    # Set the sequence and draw
    code_seq.text = " ".join(map(str, sequence))
    code_seq.autoDraw = True

    # Draw the number codes
    for button in buttons.values():
        button.autoDraw = True

    # For each number in our sequence...
    y = []
    for target in sequence:
        # Select target flicker
        target_flicker = flickers[str(target)]
        target_pos = (target_flicker.base_x, target_flicker.base_y)
        target_freq = target_flicker.freq
        target_phase = target_flicker.phase

        # ITI presentation
        for n in range(iti_frames):
            for flicker in flickers.values():
                flicker.draw2(frame=0, amp_override=1.)
            window.flip()

        code_seq.autoDraw = False
        clf_seq.autoDraw = False
        # Cue presentation
        cue.pos = target_pos
        for frame in range(cue_frames):
            for flicker in flickers.values():
                flicker.draw2(frame=0, amp_override=1.)
            # Draw the cue over the static flickers
            cue.draw()
            window.flip()

        # Flicker presentation
        marker_info = [f"{target_freq}_{target_phase}_{target}"]
        outlet.push_sample(marker_info)

        frames = 0
        t0 = trialClock.getTime()  # Retrieve time at trial onset

        for frame, n in enumerate(range(epoch_frames)):
            for flicker in flickers.values():
                flicker.draw2(frame=frame)
            frames += 1
            window.flip()

        # At the end of the trial, calculate real duration and amount of frames
        t1 = trialClock.getTime()  # Time at end of trial
        elapsed = t1 - t0
        print(f"Time elapsed: {elapsed}")
        print(f"Total frames: {frames}")
        print("")

        # Retrieve the prediction and display it
        y_pred, timestamp = clf_inlet.pull_sample(timeout=0.0)
        while y_pred==None:
            y_pred, timestamp = clf_inlet.pull_sample(timeout=0.0)
        local_timestamp = local_clock()
        print(timestamp - local_timestamp)
        if y_pred[0]==10:
            y.append('Back')
        else:
            y.append(str(y_pred[0]))

        clf_seq.text = " ".join(map(str, y))
        clf_seq.autoDraw = True
        code_seq.autoDraw = True

    for button in buttons.values():
        button.autoDraw = False
    window.flip()
    core.wait(4)

    # Stop drawing the text
    code_seq.autoDraw=False
    clf_seq.autoDraw = False
