"""
Utility functions for the online scripts
"""

import os
import sys

import numpy as np

from subprocess import call
from tempfile import gettempdir

def beep(waveform=(79, 45, 32, 50, 99, 113, 126, 127)):
    """Play a beep sound.

    Cross-platform sound playing with standard library only, no sound
    file required.

    From https://gist.github.com/juancarlospaco/c295f6965ed056dd08da
    """
    wavefile = os.path.join(gettempdir(), "beep.wav")
    if not os.path.isfile(wavefile) or not os.access(wavefile, os.R_OK):
        with open(wavefile, "w+") as wave_file:
            for sample in range(0, 300, 1):
                for wav in range(0, 8, 1):
                    wave_file.write(chr(waveform[wav]))
    if sys.platform.startswith("linux"):
        return call("chrt -i 0 aplay '{fyle}'".format(fyle=wavefile),
                    shell=1)
    if sys.platform.startswith("darwin"):
        return call("afplay '{fyle}'".format(fyle=wavefile), shell=True)
    if sys.platform.startswith("win"):  # FIXME: This is Ugly.
        return call("start /low /min '{fyle}'".format(fyle=wavefile),
                    shell=1)