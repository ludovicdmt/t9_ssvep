T9 BCI using SSVEP
====

Python scripts of a Brain Computer Interface (BCI) using SSVEP stimuli to operate a T9 (11 classes). The online (synchronous) classification relies on [TRCA](https://ieeexplore.ieee.org/document/7904641). The GUI is using Psychopy<sup>3</sup>.  
It was developped in the [Human-Factors department](https://personnel.isae-supaero.fr/neuroergonomie-et-facteurs-humains-dcas?lang=en) of ISAE-Supaero (France) by the team under the supervision of [Frédéric Dehais](https://personnel.isae-supaero.fr/frederic-dehais/).  

The code was used for the last experiment described in our paper [Improving user experience of SSVEP BCI through low amplitude depth and high frequency stimuli design](https://www.nature.com/articles/s41598-022-12733-0) (2022, Scientific Report).

## Contents

[Dependencies](#dependencies)  
[Installation](#installation)  
[Example usage](#example-usage)  
[Help](#help)

## Dependencies

* [Psychopy<sup>3</sup>](https://www.psychopy.org/download.html)
* [MNE](https://mne.tools/stable/install/mne_python.html)
* [pylsl](https://github.com/chkothe/pylsl)
* [Sklearn](https://scikit-learn.org/stable/install.html)
* [Pyriemann](https://github.com/pyRiemann/pyRiemann)
* Pickle

## Installation

Clone the repo:

```bash
git clone https://github.com/ludovicdmt/t9_ssvep
cd ${INSTALL_PATH}
```

Install conda dependencies and the project with

```bash
conda env create -f environment.yml
```

The `pyRiemann` package has to be installed separately using `pip`:
```bash
conda activate psychopy
pip install pyriemann
```

If the dependencies in `environment.yml` change, update dependencies with

```bash
conda env update --file environment.yml
```

You will need to set path to the scripts in either `run_T9.sh` or `run_T9.bat`.  
Depending on your EEG system, you may want to change the [electrodes selection](https://github.com/ludovicdmt/t9_ssvep/blob/main/classification/online_T9.py#L387). Here it works with BrainProduct system streaming the electrode number, instead of their names.

This was tested using an EEG BrainProduct system with a native sampling frequency F<sub>s</sub> of 500Hz, downsampled to 250Hz for the TRCA algorithm. If your EEG system uses a lower sampling frequency, please consider changing [the downsampling argument](https://github.com/ludovicdmt/t9_ssvep/blob/main/classification/online_T9.py#L432), to ensure that the signal fed to TRCA algorithm is sampled at 250Hz.

## Example Usage

Run it on Linux:

```bash
cd ${INSTALL_PATH}
chmod u+x run_T9.sh
./run_T9.sh
```

Or on Windows, just click on `run_T9.bat`.  

The script will first run a calibration phase, used to train the classification model. This model will be saved in the main directory, along calibration data. This model can be re-used within the same session and with the same subject.

> A PyLSL stream from an EEG is required to make the script run.

To change frequencies, phases or amplitude of the stimuli please go to the [config file](https://github.com/ludovicdmt/t9_ssvep/blob/main/presentation/T9_config_control.json).  

## Help

You will probably need to do some adjustement to collect EEG stream if you are not using a BrainProduct EEG.  
If you experience issues during  use of this code, you can post a new issue on the [issues webpage](https://github.com/ludovicdmt/t9_ssvep/issues).  
I will reply to you as soon as possible and I'm very interested in to improve it.

## References

1. M. Nakanishi, Y. Wang, X. Chen, Y. -T. Wang, X. Gao, and T.-P. Jung, "Enhancing detection of SSVEPs for a high-speed brain speller using task-related component analysis", IEEE Trans. Biomed. Eng, 65(1): 104-112, 2018. <http://ieeexplore.ieee.org/document/7904641/>
2. X. Chen, Y. Wang, M. Nakanishi, X. Gao, T. -P. Jung, S. Gao, "High-speed spelling with a non-invasive brain-computer interface", Proc. Natl. Acad. Sci. U.S.A, 112(44): E6058-E6067, 2015. <http://www.pnas.org/content/early/2015/10/14/1508080112.abstract>
3. Peirce, J. W., Gray, J. R., Simpson, S., MacAskill, M. R., Höchenberger, R., Sogo, H., Kastman, E., Lindeløv, J. (2019). PsychoPy2: experiments in behavior made easy. Behavior Research Methods. 10.3758/s13428-018-01193-y
