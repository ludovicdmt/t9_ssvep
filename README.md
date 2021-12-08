T9_SSVEP
====

Python scripts of a Brain Computer Interface (BCI) using SSVEP stimuli to operate a T9 (11 classes). The online (synchronous) classification relies on [TRCA](https://ieeexplore.ieee.org/document/7904641).


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
* Pickle 

## Installation
Clone repo
```bash
cd ${INSTALL_PATH}
git clone https://github.com/ludovicdmt/t9_ssvep
```
You will need to set path in either `run_T9.sh` or `run_T9.bat` and install the dependencies.  
Depending on your EEG system, you may want to change the [electrodes selection](https://github.com/ludovicdmt/t9_ssvep/blob/main/classification/online_T9.py#L387). Here it works with BrainProduct system streaming the electrode number, instead of their names. 

## Example Usage
Run it on Linux:
```bash
cd ${INSTALL_PATH}
chmod u+x run_T9.sh
./run_T9.sh
```
Or on Windows, just click on `run_T9.bat`

> A PyLSL stream from an EEG is required to make the script run.  

## Help
You will probably need to do some adjustement to collect EEG stream if you are not using a BrainProduct EEG.  
If you experience issues during  use of this code, you can post a new issue on the [issues webpage](https://github.com/ludovicdmt/t9_ssvep/issues).  
I will reply to you as soon as possible and I'm very interested in to improve it.