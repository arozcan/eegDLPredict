# eegDLPredict
EEG Seizure Prediction with Deep Learning Techniques

## Installation
eegDLPredict  is still under development

## generateEEGFeats 
Genarates EEG Features using CHB-MIT Scalp EEG Database
- generateFeatTimings.m : Generates EEG timings according to the epileptic phase phenome (interictal, precital, ictal, postictal)
- generateFeats.m: Generates EEG features using generated EEG timings
- generateFeats.m: Generates EEG features using generated EEG timings
- generateElectrodLocs.m: Generates EEG electrode locations

## eegPredict
Predicts EEG Seizures using generated features
- eegPredict/eegpredict.py: Classify EEG features as preictal and interictal using neural networks
