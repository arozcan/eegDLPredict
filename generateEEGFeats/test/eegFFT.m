clc,clear
addpath('functions');
addpath('../blockEdfLoad');
addpath('../ReadSaveEDF');
dataset_link='D:\Dataset\physiobank\chbmit\'; % windows icin
%bipolar_label={'FP1-F7' 'F7-T7' 'T7-P7' 'P7-O1' 'FP1-F3' 'F3-C3' 'C3-P3' 'P3-O1' 'FP2-F4' 'F4-C4' 'C4-P4' 'P4-O2' 'FP2-F8' 'F8-T8' 'T8-P8' 'P8-O2' 'FZ-CZ' 'CZ-PZ' 'T7-FT9' 'FT9-FT10' 'FT10-T8'}';

allRecords=readFileList([dataset_link 'RECORDS.html']);
seizuredRecords=readFileList([dataset_link 'RECORDS-WITH-SEIZURES.html']);
nonSeizuredRecords = setdiff(allRecords,seizuredRecords);

load refData/seizureList.mat

index=330;

[header signalHeader signalCell] = blockEdfLoad([dataset_link allRecords{index}]);

% Get number of signals
num_signals = header.num_signals;
figure
for s = 1:num_signals
        % get signal
        signal =  signalCell{s};
        Fs = signalHeader(s).samples_in_record;
        T = 1/Fs;
        L = length(signal);
        Y = fft(signal);
        P2 = abs(Y/L);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        f = Fs*(0:(L/2))/L;
        plot(f,P1);
        hold on
        title('Single-Sided Amplitude Spectrum of X(t)')
        xlabel('f (Hz)')
        ylabel('|P1(f)|')
end
hold off