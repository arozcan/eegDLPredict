function testBlockEdfLoadDistribution()
% testBlockEdfLoad. Test function for block EDF load that reads the EDF file
% in three blocks (header, signal header and signal block). Reading the
% signal block in one load allows for matrix rotations to be applied to
% extract all the signals simulteanously. The loader is faster than
% methods that read the signal one entry at a time.
%
% The function is geared towards engineers, whom generally just want the
% signal for analysis. Beyond checking that the file exists, there is very
% little checking.
%
% Our EDF tools can be found at:
%
%                  http://sleep.partners.org/edf/
%
% Function prototype:
%    [header signalHeader signalCell] = blockEdfLoad(edfFN)
%
% Test files:
%     The test files are from the  from the EDF Browser website and the 
% Sleep Heart Health Study (SHHS) (see links below). The first file is
% a generated file and the SHHS file is from an actual sleep study.
%
% External Reference:
%   test_generator.edf (EDF Browswer Website)
%   http://www.teuniz.net/edf_bdf_testfiles/index.html
%
% Version: 0.1.21
%
% ---------------------------------------------
% Dennis A. Dean, II, Ph.D
%
% Program for Sleep and Cardiovascular Medicine
% Brigam and Women's Hospital
% Harvard Medical School
% 221 Longwood Ave
% Boston, MA  02149
%
% File created: October 23, 2012
% Last update:  January 23, 2014 
%    
% Copyright © [2013] The Brigham and Women's Hospital, Inc. THE BRIGHAM AND 
% WOMEN'S HOSPITAL, INC. AND ITS AGENTS RETAIN ALL RIGHTS TO THIS SOFTWARE 
% AND ARE MAKING THE SOFTWARE AVAILABLE ONLY FOR SCIENTIFIC RESEARCH 
% PURPOSES. THE SOFTWARE SHALL NOT BE USED FOR ANY OTHER PURPOSES, AND IS
% BEING MADE AVAILABLE WITHOUT WARRANTY OF ANY KIND, EXPRESSED OR IMPLIED, 
% INCLUDING BUT NOT LIMITED TO IMPLIED WARRANTIES OF MERCHANTABILITY AND 
% FITNESS FOR A PARTICULAR PURPOSE. THE BRIGHAM AND WOMEN'S HOSPITAL, INC. 
% AND ITS AGENTS SHALL NOT BE LIABLE FOR ANY CLAIMS, LIABILITIES, OR LOSSES 
% RELATING TO OR ARISING FROM ANY USE OF THIS SOFTWARE.
%

% Test Files
edfFn1 = 'test_generator.edf';    % Generated data file

%% Brief Tutorial
% The following tutorial demonstrates the commands required to load an EDF 
% into MATLAB and to access a specific signal present in the EDF file. 
% Additional examples for accessing the information available in an EDF 
% can be found in the test file that is bundled with the loader script 
% file. 


%% (1) Download loader and test file
%  Download the loader and test file from https://github.com/DennisDean/. 
%  Unzip the download into a folder. Open MATLAB and change the directory 
%  to the folder with the loader and test file.

%% (2) Prepare to load a file
%  Prepare to load the EDF by clearing the console, clearing the workspace,
%  and closing all figures.  Define the file to read. Type the following 
%  commands.
%
clc
clear
close all
edfFn1 = 'test_generator.edf';
[header signalHeader signalCell] = blockEdfLoad(edfFn1);
header

%% (3) Inspect the loaded variables
%  Type the following commands to inspect the variables created by the load command.
header
signalHeader
signalCell

% Typing each of the variables results in the variable contents to be displayed on the screen as shown below.
% >> header = 
%                  edf_ver: '0'
%               patient_id: 'test file'
%             local_rec_id: 'EDF generator'
%      recording_startdate: '02.10.08'
%      recording_starttime: '14.27.00'
%         num_header_bytes: 4352
%                reserve_1: ''
%         num_data_records: 900
%     data_record_duration: 1
%              num_signals: 16
% >> signalHeader
% signalHeader = 
% 1x16 struct array with fields:
%     signal_labels
%     tranducer_type
%     physical_dimension
%     physical_min
%     physical_max
%     digital_min
%     digital_max
%     prefiltering
%     samples_in_record
%     reserve_2
% >> signalCell
% signalCell = 
%   Columns 1 through 5
%     [180000x1 double]    [90000x1 double]    [180000x1 double]    [180000x1 double]    [45000x1 double]
%   Columns 6 through 10
%     [90000x1 double]    [180000x1 double]    [180000x1 double]    [180000x1 double]    [180000x1 double]
%   Columns 11 through 15
%     [180000x1 double]    [180000x1 double]    [180000x1 double]    [22500x1 double]    [22500x1 double]
%   Column 16
%     [22500x1 double]

%% (4) Create variables for plotting
%  Type the following command to create plotting variables.
 signal = signalCell{1};
 samplingRate = signalHeader(1).samples_in_record;
 t = [0:length(signal)-1]/samplingRate';
 numSamplesIn30Seconds = 30*samplingRate;
%% (5) Plot data
% Type the following command to plot the first 30 seconds of the first signal.
plot(t(1:numSamplesIn30Seconds), signal(1:numSamplesIn30Seconds));
title('Test Signal')
xlabel('Time (sec.)')
ylabel('Signal Amplitude')

% Successful completion of this tutorial should result in the plot of the
% first 30 seconds from first signal in test_generator.edf.
%% (5) Select EDF information to return
% Type the following command to return selected commands
clc;clear 
edfFn1 = 'test_generator.edf';
header = blockEdfLoad(edfFn1);
who
[header signalHeader] = blockEdfLoad(edfFn1);
who
[header signalHeader signalCell] = blockEdfLoad(edfFn1);
who
% Typing each of the commands results on the following console display.
% >> edfFn1 = 'test_generator.edf';
% >> header = blockEdfLoad(edfFn1);
% >> who
% Your variables are:
% edfFn1  header  
% >> [header signalHeader] = blockEdfLoad(edfFn1);
% >> who
% Your variables are:
% edfFn1        header        signalCell    signalHeader  
% >> [header signalHeader signalCell] = blockEdfLoad(edfFn1);
% >> who
% Your variables are:
% edfFn1        header        signalCell    signalHeader  
% >>
%% (6) Select signals to return
% Type the following commands
edfFn1 = 'test_generator.edf';
signalList = {'F3' 'F4'};
[header signalHeader signalCell] = blockEdfLoad(edfFn1, signalList);
header
% Typing each of the commands results in two signals being loaded.
% 
% >> edfFn1 = 'test_generator.edf';
% >> signalList = {'F3' 'F4'};
% >> [header signalHeader signalCell] = blockEdfLoad(edfFn1, signalList);
% >> header
% 
% header = 
% 
%                  edf_ver: '0'
%               patient_id: 'test file'
%             local_rec_id: 'EDF generator'
%      recording_startdate: '02.10.08'
%      recording_starttime: '14.27.00'
%         num_header_bytes: 4352
%                reserve_1: ''
%         num_data_records: 900
%     data_record_duration: 1
%              num_signals: 2
%% (7) Select 30 second epochs to return
% Type the following commands
signalList = {'F4' 'F3'};
epochs = [2 4];  % [start stop ] epochs
[header signalHeader signalCell] = ...
                         blockEdfLoad(edfFn1, signalList);
signalCell
[header signalHeader signalCell] = ...
                         blockEdfLoad(edfFn1, signalList,epochs);
signalCell 
% Typing each of the commands results in two signals being loaded.
% 
% >> edfFn1 = '201434.EDF';
% >> signalList = {'SaO2' 'EEG'};
% >> epochs = [2 4];  % start and stop epochs
% >> [header signalHeader signalCell] = …
%                          blockEdfLoad(edfFn1, signalList);
% >> signalCell
% signalCell = 
% 
%     [37740x1 double]    [4717500x1 double]
% >> [header signalHeader signalCell] = …
%                          blockEdfLoad(edfFn1, signalList,epochs);
% >> signalCell 
% signalCell = 
% 
%     [90x1 double]    [11250x1 double]
% 
