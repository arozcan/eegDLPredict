clc,clear
addpath('functions');
addpath('refData')
addpath('../blockEdfLoad');
%dataset_link='/Volumes/MacHDD/Dataset/physiobank/chbmit/'; % mac icin
%dataset_link='D:\Dataset\physiobank\eegmmidb\'; % windows icin
dataset_link='D:\Dataset\physiobank\chbmit\'; % windows icin

allRecords=readFileList([dataset_link 'RECORDS.html']);
seizuredRecords=readFileList([dataset_link 'RECORDS-WITH-SEIZURES.html']);
nonSeizuredRecords = setdiff(allRecords,seizuredRecords);
k=1;
for i=1:size(seizuredRecords,2)
    [count,seizure_start, seizure_length ] = get_seizure_period( [dataset_link seizuredRecords{i} '.seizures']);
    for j=1:count
        seizureList(k).file = seizuredRecords{i};
        seizureList(k).seizure_start = seizure_time(j);
        seizureList(k).seizure_length = seizure_length(j);
        k=k+1;
    end
end
save('seizureList_new.mat','seizureList');

