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
% index=2;
% SP=seizureList(index).start_time;
% [sigMin,sigMax]= plotEDF( [dataset_link seizureList(index).file],seizureList(index).start_time-15,seizureList(index).start_time+15,'preictal evre                                ictal evre','','');
% y1=get(gca,'ylim')
% hold on
% plot([SP SP],y1,'r');
% 
% index=3;
% start_time=1300;
% [sigMin,sigMax]= plotEDF( [dataset_link nonSeizuredRecords{index}],start_time,start_time+15,'interictal evre',sigMin,sigMax);
% 
index=3;
start_time=1800;
[sigMin,sigMax]= plotEDF( [dataset_link nonSeizuredRecords{index}],start_time,start_time+15,'','','');
plot([start_time+3 start_time+3],y1,'r--');
plot([start_time+6 start_time+6],y1,'r--');
plot([start_time+9 start_time+9],y1,'r--');
plot([start_time+12 start_time+12],y1,'r--');
grid on
set(gca,'XTick',[]);


eegimages=images_timewin(:,1200,:,:,:);
eegimages=squeeze(eegimages);
title_list = {'delta(0.5-4Hz)' 'teta(4-8Hz)' 'alfa(8-13Hz)' 'beta(13-30Hz)' 'gama-1(30-47Hz)' 'gama-2(53-75Hz)' 'gama-3(75-97Hz)' 'gama-4(103-128Hz)' 'mean' 'var' 'skewness' 'kurtosis' 'mobility' 'complexity'};
for j=1:14
    figure;
    for i=1:5
        subplot(2,3,i);
        set(gca,'position',[0.015*i 0.7+0.015*i 0.2 0.2]);
        imagesc(squeeze(eegimages(i,j,:,:)));
        set(gca,'YTick',[]);
        set(gca,'XTick',[]);
        if i==5
         title(title_list{j});
        end
    end
end
