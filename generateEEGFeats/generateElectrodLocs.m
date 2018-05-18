clc,clear
%unipolar_channels={'T7','F7','FP1','P7','O1','P3','C3','F3','FT10','T8','F8','FP2','P8','O2','P4','C4','F4','Cz','FZ','FT9','PZ'}';
unipolar_channels={'AF7','AF3','AF4','AF8','FT7','FC3','FCz','FC4','FT8','T7','T8','TP7','CP3','CPZ','CP4','TP8','PO7','PO3','PO4','PO8'}';
load('refData/Standard1020Cap81.mat');
locs_labels = Standard1020Cap81.labels;
locs= [Standard1020Cap81.X Standard1020Cap81.Y Standard1020Cap81.Z];
locs3d =[];
for i=1:size(unipolar_channels,1)
    index= find(strcmp(locs_labels,unipolar_channels{i}),1);
    locs3d=[locs3d;locs(index,1) locs(index,2) locs(index,3)];
end
locs3d=locs3d*100;
save('10_20_eeg_locs.mat','locs3d');
