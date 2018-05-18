clc,clear
addpath('dataset')

% önce interictal veri aliniyor
load pat_2_int.mat
feat_int=feature;
% sonra preictal veri aliniyor
load pat_2_pre.mat
feat_pre=feature;
feat_all=[feat_int(:,1:280) zeros(size(feat_int,1),1) ;feat_pre(:,1:280) ones(size(feat_pre,1),1)];
feat_all(:,1:280)=normc(feat_all(:,1:280));
% 
% [coeff,score,latent,tsquared,explained] = pca(feat_all(:,1:280));
% 
% scatter3(score(1:size(feat_int,1),1),score(1:size(feat_int,1),2),score(1:size(feat_int,1),3),'b')
% hold on
% scatter3(score(size(feat_int,1):end,1),score(size(feat_int,1):end,2),score(size(feat_int,1):end,3),'r')

[trainedClassifier, validationAccuracy] = trainClassifier(feat_all);