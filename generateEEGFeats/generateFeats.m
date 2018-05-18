4clc,clear
addpath('functions');
addpath('refData')
addpath('../blockEdfLoad');
%dataset_link='/Volumes/MacHDD/Dataset/physiobank/chbmit/'; % mac icin
%dataset_link='D:\Dataset\physiobank\eegmmidb\'; % windows icin
dataset_link='D:\Dataset\physiobank\chbmit\'; % windows icin
bipolar_label={'FP1-F7' 'F7-T7' 'T7-P7' 'P7-O1' 'FP1-F3' 'F3-C3' 'C3-P3' 'P3-O1' 'FP2-F4' 'F4-C4' 'C4-P4' 'P4-O2' 'FP2-F8' 'F8-T8' 'T8-P8' 'P8-O2' 'FZ-CZ' 'CZ-PZ' 'T7-FT9' 'FT9-FT10' 'FT10-T8'}';


% her hasta icin evre zamanlarini iceren tablo yukleniyor
load refData/pat_records.mat
% pencere uzunlugu 20 sn
window_length=4;
overlap = 0.5;
time_diff = false;

%% her hasta icin 4sn lik pencere kullanilarak preictal ve interictal oznitelikler cikarilacak
calc_feature=1;
if calc_feature==1
    for p=1:length(pat_records)
        % interictal kayitlar icin oznitelikler cikariliyor
        % ilgili hasta icin cikarilacak interictal featurelar
        pat_features(p).interictal.r_id=[];
        pat_features(p).interictal.s_bant_pow=[];
        pat_features(p).interictal.s_moment=[];
        pat_features(p).interictal.hjorth_p=[];
        pat_features(p).interictal.alg_c=[];
        pat_features(p).interictal.idx=[];
        pat_features(p).interictal.file={};
        pat_features(p).interictal.start=[];
        pat_features(p).interictal.end=[];
        
        rid=1;
        for i=1:length(pat_records(p).interictal)
            feats=extract_features( [dataset_link pat_records(p).interictal(i).file], pat_records(p).interictal(i).start, pat_records(p).interictal(i).end, bipolar_label,window_length,overlap,time_diff);

            if isempty(feats)
                continue;
            end

            %oznitelikler kaskad ekleniyor.
            idx=size(pat_features(p).interictal.r_id,1);
            pat_features(p).interictal.r_id=[pat_features(p).interictal.r_id; repmat(rid,feats.length,1)];
            pat_features(p).interictal.s_bant_pow=[pat_features(p).interictal.s_bant_pow; feats.spectral_bant_power];
            pat_features(p).interictal.s_moment=[pat_features(p).interictal.s_moment; feats.statistical_moments];
            pat_features(p).interictal.hjorth_p=[pat_features(p).interictal.hjorth_p; feats.hjorth_parameters];
            pat_features(p).interictal.alg_c=[pat_features(p).interictal.alg_c; feats.alg_complexity];
            pat_features(p).interictal.idx = [pat_features(p).interictal.idx; idx+1];
            pat_features(p).interictal.file{end+1} =pat_records(p).interictal(i).file;
            pat_features(p).interictal.start = [pat_features(p).interictal.start; pat_records(p).interictal(i).start];
            pat_features(p).interictal.end = [pat_features(p).interictal.end; pat_records(p).interictal(i).end];
            disp([num2str(p),'_interictal_',num2str(i)]);
            rid=rid+1;
        end

        % preictal kayitlar icin oznitelikler cikariliyor
        % ilgili hasta icin cikarilacak interictal featurelar
        pat_features(p).preictal.r_id=[];
        pat_features(p).preictal.s_bant_pow=[];
        pat_features(p).preictal.s_moment=[];
        pat_features(p).preictal.hjorth_p=[];
        pat_features(p).preictal.alg_c=[];
        pat_features(p).preictal.b_id=[];
        pat_features(p).preictal.idx=[];
        pat_features(p).preictal.file={};
        pat_features(p).preictal.start=[];
        pat_features(p).preictal.end=[];
        rid=1;
        for i=1:length(pat_records(p).preictal)
            % onceki dosyada veri var mi
            if isfield( pat_records(p).preictal(i), 'pre_file' )
                if ~isempty(pat_records(p).preictal(i).pre_file)
                    feats=extract_features( [dataset_link pat_records(p).preictal(i).pre_file], pat_records(p).preictal(i).pre_start, pat_records(p).preictal(i).pre_end, bipolar_label,window_length,overlap,time_diff);
                    if ~isempty(feats)
                        %oznitelikler kaskad ekleniyor.
                        idx=size(pat_features(p).preictal.r_id,1);
                        pat_features(p).preictal.r_id=[pat_features(p).preictal.r_id; repmat(rid,feats.length,1)];
                        pat_features(p).preictal.s_bant_pow=[pat_features(p).preictal.s_bant_pow; feats.spectral_bant_power];
                        pat_features(p).preictal.s_moment=[pat_features(p).preictal.s_moment; feats.statistical_moments];
                        pat_features(p).preictal.hjorth_p=[pat_features(p).preictal.hjorth_p; feats.hjorth_parameters];
                        pat_features(p).preictal.alg_c=[pat_features(p).preictal.alg_c; feats.alg_complexity];
                        pat_features(p).preictal.b_id=[pat_features(p).preictal.b_id; repmat(0,feats.length,1)];
                        pat_features(p).preictal.idx = [pat_features(p).preictal.idx; idx+1];
                        pat_features(p).preictal.file{end+1}=pat_records(p).preictal(i).pre_file;
                        pat_features(p).preictal.start = [pat_features(p).preictal.start; pat_records(p).preictal(i).pre_start];
                        pat_features(p).preictal.end = [pat_features(p).preictal.end; pat_records(p).preictal(i).pre_end];
                    end
                end
            end

            % o dosyadaki veri
            feats=extract_features( [dataset_link pat_records(p).preictal(i).file], pat_records(p).preictal(i).start, pat_records(p).preictal(i).end, bipolar_label,window_length,overlap,time_diff);

            if isempty(feats)
                continue;
            end

            %oznitelikler kaskad ekleniyor.
            idx=size(pat_features(p).preictal.r_id,1);
            pat_features(p).preictal.r_id=[pat_features(p).preictal.r_id; repmat(rid,feats.length,1); ];
            pat_features(p).preictal.s_bant_pow=[pat_features(p).preictal.s_bant_pow; feats.spectral_bant_power];
            pat_features(p).preictal.s_moment=[pat_features(p).preictal.s_moment; feats.statistical_moments];
            pat_features(p).preictal.hjorth_p=[pat_features(p).preictal.hjorth_p; feats.hjorth_parameters];
            pat_features(p).preictal.alg_c=[pat_features(p).preictal.alg_c; feats.alg_complexity];
            pat_features(p).preictal.b_id=[pat_features(p).preictal.b_id; repmat(1,feats.length,1)];
            pat_features(p).preictal.idx = [pat_features(p).preictal.idx; idx+1];
            pat_features(p).preictal.file{end+1}= pat_records(p).preictal(i).file;
            pat_features(p).preictal.start = [pat_features(p).preictal.start; pat_records(p).preictal(i).start];
            pat_features(p).preictal.end = [pat_features(p).preictal.end; pat_records(p).preictal(i).end];
            disp([num2str(p),'_preictal_',num2str(i)]);
            rid=rid+1;
        end
    end
    save('dataset/pat_features','pat_features');
end

%% dataset python ortaminda islenecek formata donusturuluyor
load dataset/pat_features.mat
conv_feature=1;
if conv_feature==1
    for p=1:length(pat_records)
        % interictal oznitelikler donusturuluyor
        % windows x features x channels
        feature=[];
        feature=cat(2,pat_features(p).interictal.s_bant_pow,pat_features(p).interictal.s_moment, pat_features(p).interictal.hjorth_p, pat_features(p).interictal.alg_c );
        % windows x [features x channels]
        feature=reshape(permute(feature,[1 3 2]),[],size(feature,2)*size(feature,3));
        % kayit etiketi sona ekleniyor
        % windows x ([features x channels]+r_id)
        feature=cat(2,feature,pat_features(p).interictal.r_id);
        save(['dataset/pat_' num2str(p) '_int.mat'],'feature');
        
        % preictal oznitelikler donusturuluyor
        % windows x features x channels
        feature=[];
        feature=cat(2,pat_features(p).preictal.s_bant_pow,pat_features(p).preictal.s_moment,pat_features(p).preictal.hjorth_p, pat_features(p).preictal.alg_c );
        % windows x [features x channels]
        feature=reshape(permute(feature,[1 3 2]),[],size(feature,2)*size(feature,3));
        % kayit etiketleri sona ekleniyor
        % windows x ([features x channels]+b_id+r_id)
        feature=cat(2,feature,pat_features(p).preictal.b_id,pat_features(p).preictal.r_id);
        save(['dataset/pat_' num2str(p) '_pre.mat'],'feature');
        
        % patient info
        % interictal pencere sayisi
        pat_info.interictal_count=size(pat_features(p).interictal.s_bant_pow,1);
        % preictal pencere sayisi
        pat_info.preictal_count=size(pat_features(p).preictal.s_bant_pow,1);
        % interictal kayit indeksleri
        [val idx]=unique(pat_features(p).interictal.r_id);
        pat_info.interictal_idx=idx;
        % preictal kayit indeksleri
        [val idx]=unique(pat_features(p).preictal.r_id);
        pat_info.preictal_idx=idx;
        % diger ozellikler
        pat_info.interictal.idx=pat_features(p).interictal.idx;
        pat_info.interictal.file=pat_features(p).interictal.file';
        pat_info.interictal.start=pat_features(p).interictal.start;
        pat_info.interictal.end=pat_features(p).interictal.end;
        pat_info.preictal.idx=pat_features(p).preictal.idx;
        pat_info.preictal.file=pat_features(p).preictal.file';
        pat_info.preictal.start=pat_features(p).preictal.start;
        pat_info.preictal.end=pat_features(p).preictal.end;
        save(['dataset/pat_' num2str(p) '_info.mat'],'pat_info');
    end
end

