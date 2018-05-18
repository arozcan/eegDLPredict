clc,clear
addpath('functions');
addpath('refData')
addpath('../blockEdfLoad');
%dataset_link='/Volumes/MacHDD/Dataset/physiobank/chbmit/'; % mac icin
%dataset_link='D:\Dataset\physiobank\eegmmidb\'; % windows icin
dataset_link='D:\Dataset\physiobank\chbmit\'; % windows icin
bipolar_label={'FP1-F7' 'F7-T7' 'T7-P7' 'P7-O1' 'FP1-F3' 'F3-C3' 'C3-P3' 'P3-O1' 'FP2-F4' 'F4-C4' 'C4-P4' 'P4-O2' 'FP2-F8' 'F8-T8' 'T8-P8' 'P8-O2' 'FZ-CZ' 'CZ-PZ' 'T7-FT9' 'FT9-FT10' 'FT10-T8'}';

% veriseti eeg kayit bilgileri
load refData/records.mat
% veriseti nobet bilgileri
load refData/seizureList.mat

%% her hasta icin preictal ve ictal evrelerin belirlenmesi
% preictal evre uzunlugu 30 dk
preictal_len=60;
% sph uzunlugu 0 dk
sph_len=1;
% interictal evrenin ictal evreye uzakligi 60 dk
intercital_dist=61;
% minimum interictal uzunluðu 15 dk
minInterictal_len=30;
% minimum preictal uzunluðu 15 dk
minPreictal_len=15;
% postictal evre uzunlugu 10 dk
postictal_len=10;

% hasta sayisi
C=cellfun(@int16,{records.subject},'unif',0);
[patient,pat_idx]=unique(cell2mat(C));
%son hasta icin durdurucu index ekleniyor
pat_idx=[pat_idx;length(records)+1];

% nobet iceren dosyalar
C=cellfun(@char,{seizureList.file},'unif',0);
[seizured_file,sei_idx]=unique(C,'stable');
sei_idx=[sei_idx;length(seizureList)+1];


%% records tablosuna kayýtlar arasýndaki zamanlar ekleniyor
for p=1:length(patient)
    % her hastanin kayitlari ayri degerlendirilecek
    records(pat_idx(p)).time_diff=0;
    for r=pat_idx(p)+1:pat_idx(p+1)-1
        % iki kayýt arasýndaki zaman
        records(r).time_diff=seconds(records(r).startTime-records(r-1).endTime);
        if records(r).time_diff < 0
            records(r).time_diff = records(r).time_diff + 24*60*60;
        end
    end
end

%% records tablosuna nöbet içeren satirlar icin ilk nobetin baslangic ve
% son nobetin bitis zamanlari ekleniyor.
for p=1:length(seizured_file) 
    record_idx=find(strcmp([records.file],seizured_file(p))==1);
    % toplam nobet sayisi
    records(record_idx).s_count=sei_idx(p+1)-sei_idx(p);
    % kayit icerisindeki ilk seizure baslangici
    records(record_idx).s_start=seizureList(sei_idx(p)).seizure_start;
    % kayit icerisindeki son seizure bitisi
    records(record_idx).s_end=seizureList(sei_idx(p+1)-1).seizure_end;
end

%% her hasta icin interictal kayitlar belirleniyor
for r=1:length(records)
    records(r).isInterIctal = 'yes';
    records(r).i_start=0;
    records(r).i_end=records(r).length;
end

for p=1:length(patient)
    
    for r=pat_idx(p):pat_idx(p+1)-1
        % bu kayit nobet iceriyor mu?
        if records(r).isSeizured
            records(r).isInterIctal = 'no';
            %onceki kayitlari degerlendir
            total_int=records(r).s_start+records(r).time_diff;   
            k=r-1;
            while k>=pat_idx(p) && total_int<intercital_dist*60
                i_end=records(k).length-(intercital_dist*60 - total_int);
                if i_end <= 0
                    records(k).isInterIctal = 'no';
                elseif records(k).isSeizured == 0 && ~strcmp(records(k).isInterIctal,'no')
                    records(k).isInterIctal = 'semi';
                    records(k).i_end = i_end;
                end
                if ~isempty(records(k).i_start)
                     if records(k).i_start >= i_end
                        records(k).isInterIctal = 'no';
                     end
                end
                total_int = total_int + records(k).length + records(k).time_diff;
                k=k-1;
            end
            %sonraki kayitlari degerlendir
            total_int=records(r).length-records(r).s_end + records(r+1).time_diff;   
            k=r+1;
            while k<pat_idx(p+1) && total_int<intercital_dist*60
                i_start=intercital_dist*60 - total_int;
                if i_start >= records(k).length
                    records(k).isInterIctal = 'no';
                elseif records(k).isSeizured == 0 && ~strcmp(records(k).isInterIctal,'no')
                    records(k).isInterIctal = 'semi';
                    records(k).i_start = i_start;
                    if isempty(records(k).i_end)
                        records(k).i_end=records(k).length;
                    end
                end
                if ~isempty(records(k).i_end)
                     if i_start >= records(k).i_end
                        records(k).isInterIctal = 'no';
                     end
                end
                if k+1<pat_idx(p+1)
                    total_int = total_int + records(k).length + records(k+1).time_diff;
                end
                k=k+1;
            end
        end
    end
end

for p=1:length(patient)
    k=1;
    for r=pat_idx(p):pat_idx(p+1)-1
        % her hasta icin minimum 30 dk olan interictal kayitlar
        % degerlendirmeye alinacak
        if (strcmp(records(r).isInterIctal,'yes') || strcmp(records(r).isInterIctal,'semi')) && (records(r).i_end-records(r).i_start)>(minInterictal_len*60)
            pat_records(p).interictal(k).file=char(records(r).file);
            pat_records(p).interictal(k).start=records(r).i_start;
            pat_records(p).interictal(k).end=records(r).i_end;
            pat_records(p).interictal(k).length=pat_records(p).interictal(k).end-pat_records(p).interictal(k).start;
            k=k+1;
        end
    end
end

%% her hasta icin preictal kayitlar belirleniyor
k=1;
for s=1:length(seizureList)
    p=seizureList(s).subject;
    % ilk nobetse veya ayni kayit icinde daha once baska nobet yoksa
    if s==1 || ~strcmp(seizureList(s).file,seizureList(s-1).file) 
        
        cur_ind=find(strcmp([records.file],seizureList(s).file)==1);
        % bir onceki kayit farkli hastaya mi ait
        if records(cur_ind-1).subject ~= records(cur_ind).subject
            %nobetten onceki 15+5 dakika kayit icinde mi
            if seizureList(s).seizure_start >= (minPreictal_len + sph_len)*60;
                pat_records(p).preictal(k).file=char(seizureList(s).file);
                pat_records(p).preictal(k).end=seizureList(s).seizure_start-sph_len*60;
                % preictal tam olarak 30 dk
                if seizureList(s).seizure_start >= (preictal_len + sph_len)*60;
                    pat_records(p).preictal(k).start=pat_records(p).preictal(k).end-preictal_len*60;
                % preictal 30 ile 15 dk arasinda
                else
                    pat_records(p).preictal(k).start=0;
                end
                pat_records(p).preictal(k).length=pat_records(p).preictal(k).end-pat_records(p).preictal(k).start;
                k=k+1;
            end
        % bir önceki kayit ayni hastaya aitse ve nobet iceriyorsa
        elseif records(cur_ind-1).isSeizured
            %onceki kayittaki son nobetin bitisi ile mevcut nobetin
            %baslangici arasindaki zaman yeterli mi
            if ((records(cur_ind-1).length - records(cur_ind-1).s_end) + records(cur_ind).time_diff + seizureList(s).seizure_start) >= (postictal_len + minPreictal_len + sph_len)*60
                
                % preictal tam olarak 30 dk ve ilgili kayit icinde
                if seizureList(s).seizure_start > (preictal_len + sph_len)*60
                    pat_records(p).preictal(k).file=char(seizureList(s).file);
                    pat_records(p).preictal(k).end=seizureList(s).seizure_start-sph_len*60;
                    pat_records(p).preictal(k).start=pat_records(p).preictal(k).end-preictal_len*60;
                    pat_records(p).preictal(k).length=pat_records(p).preictal(k).end-pat_records(p).preictal(k).start;
                % preictal kayitlarda yok
                elseif seizureList(s).seizure_start <= sph_len*60 && (seizureList(s).seizure_start + records(cur_ind).time_diff) > (sph_len+ minPreictal_len)*60
                    k=k-1;
                % preictal onceki kayit icinde
                elseif seizureList(s).seizure_start <= sph_len*60 
                    pat_records(p).preictal(k).file=char(records(cur_ind-1).file);
                    pat_records(p).preictal(k).end = records(cur_ind-1).length + records(cur_ind).time_diff + seizureList(s).seizure_start - sph_len*60;
                    pat_records(p).preictal(k).start = pat_records(p).preictal(k).end - preictal_len*60;
                    % preictal bitisi kayit disina tasmissa
                    if pat_records(p).preictal(k).end > records(cur_ind-1).length
                        pat_records(p).preictal(k).end = records(cur_ind-1).length;
                    end
                    % preictal baslangici bir onceki postictal'e tasmissa
                    if pat_records(p).preictal(k).start < records(cur_ind-1).s_end + postictal_len*60;
                        pat_records(p).preictal(k).start = records(cur_ind-1).s_end + postictal_len*60;
                    end
                    pat_records(p).preictal(k).length=pat_records(p).preictal(k).end-pat_records(p).preictal(k).start;
                % preictal iki kayitta birden
                else 
                    pat_records(p).preictal(k).file=char(seizureList(s).file);
                    pat_records(p).preictal(k).end=seizureList(s).seizure_start-sph_len*60;
                    pat_records(p).preictal(k).start=0;
                    pat_records(p).preictal(k).length=pat_records(p).preictal(k).end-pat_records(p).preictal(k).start;
                    if (sph_len+preictal_len)*60 > seizureList(s).seizure_start + records(cur_ind).time_diff && records(cur_ind-1).s_end + postictal_len*60 < records(cur_ind-1).length
                        pat_records(p).preictal(k).pre_file=char(records(cur_ind-1).file);
                        pat_records(p).preictal(k).pre_end = records(cur_ind-1).length;
                        pat_records(p).preictal(k).pre_start = records(cur_ind-1).length + records(cur_ind).time_diff + seizureList(s).seizure_start - (sph_len+preictal_len)*60;
                        % preictal baslangici bir onceki postictal'e tasmissa
                        if pat_records(p).preictal(k).pre_start < records(cur_ind-1).s_end + postictal_len*60;
                            pat_records(p).preictal(k).pre_start = records(cur_ind-1).s_end + postictal_len*60;
                        end
                        pat_records(p).preictal(k).length = pat_records(p).preictal(k).length + pat_records(p).preictal(k).pre_end-pat_records(p).preictal(k).pre_start;
                        % preictal yeterli uzunlukta degilse degerlendirme
                        % disi birak
                        if (pat_records(p).preictal(k).end - pat_records(p).preictal(k).start) + (pat_records(p).preictal(k).pre_end - pat_records(p).preictal(k).pre_start) < minPreictal_len*60
                            pat_records(p).preictal(k) = [];
                            k=k-1;
                        end
                    else
                        % preictal yeterli uzunlukta degilse degerlendirme
                        % disi birak
                        if pat_records(p).preictal(k).end - pat_records(p).preictal(k).start < minPreictal_len*60
                            pat_records(p).preictal(k) = [];
                            k=k-1;
                        end
                    end
                end
                k=k+1;
            end
        % bir önceki kayit ayni hastaya aitse ve nobet icermiyorsa
        else
             % preictal tam olarak 30 dk ve ilgili kayit icinde
            if seizureList(s).seizure_start > (preictal_len + sph_len)*60;
                pat_records(p).preictal(k).file=char(seizureList(s).file);
                pat_records(p).preictal(k).end=seizureList(s).seizure_start-sph_len*60;
                pat_records(p).preictal(k).start=pat_records(p).preictal(k).end-preictal_len*60;
                pat_records(p).preictal(k).length=pat_records(p).preictal(k).end-pat_records(p).preictal(k).start;
            % preictal kayitlarda yok
            elseif seizureList(s).seizure_start <= sph_len*60 && (seizureList(s).seizure_start + records(cur_ind).time_diff) > (sph_len+ minPreictal_len)*60
                k=k-1;
            % preictal onceki kayit icinde
            elseif seizureList(s).seizure_start <= sph_len*60 
                pat_records(p).preictal(k).file=char(records(cur_ind-1).file);
                pat_records(p).preictal(k).end = records(cur_ind-1).length + records(cur_ind).time_diff + seizureList(s).seizure_start - sph_len*60;
                pat_records(p).preictal(k).start = pat_records(p).preictal(k).end - preictal_len*60;
                pat_records(p).preictal(k).length=pat_records(p).preictal(k).end-pat_records(p).preictal(k).start;
                % preictal bitisi kayit disina tasmissa
                if pat_records(p).preictal(k).end > records(cur_ind-1).length
                    pat_records(p).preictal(k).end = records(cur_ind-1).length;
                end
            % preictal iki kayitta birden
            else 
                pat_records(p).preictal(k).file=char(seizureList(s).file);
                pat_records(p).preictal(k).end=seizureList(s).seizure_start-sph_len*60;
                pat_records(p).preictal(k).start=0;
                pat_records(p).preictal(k).length=pat_records(p).preictal(k).end-pat_records(p).preictal(k).start;
                if (sph_len+preictal_len)*60 > seizureList(s).seizure_start + records(cur_ind).time_diff
                    pat_records(p).preictal(k).pre_file=char(records(cur_ind-1).file);
                    pat_records(p).preictal(k).pre_end = records(cur_ind-1).length;
                    pat_records(p).preictal(k).pre_start = records(cur_ind-1).length + records(cur_ind).time_diff + seizureList(s).seizure_start - (sph_len+preictal_len)*60;
                    pat_records(p).preictal(k).length = pat_records(p).preictal(k).length + pat_records(p).preictal(k).pre_end-pat_records(p).preictal(k).pre_start;
                    % preictal yeterli uzunlukta degilse degerlendirme
                    % disi birak
                    if (pat_records(p).preictal(k).end - pat_records(p).preictal(k).start) + (pat_records(p).preictal(k).pre_end - pat_records(p).preictal(k).pre_start) < minPreictal_len*60
                        pat_records(p).preictal(k) = [];
                        k=k-1;
                    end
                else
                    % preictal yeterli uzunlukta degilse degerlendirme
                    % disi birak
                    if pat_records(p).preictal(k).end - pat_records(p).preictal(k).start < minPreictal_len*60
                        pat_records(p).preictal(k) = [];
                        k=k-1;
                    end
                end
            end
            k=k+1;
        end
    % ayni dosya icerisinde daha once baska bir nobet varsa    
    elseif strcmp(seizureList(s).file,seizureList(s-1).file) 
        %onceki nobetin bitisiyle bu nobetin baslangici arasýnda preictal
        %icin yeterli sure var mi
        if seizureList(s).seizure_start-seizureList(s-1).seizure_end >= (minPreictal_len+sph_len+postictal_len)*60
            pat_records(p).preictal(k).file=char(seizureList(s).file);
            pat_records(p).preictal(k).end=seizureList(s).seizure_start-sph_len*60;
            pat_records(p).preictal(k).start=pat_records(p).preictal(k).end-preictal_len*60;
            % preictal baslangici bir onceki postictal'e tasmissa
            if pat_records(p).preictal(k).start < seizureList(s-1).seizure_end + postictal_len*60;
                pat_records(p).preictal(k).start = seizureList(s-1).seizure_end + postictal_len*60;
            end
            pat_records(p).preictal(k).length=pat_records(p).preictal(k).end-pat_records(p).preictal(k).start;
            k=k+1;
        end
    end
    
    % sonraki nobetin subjecti farkli ise k'yi sifirla
    if s+1 <= length(seizureList) && seizureList(s+1).subject ~= seizureList(s).subject
        k=1;
    end
end

save('refData/pat_records','pat_records');





