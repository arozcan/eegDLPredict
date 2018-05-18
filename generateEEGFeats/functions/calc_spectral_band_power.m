function [spectral_bant_power] = calc_spectral_band_power(signal,fs,window_length, overlap, time_diff)
    filt_range = [57 63 ;117 123];
    filt_range = [filt_range; 15.7 16.3; 31.7 32.3; 47.7 48.3; 63.7 64.3; ...
        75.7 76.3; 79.7 80.3; 95.7 96.3; 111.7 112.3; 127.7 128];
    delta=[0.5 4];
    teta=[4 8];
    alpha=[8 13];
    beta=[13 30];
    gama1=[30 50];
    gama2=[50 75];
    gama3=[75 100];
    gama4=[100 128];
    freqs=[delta;teta;alpha;beta;gama1;gama2;gama3;gama4];
    offset=(ceil(1/(1-overlap))-1);
    spectral_bant_power=zeros(floor(length(signal{1,2})/(window_length*(1-overlap)*fs))-offset,size(signal,1),size(freqs,1));
    for k=1:floor(length(signal{1,2})/(window_length*(1-overlap)*fs))-offset
        for j=1:size(signal,1)
            ind_start = (k-1)*fs*window_length*(1-overlap)+1;
            ind_end = ind_start+fs*window_length-1;
            signal_window=signal{j,2}(ind_start:ind_end);
            if time_diff
                signal_window = diff(signal_window);
                signal_window = [signal_window;signal_window(end)];
            end
            [Pxx,F] = periodogram(signal_window,hamming(length(signal_window)),length(signal_window),fs);
            spectral_bant_power(k,j,:)=bandpoweropt(Pxx,F,freqs,filt_range,'psd');
        end
    end
    spectral_bant_power = permute(spectral_bant_power,[1 3 2]);
end