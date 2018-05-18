function [ feats ] = extract_features( file, f_start, f_end, bipolar_label, window_length, overlap, time_diff)
        [header signalHeader signalCell] = blockEdfLoad(file);

        signal_labels=struct2cell(signalHeader');
        signal_labels=signal_labels(1,:);
        % kayitta bulunan kanallarin listesi kontrol ediliyor.
        if ~isempty(find(ismember(bipolar_label,signal_labels)==0))
            feats=[];
            return;
        end

        sample_in_record=signalHeader(1).samples_in_record;
        % bipolar olan kayitlar unipolara cevriliyor
        signal= convertBipolar2UnipolarBasic( signalHeader,signalCell,...
                           [sample_in_record*(f_start)+1 ...
                           sample_in_record*(f_end)]);
        fs=signalHeader(1).samples_in_record;
        % sinyal gurultuden temizleniyor
        %interictal_signal=filterNoises(interictal_signal,fs);

        %oznitelikler cikariliyor
        offset=(ceil(1/(1-overlap))-1);
        feats.length = floor(length(signal{1,2})/(window_length*(1-overlap)*fs))-offset;
        feats.spectral_bant_power=calc_spectral_band_power(signal,fs,window_length,overlap, time_diff);
        feats.statistical_moments=calc_statistical_moments(signal,fs,window_length,overlap);
        feats.hjorth_parameters= calc_hjorth_parameters(signal,fs,window_length,overlap);
        feats.alg_complexity = calc_alg_complexity(signal,fs,window_length,overlap);
end
