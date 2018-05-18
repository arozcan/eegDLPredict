function [ filtered_signal ] = filterNoises( signal, fs )
    ac_noise_filter = designfilt('bandstopiir','FilterOrder',2, ...
               'HalfPowerFrequency1',57,'HalfPowerFrequency2',63, ...
               'DesignMethod','butter','SampleRate',fs);
    for i=1:length(signal)
        signal{i,2} = filtfilt(ac_noise_filter,signal{1,2});
    end
end
