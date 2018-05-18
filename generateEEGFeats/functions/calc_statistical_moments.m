function [statistical_moments] = calc_statistical_moments(signal,fs,window_length, overlap)
    offset=(ceil(1/(1-overlap))-1);
    statistical_moments=zeros(floor(length(signal{1,2})/(window_length*(1-overlap)*fs))-offset,4,size(signal,1));
    for k=1:floor(length(signal{1,2})/(window_length*(1-overlap)*fs))-offset
        for j=1:size(signal,1)
            ind_start = (k-1)*fs*window_length*(1-overlap)+1;
            ind_end = ind_start+fs*window_length-1;
            signal_window=signal{j,2}(ind_start:ind_end);
            statistical_moments(k,1,j)=mean(signal_window);
            statistical_moments(k,2,j)=var(signal_window);
            statistical_moments(k,3,j)=skewness(signal_window);
            statistical_moments(k,4,j)=kurtosis(signal_window);
        end
    end
end