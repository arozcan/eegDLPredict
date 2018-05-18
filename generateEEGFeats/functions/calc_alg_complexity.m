function [ alg_complexity ] = calc_alg_complexity( signal,fs,window_length,overlap )
    %CALC_ALG_COMPLEXITY Summary of this function goes here
    %   Detailed explanation goes here
    offset=(ceil(1/(1-overlap))-1);
    alg_complexity=zeros(floor(length(signal{1,2})/(window_length*(1-overlap)*fs))-offset,1,size(signal,1));
    for j=1:size(signal,1)
        for k=1:floor(length(signal{1,2})/(window_length*(1-overlap)*fs))-offset
            ind_start = (k-1)*fs*window_length*(1-overlap)+1;
            ind_end = ind_start+fs*window_length-1;
            window=signal{j,2}(ind_start:ind_end);
            complexity = sum(abs(diff(window)))/(length(window)-1);       
            alg_complexity(k,1,j)=complexity;
        end
    end

end

