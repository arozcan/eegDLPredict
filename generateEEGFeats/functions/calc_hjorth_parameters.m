function [hjorth_parameters] = calc_hjorth_parameters(signal,fs,window_length,overlap)
    offset=(ceil(1/(1-overlap))-1);
    hjorth_parameters=zeros(floor(length(signal{1,2})/(window_length*(1-overlap)*fs))-offset,2,size(signal,1));
    for j=1:size(signal,1)
        for k=1:floor(length(signal{1,2})/(window_length*(1-overlap)*fs))-offset
            ind_start = (k-1)*fs*window_length*(1-overlap)+1;
            ind_end = ind_start+fs*window_length-1;
            
            window=signal{j,2}(ind_start:ind_end);
            dwindow = diff([0;window]);
            ddwindow = diff([0;dwindow]);
            mx2 = abs(mean(window.^2));
            mdx2 = abs(mean(dwindow.^2));
            mddx2 = abs(mean(ddwindow.^2));

            mob = mdx2 / mx2;
            mobility = sqrt(mob);
            complexity = sqrt(mddx2 / mdx2 - mob);
            
            hjorth_parameters(k,1,j)=mobility;
            hjorth_parameters(k,2,j)=complexity;
        end
    end
end