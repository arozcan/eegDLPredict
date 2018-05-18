function [ entropy ] = calc_entropy( signal,fs,window_length )
    f_count=1;
    m = 2;      % embedded dimension
    tau = 1;    % time delay for downsampling
    r = 0.2;    % filter factor "Approximate Entropy in the Electroencephalogram During Wake and Sleep"
    entropy=zeros(floor(length(signal{1,2})/(window_length*fs)),f_count,size(signal,1));
    for k=1:floor(length(signal{1,2})/(window_length*fs))
        for j=1:size(signal,1)
            signal_window=signal{j,2}((k-1)*fs*window_length+1:(k)*fs*window_length);
            % calculate standard deviations
            sd1 = std(signal_window);
            entropy(k,1,j)=entropy_sample(m, r*sd1, signal_window, tau);
        end
    end
end

