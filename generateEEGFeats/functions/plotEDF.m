function [sigMin,sigMax]= plotEDF(edfFile,startTime,endTime,fTitle,sigEMin,sigEMax)
    [header signalHeader signalCell] = blockEdfLoad(edfFile);

    % Create figure
    %fig = figure();

    % Plot First 30 Seconds
    tmin = startTime;
    tmax = endTime;

    % Get number of signals
    num_signals = header.num_signals;
    
    % Add each signal to figure
    for s = 1:num_signals
        % get signal
        signal =  signalCell{s};
        samplingRate = signalHeader(s).samples_in_record;
        record_duration = header.data_record_duration;
        t = [0:length(signal)-1]/samplingRate';

        % Identify first 30 seconds
        indexes = find(tmin<=t & t<tmax);
        signal = signal(indexes);
        t = t(indexes);

        % Normalize signal
        if isempty(sigEMin)
            sigMin(s) = min(signal);
            sigMax(s) = max(signal);
        else
            sigMin(s) = sigEMin(s);
            sigMax(s) = sigEMax(s);
        end
        signalRange = sigMax(s) - sigMin(s);
        signal = (signal - sigMin(s));
        if signalRange~= 0
            signal = signal/(sigMax(s)-sigMin(s));
        end
        signal = signal -0.5*mean(signal) + num_signals - s + 1;

        % Plot signal
        edf_plot=plot(t, signal);
        hold on
    end

    % Set title
    %title(['Hasta - ',header.local_rec_id,fTitle]);
    title([fTitle]);

    % Set axis limits
    v = axis();
    v(1:2) = [tmin tmax];
    v(3:4) = [-0.5 num_signals+1.5];
    axis(v);

    % Set x axis
    xlabel('Zaman(sn)');

    % Set yaxis labels
    signalLabels = cell(1,num_signals);
    for s = 1:num_signals
        signalLabels{num_signals-s+1} = signalHeader(s).signal_labels;
    end
    set(gca, 'YTick', [1:1:num_signals]);
    set(gca, 'YTickLabel', signalLabels);
end