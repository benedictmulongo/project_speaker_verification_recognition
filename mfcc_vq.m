%addpath lab_library/;
%Add the directory and its subdirectories 
%addpath(genpath('lab_library/'))
% clc % clear command window

function [ads, speaker, labelCount] = mfcc_vq() 
    file_location = fullfile("speech_commands_data");
    ads = audioDatastore(file_location, 'IncludeSubfolders',true,'LabelSource','folderNames', 'FileExtensions','.wav');
    ads = subset(ads,ads.Labels==categorical("stop"));
    %
    [~,fileName] = cellfun(@(x)fileparts(x),ads.Files,'UniformOutput',false);
    fileName = split(fileName,'_');
    speaker = strcat('a',fileName(:,1));
    ads.Labels = categorical(speaker);
    labelCount = countEachLabel(ads);
    
    
    %Read from the training datastore and listen to a file. Reset the datastore.
    [audioData,audioInfo] = read(ads);
    fs = audioInfo.SampleRate;
    t = (0:size(audioData,1)-1)/fs;
    sound(audioData,fs)
    figure;
    plot(t,audioData)
    xlabel('Time (s)')
    ylabel('Amplitude')
    axis([0 t(end) -1 1])
    title('Sample Utterance from Training Set')
    reset(ads)
    
    % Features extractions
    numCoeffs = 20;
    deltaWindowLength = 9;
    windowDuration = 0.025;
    hopDuration = 0.01;
    %afe = feature_Extraction(fs, numCoeffs, deltaWindowLength, windowDuration, hopDuration);
    afe = feature_mfcc(fs, windowDuration, hopDuration);
    
    disp("mfcc : ");
    [audioDataX, fs1] = audioread(char(ads.Files(1,1)));    
    [audioData1, fs2] = audioread(char(ads.Files(100,1)));    
    [audioData2, fs3] = audioread(char(ads.Files(200,1)));    
    
    % Normalize the audio.
    %audioData1 = audioData1./max(abs(audioData1));
    %idx = detectSpeech(audioData1,fs);
    %audioData1 = audioData1(idx(1,1):idx(1,2));
    % Call extract on the audioFeatureExtractor object to extract features from audio data.
    features = extract(afe, audioData1);
    [numHops, numFeatures] = size(features)
    features = features';
    disp("Plot mfcc : ")
    figure;
    imagesc(features(2:end,:))
    colormap winter;
    figure;
    pcolor(features(2:end,:))
    colormap winter;
    
    
end

function [afe] = feature_mfcc(fs, windowDuration, hopDuration)

    windowSamples = round(windowDuration*fs)
    hopSamples = round(hopDuration*fs)
    overlapSamples = windowSamples - hopSamples;

    afe = audioFeatureExtractor('SampleRate',fs,'Window',hann(windowSamples,'periodic'), ...
        'OverlapLength',overlapSamples,'mfcc',true);
end 


function [afe] = feature_Extraction(fs, numCoeffs, deltaWindowLength, windowDuration, hopDuration)

    windowSamples = round(windowDuration*fs);
    hopSamples = round(hopDuration*fs);
    overlapSamples = windowSamples - hopSamples;

    afe = audioFeatureExtractor('SampleRate',fs, 'Window',hann(windowSamples,'periodic'), 'OverlapLength',overlapSamples,...
        'mfcc',true,'mfccDelta',true, 'mfccDeltaDelta',true);
    setExtractorParams(afe,'mfcc','DeltaWindowLength',deltaWindowLength,'NumCoeffs',numCoeffs)

end 