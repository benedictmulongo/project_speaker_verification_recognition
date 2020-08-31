%addpath lab_library/;
%Add the directory and its subdirectories 
%addpath(genpath('lab_library/'))
% clc % clear command window
function [ads, adsTrain, adsEnrollAndVerify] = mfcc_vq_recognition() 
    file_location = fullfile("speaker_data_data_same", "stop");
    ads = audioDatastore(file_location, 'IncludeSubfolders',true,'LabelSource','folderNames', 'FileExtensions','.wav');
    labelCount = countEachLabel(ads);
    
    % Split the data 
    speakersToTest = categorical(["speaker_0", "speaker_1", "speaker_2", "speaker_3", "speaker_4", ...
        "speaker_5", "speaker_6", "speaker_7", "speaker_8", "speaker_9"]); % Test data
    adsTrain = subset(ads,~ismember(ads.Labels,speakersToTest));
    adsEnrollAndVerify = subset(ads,ismember(ads.Labels,speakersToTest));
    disp("Distribution of the speakers in the Train data : ")
    countEachLabel(adsTrain)
    %%% disp("Distribution of the speakers in the Test data : ")
    %countEachLabel(adsEnrollAndVerify)
    
    speakers = unique(adsTrain.Labels);
    numSpeakers = numel(speakers);
    disp("Speakers data for-lop : ")
    for ii = 1:numSpeakers
        speaker_name = split(char(speakers(ii)),"_");
        speaker_index = str2double(speaker_name(2));
        adsTrainSubset = subset(adsTrain,adsTrain.Labels==speakers(ii));
        disp("count : ")
        disp(ii)
        disp("speaker_index : ")
        disp(speaker_index)
        disp("------------------------------------------")
    end
    

end

function [afe] = feature_mfcc(fs, windowDuration, hopDuration)
    windowSamples = round(windowDuration*fs);
    hopSamples = round(hopDuration*fs);
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