%addpath lab_library/;
%Add the directory and its subdirectories 
%addpath(genpath('lab_library/'))
% clc % clear command window
function [ads, adsTrain, adsTest, SpeakersTrainFeaturesByIdx, code] = mfcc_vectorQ_recognition() 

    file_location = fullfile("speaker_data_data_same", "stop");
    ads = audioDatastore(file_location, 'IncludeSubfolders',true,'LabelSource','folderNames', 'FileExtensions','.wav');
    labelCount = countEachLabel(ads);
    
    speakersToTest = categorical(["speaker_1", "speaker_2", "speaker_3", "speaker_4", ...
        "speaker_5", "speaker_6", "speaker_7", "speaker_8", "speaker_9", "speaker_10"]);
    
    %speakersToTest = categorical(["speaker_10", "speaker_20", "speaker_30", "speaker_40", ...
    %    "speaker_50", "speaker_60", "speaker_70", "speaker_80", "speaker_90", "speaker_100"]);
    
    ads = subset(ads,ismember(ads.Labels,speakersToTest)); % reduce to ten speakers 
    % Split the data 
    [adsTrain,adsVerify] = splitEachLabel(ads,4);
    [adsTest, ~] = splitEachLabel(adsVerify,1);
    %Read from the training datastore and listen to a file. Reset the datastore.
    [audioData,audioInfo] = read(ads);
    fs = audioInfo.SampleRate;
    reset(ads)
    
    % Features extractions
    numCoeffs = 20;
    deltaWindowLength = 9;
    windowDuration = 0.04;
    hopDuration = 0.01;
    %afe = feature_Extraction(fs, numCoeffs, deltaWindowLength, windowDuration, hopDuration);
    afe = feature_mfcc(fs, windowDuration, hopDuration);
    [allFeatures, normFactors]  = feature_Normalization_Factors(adsTrain, afe);
    codebook_size = 5;
    tol = 0.0001;
    [SpeakersTrainFeaturesByIdx, code] = train_codebook(adsTrain,afe,normFactors, codebook_size, tol );
    disp("--------------------------------------------------------------------------------"); 
    speakers = unique(adsTest.Labels);
    n = numel(speakers);
    for k = 1:n                     % read test sound file of each speaker
        adsTestSubset = subset(adsTest,adsTest.Labels==speakers(k));
        while hasdata(adsTestSubset)
            audioData = read(adsTestSubset);
            [v, ~] = helperFeatureExtraction(audioData,afe,normFactors); % Compute MFCC's
            v = VQ_initial(v,codebook_size, tol);
            distmin = inf;
            k1 = 0;
            for l = 1:length(code)      % each trained codebook, compute distortion
                d = disteu(v, code{l});
                %d = pdist2(v, code{l});
                %disp("d : ")
                %disp(d)
                dist = sum(min(d,[],2)) / size(d,1);
                if dist < distmin
                    distmin = dist;
                    k1 = l;
                end      
            end
            
            msg = sprintf('****Speaker %d matches with speaker %d, min : %d', k, k1, distmin);
            disp(msg); 
            %disp(speakers(k)); 
            %disp(speakers(k1)); 
            disp("__________________________________________________________________________")
        end

    end
    %disp("Train : ")
    %disp(adsTrain.Labels)
    %disp("Test : ")
    %disp(adsTest.Labels)
end


function [SpeakersTrainFeaturesByIdx, code] = train_codebook(adsTrain,afe,normFactors, codebook_size, tol )
    speakers = unique(adsTrain.Labels);
    numSpeakers = numel(speakers);
    SpeakersTrainFeaturesByIdx = cell(numSpeakers,1);
    code = cell(numSpeakers,1);
    %codebook_size = 5;
    disp("Speakers data for-lop : ")
    
    for ii = 1:numSpeakers
        speaker_name = split(char(speakers(ii)),"_");
        speaker_index = str2double(speaker_name(2));
        
        adsTrainSubset = subset(adsTrain,adsTrain.Labels==speakers(ii));
        speaker_features = [];
        while hasdata(adsTrainSubset)
            audioData = read(adsTrainSubset);
            [features, numFrames] = helperFeatureExtraction(audioData,afe,normFactors);
            speaker_features = horzcat(speaker_features,  features);
            %{ 
            if ii == 1
                figure;
                pcolor(features);
                colormap winter; 
            end 
            %}
        end
        %disp("#Speakers : ")
        %disp(ii)
        SpeakersTrainFeaturesByIdx{ii} = speaker_features; % NumOfCoefss X NumOfFrames
        code{ii} = VQ_initial(speaker_features,codebook_size, tol);
        %SpeakersTrainFeaturesByIdx{speaker_index} = speaker_features;
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


function [features,numFrames] = helperFeatureExtraction(audioData,afe,normFactors)
    % Normalize
    audioData = audioData/max(abs(audioData(:)));
    
    % Protect against NaNs
    audioData(isnan(audioData)) = 0;
    
    % Isolate speech segment
    % The dataset used in this example has one word per audioData, if more
    % than one is speech section is detected, just use the longest
    % detected.
    idx = detectSpeech(audioData,afe.SampleRate);
    if size(idx,1)>1
        [~,seg] = max(idx(:,2) - idx(:,1));
    else
        seg = 1;
    end
    audioData = audioData(idx(seg,1):idx(seg,2));
    
    % Feature extraction
    features = extract(afe,audioData);

    % Feature normalization
    if ~isempty(normFactors)
        features = (features-normFactors.Mean')./normFactors.STD';
    end
    features = features';
    
    % Cepstral mean subtraction (for channel noise)
    if ~isempty(normFactors)
        features = features - mean(features,'all');
    end
    
    numFrames = size(features,2);
end



function [allFeatures, normFactors]  = feature_Normalization_Factors(adsTrain, afe)

    tic
    featuresAll = cell(0,numel(adsTrain.Files));
    for iii = 1:numel(adsTrain.Files)
        audioData = read(adsTrain);
        featuresAll{iii} = helperFeatureExtraction(audioData,afe,[]);
    end

    allFeatures = cat(2,featuresAll{:});
    % Calculate the global mean and standard deviation of each feature
    normFactors.Mean = mean(allFeatures,2,'omitnan');
    normFactors.STD = std(allFeatures,[],2,'omitnan');
    fprintf('Feature extraction from training set complete (%0.0f seconds).',toc)
end