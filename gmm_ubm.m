%addpath lab_library/;
%Add the directory and its subdirectories 
%addpath(genpath('lab_library/'))
% clc % clear command window

function [ads, speaker] = gmm_ubm() 
%addpath(genpath('speech_commands_data/'));
    %clc;
    file_location = fullfile("speech_commands_data");
    ads = audioDatastore(file_location, 'IncludeSubfolders',true,'LabelSource','folderNames', 'FileExtensions','.wav');
    ads = subset(ads,ads.Labels==categorical("stop"));
    %
    [~,fileName] = cellfun(@(x)fileparts(x),ads.Files,'UniformOutput',false);
    fileName = split(fileName,'_');
    speaker = strcat('a',fileName(:,1));
    ads.Labels = categorical(speaker);
    
    numSpeakersToEnroll = 10;
    labelCount = countEachLabel(ads);
    forEnrollAndTestSet = labelCount{:,1}(labelCount{:,2}>=3);
    forEnroll = forEnrollAndTestSet(randi([1,numel(forEnrollAndTestSet)],numSpeakersToEnroll,1));
    tf = ismember(ads.Labels,forEnroll);
    adsEnrollAndValidate = subset(ads,tf);
    adsEnroll = splitEachLabel(adsEnrollAndValidate,2);

    adsTest = subset(ads,ismember(ads.Labels,forEnrollAndTestSet));
    adsTest = subset(adsTest,~ismember(adsTest.Files,adsEnroll.Files));

    forUBMTraining = ~(ismember(ads.Files,adsTest.Files) | ismember(ads.Files,adsEnroll.Files));
    adsTrainUBM = subset(ads,forUBMTraining);

    %Read from the training datastore and listen to a file. Reset the datastore.
    [audioData,audioInfo] = read(adsTrainUBM);
    fs = audioInfo.SampleRate;
    t = (0:size(audioData,1)-1)/fs;
    sound(audioData,fs)
    figure;
    plot(t,audioData)
    xlabel('Time (s)')
    ylabel('Amplitude')
    axis([0 t(end) -1 1])
    title('Sample Utterance from Training Set')
    reset(adsTrainUBM)
    
    % Features extractions
    numCoeffs = 20;
    deltaWindowLength = 9;
    windowDuration = 0.025;
    hopDuration = 0.01;
    afe = feature_Extraction(fs, numCoeffs, deltaWindowLength, windowDuration, hopDuration);
    
    % Normalize the audio.
    audioData = audioData./max(abs(audioData));
    idx = detectSpeech(audioData,fs);
    audioData = audioData(idx(1,1):idx(1,2));
    features = extract(afe,audioData);
    [numHops,numFeatures] = size(features)
    
    % Extract all Features 
    [allFeatures, normFactors, numPar] = global_Feature_Normalization_Factors(ads, afe);
    [ubm, numComponents]  = universal_Background_Model(adsTrainUBM, numFeatures, numPar, afe, normFactors);
    [enrolledGMMs, ~, ~]  = enroll(adsEnroll, numFeatures, numComponents, ubm, afe,normFactors);
    [llr, numSpeakers, speakers]  = speaker_False_Rejection_Rate(enrolledGMMs, adsEnroll, adsTest, afe, normFactors, ubm);
    
end 


%%
function [llr, numSpeakers, speakers]  = speaker_False_Rejection_Rate(enrolledGMMs, adsEnroll, adsTest, afe, normFactors, ubm)
    speakers = unique(adsEnroll.Labels);
    numSpeakers = numel(speakers);
    llr = cell(numSpeakers,1);
    tic
    parfor speakerIdx = 1:numSpeakers
        localGMM = enrolledGMMs.(string(speakers(speakerIdx))); 
        adsTestSubset = subset(adsTest,adsTest.Labels==speakers(speakerIdx));
        llrPerSpeaker = zeros(numel(adsTestSubset.Files),1);
        for fileIdx = 1:numel(adsTestSubset.Files)
            audioData = read(adsTestSubset);
            [x,numFrames] = helperFeatureExtraction(audioData,afe,normFactors);

            logLikelihood = helperGMMLogLikelihood(x,localGMM);
            Lspeaker = helperLogSumExp(logLikelihood);

            logLikelihood = helperGMMLogLikelihood(x,ubm);
            Lubm = helperLogSumExp(logLikelihood);

            llrPerSpeaker(fileIdx) = mean(movmedian(Lspeaker - Lubm,3));
        end
        llr{speakerIdx} = llrPerSpeaker;
    end
    fprintf('False rejection rate computed in %0.2f seconds.\n',toc)

    %Plot the false rejection rate as a function of the threshold.
    llr = cat(1,llr{:});
    thresholds = -0.5:0.01:2.5;
    FRR = mean(llr<thresholds);
    figure;
    plot(thresholds,FRR*100)
    title('False Rejection Rate (FRR)')
    xlabel('Threshold')
    ylabel('Incorrectly Rejected (%)')
    grid on
    

end
%%


%%
function [enrolledGMMs, numSpeakers, speakers]  = enroll(adsEnroll, numFeatures, numComponents, ubm, afe,normFactors)
    relevanceFactor = 16;
    speakers = unique(adsEnroll.Labels);
    numSpeakers = numel(speakers);
    gmmCellArray = cell(numSpeakers,1);
    
    tic
    parfor ii = 1:numSpeakers
        % Subset the datastore to the speaker you are adapting.
        adsTrainSubset = subset(adsEnroll,adsEnroll.Labels==speakers(ii));
        N = zeros(1,numComponents);
        F = zeros(numFeatures,numComponents);
        S = zeros(numFeatures,numComponents);
        while hasdata(adsTrainSubset)
            audioData = read(adsTrainSubset);
            features = helperFeatureExtraction(audioData,afe,normFactors);
            [n,f,s,l] = helperExpectation(features,ubm);
            N = N + n;
            F = F + f;
            S = S + s;
        end

        % Determine the maximum likelihood
        gmm = helperMaximization(N,F,S);
        
        % Determine adaption coefficient
        alpha = N ./ (N + relevanceFactor);
        
        % Adapt the means
        gmm.mu = alpha.*gmm.mu + (1-alpha).*ubm.mu;

        % Adapt the variances
        gmm.sigma = alpha.*(S./N) + (1-alpha).*(ubm.sigma + ubm.mu.^2) - gmm.mu.^2;
        gmm.sigma = max(gmm.sigma,eps);

        % Adapt the weights
        gmm.ComponentProportion = alpha.*(N/sum(N)) + (1-alpha).*ubm.ComponentProportion;
        gmm.ComponentProportion = gmm.ComponentProportion./sum(gmm.ComponentProportion);

        gmmCellArray{ii} = gmm;
    end
    fprintf('Enrollment completed in %0.2f seconds.\n',toc)
    %For bookkeeping purposes, convert the cell array of GMMs to a struct, 
    % with the fields being the speaker IDs and the values being the GMM structs.
    for i = 1:numel(gmmCellArray)
        enrolledGMMs.(string(speakers(i))) = gmmCellArray{i};
    end

end
%%

%%
function [ubm, numComponents]  = universal_Background_Model(adsTrainUBM, numFeatures, numPar, afe, normFactors)
    % Initialize GMM
    numComponents =32;
    alpha = ones(1,numComponents)/numComponents;

    mu = randn(numFeatures,numComponents);
    sigma = rand(numFeatures,numComponents);
    ubm = struct('ComponentProportion',alpha,'mu',mu,'sigma',sigma);

    % Train UBM Using Expectation-Maximization
    maxIter = 20;
    targetLogLikelihood = 0;
    tol = 0.005;
    pastL = -inf; % initialization of previous log-likelihood

    tic
    for iter = 1:maxIter

        % EXPECTATION
        N = zeros(1,numComponents);
        F = zeros(numFeatures,numComponents);
        S = zeros(numFeatures,numComponents);
        L = 0;
        for ii = 1:numPar
            adsPart = partition(adsTrainUBM,numPar,ii);
            while hasdata(adsPart)
                audioData = read(adsPart);

                % Extract features
                features = helperFeatureExtraction(audioData,afe,normFactors);

                % Compute a posteriori log-likelihood
                logLikelihood = helperGMMLogLikelihood(features,ubm);

                % Compute a posteriori normalized probability
                logLikelihoodSum = helperLogSumExp(logLikelihood);
                gamma = exp(logLikelihood - logLikelihoodSum)';

                % Compute Baum-Welch statistics
                n = sum(gamma,1);
                f = features * gamma;
                s = (features.*features) * gamma;

                % Update the sufficient statistics over utterances
                N = N + n;
                F = F + f;
                S = S + s;

                % Update the log-likelihood
                L = L + sum(logLikelihoodSum);
            end
        end

        % Print current log-likelihood and stop if it meets criteria.
        L = L/numel(adsTrainUBM.Files);
        fprintf('\tIteration %d, Log-likelihood = %0.3f\n',iter,L)
        if L > targetLogLikelihood || abs(pastL - L) < tol
            break
        else
            pastL = L;
        end

        % MAXIMIZATION
        N = max(N,eps);
        ubm.ComponentProportion = max(N/sum(N),eps);
        ubm.ComponentProportion = ubm.ComponentProportion/sum(ubm.ComponentProportion);
        ubm.mu = bsxfun(@rdivide,F,N);
        ubm.sigma = max(bsxfun(@rdivide,S,N) - ubm.mu.^2,eps);
    end
    
    fprintf('UBM training completed in %0.2f seconds.\n',toc)

end
%%



%%
function [allFeatures, normFactors, numPar] = global_Feature_Normalization_Factors(ads, afe) 

    featuresAll = {};
    if ~isempty(ver('parallel'))
        pool = gcp;
        numPar = numpartitions(ads,pool);
    else
        numPar = 1;
    end

    parfor ii = 1:numPar
        adsPart = partition(ads,numPar,ii);
        featuresPart = cell(0,numel(adsPart.Files));
        for iii = 1:numel(adsPart.Files)
            audioData = read(adsPart);
            featuresPart{iii} = helperFeatureExtraction(audioData,afe,[]);
        end
        featuresAll = [featuresAll,featuresPart];
    end
    allFeatures = cat(2,featuresAll{:});

    normFactors.Mean = mean(allFeatures,2,'omitnan');
    normFactors.STD = std(allFeatures,[],2,'omitnan');
end

%%


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


%%%%%%%%%5

function y = helperLogSumExp(x)
    % Calculate the log-sum-exponent while avoiding overflow
    a = max(x,[],1);
    y = a + sum(exp(bsxfun(@minus,x,a)),1);
end

%Expectation
function [N,F,S,L] = helperExpectation(features,gmm)
    post = helperGMMLogLikelihood(features,gmm);
    % Sum the likelihood over the frames
    L = helperLogSumExp(post);
    % Compute the sufficient statistics
    gamma = exp(post-L)';

    N = sum(gamma,1);
    F = features * gamma;
    S = (features.*features) * gamma;
    L = sum(L);
end

%Maximization

function gmm = helperMaximization(N,F,S)
    N = max(N,eps);
    gmm.ComponentProportion = max(N/sum(N),eps);
    gmm.mu = bsxfun(@rdivide,F,N);
    gmm.sigma = max(bsxfun(@rdivide,S,N) - gmm.mu.^2,eps);
end

%Gaussian Multi-Component Mixture Log-Likelihood

function L = helperGMMLogLikelihood(x,gmm)
    xMinusMu = repmat(x,1,1,numel(gmm.ComponentProportion)) - permute(gmm.mu,[1,3,2]);
    permuteSigma = permute(gmm.sigma,[1,3,2]);
    
    Lunweighted = -0.5*(sum(log(permuteSigma),1) + sum(bsxfun(@times,xMinusMu,(bsxfun(@rdivide,xMinusMu,permuteSigma))),1) + size(gmm.mu,1)*log(2*pi));

    temp = squeeze(permute(Lunweighted,[1,3,2]));
    if size(temp,1)==1
        % If there is only one frame, the trailing singleton dimension was
        % removed in the permute. This accounts for that edge case
        temp = temp';
    end
    L = bsxfun(@plus,temp,log(gmm.ComponentProportion)');
end

function [ads, speaker] = back_up() 
%addpath(genpath('speech_commands_data/'));
    %clc;
    file_location = fullfile("speech_commands_data");
    ads = audioDatastore(file_location, 'IncludeSubfolders',true,'LabelSource','folderNames', 'FileExtensions','.wav');
    ads = subset(ads,ads.Labels==categorical("stop"));
    %
    [~,fileName] = cellfun(@(x)fileparts(x),ads.Files,'UniformOutput',false);
    fileName = split(fileName,'_');
    speaker = strcat('a',fileName(:,1));
    ads.Labels = categorical(speaker);
    
    numSpeakersToEnroll = 10;
    labelCount = countEachLabel(ads);
    forEnrollAndTestSet = labelCount{:,1}(labelCount{:,2}>=3);
    forEnroll = forEnrollAndTestSet(randi([1,numel(forEnrollAndTestSet)],numSpeakersToEnroll,1));
    tf = ismember(ads.Labels,forEnroll);
    adsEnrollAndValidate = subset(ads,tf);
    adsEnroll = splitEachLabel(adsEnrollAndValidate,2);

    adsTest = subset(ads,ismember(ads.Labels,forEnrollAndTestSet));
    adsTest = subset(adsTest,~ismember(adsTest.Files,adsEnroll.Files));

    forUBMTraining = ~(ismember(ads.Files,adsTest.Files) | ismember(ads.Files,adsEnroll.Files));
    adsTrainUBM = subset(ads,forUBMTraining);

    %Read from the training datastore and listen to a file. Reset the datastore.
    [audioData,audioInfo] = read(adsTrainUBM);
    fs = audioInfo.SampleRate;
    t = (0:size(audioData,1)-1)/fs;
    sound(audioData,fs)
    figure;
    plot(t,audioData)
    xlabel('Time (s)')
    ylabel('Amplitude')
    axis([0 t(end) -1 1])
    title('Sample Utterance from Training Set')
    reset(adsTrainUBM)
    
    
    % Features extractions
    numCoeffs = 20;
    deltaWindowLength = 9;
    windowDuration = 0.025;
    hopDuration = 0.01;
    afe = feature_Extraction(fs, numCoeffs, deltaWindowLength, windowDuration, hopDuration);
    
    % Normalize the audio.
    audioData = audioData./max(abs(audioData));
    %figure;
    %detectSpeech(audioData,fs);
    
    %Call detectSpeech again. This time, return the indices of the speech
    %region and use them to remove nonspeech regions from the audio clip.
    idx = detectSpeech(audioData,fs);
    disp("idx : ")
    disp(idx)
    audioData = audioData(idx(1,1):idx(1,2));

    %Call extract on the audioFeatureExtractor object to extract features from audio data. 
    %The size output from extract is numHops-by-numFeatures.
    features = extract(afe,audioData);
    [numHops,numFeatures] = size(features)
    
end 