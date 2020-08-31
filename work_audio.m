%addpath lab_library/;
%Add the directory and its subdirectories 
%addpath(genpath('lab_library/'))
% clc % clear command window
function [adsTrain, numPar] = work_audio() 

    file_location1 = fullfile("SPEECH DATA","FEMALE","MIC");
    file_location2 = fullfile("SPEECH DATA","MALE","MIC");
    ads = audioDatastore([file_location1, file_location2], 'IncludeSubfolders',true, 'FileExtensions','.wav');
    fileNames = ads.Files;

    speakerIDs = extractBetween(fileNames,'mic_','_');  % {'F1'}, ..., {'F10'}, {'M1'}, ..., {'M10'}
    ads.Labels = categorical(speakerIDs); % return an array F1, ..., F10, M1, ..., M10
    % Count the number of utterances from each speaker  based on ads.Labels 
    %countEachLabel(ads)

    % Split the data 
    speakersToTest = categorical(["M01","M05","F01","F05"]); % Test data
    adsTrain = subset(ads,~ismember(ads.Labels,speakersToTest));
    adsEnrollAndVerify = subset(ads,ismember(ads.Labels,speakersToTest));
    adsTest = subset(ads,ismember(ads.Labels,speakersToTest));

    adsTrain = shuffle(adsTrain);
    adsEnrollAndVerify = shuffle(adsEnrollAndVerify);
    adsTest = shuffle(adsTest);
    disp("Distribution of the speakers in the Train data : ")
    countEachLabel(adsTrain)
    disp("Distribution of the speakers in the Test data : ")
    countEachLabel(adsTest)
    
    %%%%
    [audio,audioInfo] = read(adsTrain);
    fs = audioInfo.SampleRate;
    % Reset the train data 
    reset(adsTrain)
    
    % Reduce dataset 
    reduceDataset = false;
    number = 21;
    [adsTrain, adsEnrollAndVerify] = reduce_data(adsTrain, adsEnrollAndVerify, reduceDataset, number);
    
    % Features extractions
    numCoeffs = 20;
    deltaWindowLength = 9;
    windowDuration = 0.025;
    hopDuration = 0.01;
    afe = feature_Extraction(fs, numCoeffs, deltaWindowLength, windowDuration, hopDuration);
    
    % Extract features from the audio read from the training datastore
    features = extract(afe,audio);
    [numHops,numFeatures] = size(features)
    
    if ~isempty(ver('parallel')) && ~reduceDataset
        pool = gcp('nocreate');
        numPar = numpartitions(adsTrain,pool);
    else
        numPar = 1;
    end
    
    [allFeatures, normFactors]  = feature_Normalization_Factors(adsTrain, numPar, afe);
    [ubm, numComponents] = universal_Background_Model(numFeatures, afe, normFactors);
    [numSpeakers, N, F, Nc, Fc]  = baum_Welch_Statistics(adsTrain, numPar, ubm, afe, numFeatures, numComponents, normFactors);
    [T, I]  = total_Variability_Space(ubm, numSpeakers, numComponents, numFeatures, N, F, Nc);
    %[ivectorPerSpeaker, ubmMu] = i_Vector_Extraction(adsTrain, numTdim, ubm, afe, normFactors, numFeatures, T, I);
    [ivectorPerSpeaker, ubmMu, TS, TSi]  = i_Vector_Extraction(adsTrain, numTdim, ubm, afe, normFactors, numFeatures, T, I);
    performLDA = true;
    performWCCN = true ;
    [projectionMatrix]  = projection_Matrix(ivectorPerSpeaker, performLDA, performWCCN);
    % Enroll
    disp("Enroll new speakers : ")
    [enrolledSpeakers, adsVerify]  = enroll(projectionMatrix, adsEnrollAndVerify, I, T,TS, TSi, ubm, afe, numFeatures, normFactors, ubmMu);
    [speakersToTest, cssFRR]  = false_Rejection_Rate(enrolledSpeakers, adsVerify, projectionMatrix, afe, normFactors, ubm, ubmMu, numFeatures, I, TS, TSi, T);
end 

%%
function [speakersToTest, cssFRR]  = false_Rejection_Rate(enrolledSpeakers, adsVerify, projectionMatrix, afe, normFactors, ubm, ubmMu, numFeatures, I, TS, TSi, T)

    speakersToTest = unique(adsVerify.Labels);
    numSpeakers = numel(speakersToTest);
    thresholdsToTest = -1:0.0001:1;
    cssFRR = cell(numSpeakers,1);
    tic
    parfor speakerIdx = 1:numSpeakers
        adsPart = subset(adsVerify,adsVerify.Labels==speakersToTest(speakerIdx));
        numFiles = numel(adsPart.Files);

        ivectorToTest = enrolledSpeakers.(string(speakersToTest(speakerIdx))); %#ok<PFBNS> 

        css = zeros(numFiles,1);
        for fileIdx = 1:numFiles
            audioData = read(adsPart);

            % Extract features
            Y = helperFeatureExtraction(audioData,afe,normFactors);

            % Compute a posteriori log-likelihood
            logLikelihood = helperGMMLogLikelihood(Y,ubm);

            % Compute a posteriori normalized probability
            amax = max(logLikelihood,[],1);
            logLikelihoodSum = amax + log(sum(exp(logLikelihood-amax),1));
            gamma = exp(logLikelihood - logLikelihoodSum)';

            % Compute Baum-Welch statistics
            n = sum(gamma,1);
            f = Y * gamma - n.*(ubmMu);

            % Extract i-vector
            ivector = pinv(I + (TS.*repelem(n(:),numFeatures))' * T) * TSi * f(:);

            % Intersession Compensation
            ivector = projectionMatrix*ivector;

            % Cosine Similarity Score
            css(fileIdx) = dot(ivectorToTest,ivector)/(norm(ivector)*norm(ivectorToTest));
        end
        cssFRR{speakerIdx} = css;
    end
    cssFRR = cat(1,cssFRR{:});
    FRR = mean(cssFRR<thresholdsToTest);
    fprintf('FRR calculated (%0.0f seconds).\n',toc)


    figure;
    plot(thresholdsToTest,FRR)
    title('False Rejection Rate (FRR)')
    xlabel('Threshold')
    ylabel('FRR')
    grid on
    axis([thresholdsToTest(find(FRR~=0,1)) thresholdsToTest(find(FRR==1,1)) 0 1])

end
%%


%%
function [enrolledSpeakers, adsVerify]  = enroll(projectionMatrix, adsEnrollAndVerify, I, T,TS, TSi, ubm, afe, numFeatures, normFactors, ubmMu)

    numFilesPerSpeakerForEnrollment = 20;
    [adsEnroll,adsVerify] = splitEachLabel(adsEnrollAndVerify,numFilesPerSpeakerForEnrollment);
    adsVerify = shuffle(adsVerify);
    adsEnroll = shuffle(adsEnroll);
    disp("Display countEachLabel(adsEnroll) :");
    countEachLabel(adsEnroll)
    disp("Display countEachLabel(adsVerify) :");
    countEachLabel(adsVerify)
    speakers = unique(adsEnroll.Labels);
    numSpeakers = numel(speakers);
    enrolledSpeakersByIdx = cell(numSpeakers,1);
    tic
    parfor speakerIdx = 1:numSpeakers
        % Subset the datastore to the speaker you are adapting.
        adsPart = subset(adsEnroll,adsEnroll.Labels==speakers(speakerIdx));
        numFiles = numel(adsPart.Files);

        ivectorMat = zeros(size(projectionMatrix,1),numFiles);
        for fileIdx = 1:numFiles
            audioData = read(adsPart);

            % Extract features
            Y = helperFeatureExtraction(audioData,afe,normFactors);

            % Compute a posteriori log-likelihood
            logLikelihood = helperGMMLogLikelihood(Y,ubm);

            % Compute a posteriori normalized probability
            amax = max(logLikelihood,[],1);
            logLikelihoodSum = amax + log(sum(exp(logLikelihood-amax),1));
            gamma = exp(logLikelihood - logLikelihoodSum)';

            % Compute Baum-Welch statistics
            n = sum(gamma,1);
            f = Y * gamma - n.*(ubmMu);

            %i-vector Extraction
            ivector = pinv(I + (TS.*repelem(n(:),numFeatures))' * T) * TSi * f(:);

            % Intersession Compensation
            ivector = projectionMatrix*ivector;

            ivectorMat(:,fileIdx) = ivector;
        end
        % i-vector model
        enrolledSpeakersByIdx{speakerIdx} = mean(ivectorMat,2);
    end
    fprintf('Speakers enrolled (%0.0f seconds).\n',toc)

    enrolledSpeakers = struct;
    for s = 1:numSpeakers
        enrolledSpeakers.(string(speakers(s))) = enrolledSpeakersByIdx{s};
    end

end
%%

%%
function [projectionMatrix]  = projection_Matrix(ivectorPerSpeaker, performLDA, performWCCN)
% Create a matrix of the training vectors and a map indicating which i-vector corresponds to which speaker.
% Initialize the projection matrix as an identity matrix.
w = ivectorPerSpeaker;
utterancePerSpeaker = cellfun(@(x)size(x,2),w);
ivectorsTrain = cat(2,w{:});
projectionMatrix = eye(size(w{1},1));

%performLDA = true;
if performLDA
    tic
    numEigenvectors = 16;

    Sw = zeros(size(projectionMatrix,1));
    Sb = zeros(size(projectionMatrix,1));
    wbar = mean(cat(2,w{:}),2);
    for ii = 1:numel(w)
        ws = w{ii};
        wsbar = mean(ws,2);
        Sb = Sb + (wsbar - wbar)*(wsbar - wbar)';
        Sw = Sw + cov(ws',1);
    end
    
    [A,~] = eigs(Sb,Sw,numEigenvectors);
    A = (A./vecnorm(A))';

    ivectorsTrain = A * ivectorsTrain;
    
    w = mat2cell(ivectorsTrain,size(ivectorsTrain,1),utterancePerSpeaker);
    
    projectionMatrix = A * projectionMatrix;
    
    fprintf('LDA projection matrix calculated (%0.2f seconds).',toc)
end


%performWCCN = true;
if performWCCN
    tic
    alpha = 0.9;
    
    W = zeros(size(projectionMatrix,1));
    for ii = 1:numel(w)
        W = W + cov(w{ii}',1);
    end
    W = W/numel(w);
    
    W = (1 - alpha)*W + alpha*eye(size(W,1));
    
    B = chol(pinv(W),'lower');
    
    projectionMatrix = B * projectionMatrix;
    
    fprintf('WCCN projection matrix calculated (%0.4f seconds).',toc)
end


end 
%%


%%
function [ivectorPerSpeaker, ubmMu, TS, TSi]  = i_Vector_Extraction(adsTrain, numTdim, ubm, afe, normFactors, numFeatures, T, I)
    speakers = unique(adsTrain.Labels);
    numSpeakers = numel(speakers);
    ivectorPerSpeaker = cell(numSpeakers,1);
    TS = T./Sigma;
    TSi = TS';
    ubmMu = ubm.mu;
    tic
    parfor speakerIdx = 1:numSpeakers

        % Subset the datastore to the speaker you are adapting.
        adsPart = subset(adsTrain,adsTrain.Labels==speakers(speakerIdx));
        numFiles = numel(adsPart.Files);

        ivectorPerFile = zeros(numTdim,numFiles);
        for fileIdx = 1:numFiles
            audioData = read(adsPart);

            % Extract features
            Y = helperFeatureExtraction(audioData,afe,normFactors);

            % Compute a posteriori log-likelihood
            logLikelihood = helperGMMLogLikelihood(Y,ubm);

            % Compute a posteriori normalized probability
            amax = max(logLikelihood,[],1);
            logLikelihoodSum = amax + log(sum(exp(logLikelihood-amax),1));
            gamma = exp(logLikelihood - logLikelihoodSum)';

            % Compute Baum-Welch statistics
            n = sum(gamma,1);
            f = Y * gamma - n.*(ubmMu);

            ivectorPerFile(:,fileIdx) = pinv(I + (TS.*repelem(n(:),numFeatures))' * T) * TSi * f(:);
        end
        ivectorPerSpeaker{speakerIdx} = ivectorPerFile;
    end
    fprintf('I-vectors extracted from training set (%0.0f seconds).\n',toc)
end
%%



%%
function [T, I]  = total_Variability_Space(ubm, numSpeakers, numComponents, numFeatures, N, F, Nc)
    % Create the sigma variable.
    Sigma = ubm.sigma(:);
    % pecify the dimension of the total variability space. A typical value used for the TIMIT data set is 1000.
    numTdim = 256;
    % Initialize T and the identity matrix, and preallocate cell arrays.
    T = randn(numel(ubm.sigma),numTdim);
    T = T/norm(T);
    I = eye(numTdim);

    Ey = cell(numSpeakers,1);
    Eyy = cell(numSpeakers,1);
    Linv = cell(numSpeakers,1);
    % Set the number of iterations for training. A typical value reported is 20.
    numIterations = 5;
    % Run the training loop.
    for iterIdx = 1:numIterations
        tic
        % 1. Calculate the posterior distribution of the hidden variable
        TtimesInverseSSdiag = (T./Sigma)';
        parfor s = 1:numSpeakers
            L = (I + TtimesInverseSSdiag.*N{s}*T);
            Linv{s} = pinv(L);
            Ey{s} = Linv{s}*TtimesInverseSSdiag*F{s};
            Eyy{s} = Linv{s} + Ey{s}*Ey{s}';
        end

        % 2. Accumlate statistics across the speakers
        Eymat = cat(2,Ey{:});
        FFmat = cat(2,F{:});
        Kt = FFmat*Eymat';
        K = mat2cell(Kt',numTdim,repelem(numFeatures,numComponents));

        newT = cell(numComponents,1);
        for c = 1:numComponents
            AcLocal = zeros(numTdim);
            for s = 1:numSpeakers
                AcLocal = AcLocal + Nc{s}(:,:,c)*Eyy{s};
            end
        % 3. Update the Total Variability Space
            newT{c} = (pinv(AcLocal)*K{c})';
        end
        T = cat(1,newT{:});
        fprintf('Training Total Variability Space: %d/%d complete (%0.0f seconds).\n',iterIdx,numIterations,toc)
    end

end
%%


%%
function [numSpeakers, N, F, Nc, Fc]  = baum_Welch_Statistics(adsTrain, numPar, ubm, afe, numFeatures, numComponents, normFactors)
    numSpeakers = numel(adsTrain.Files);
    Nc = {};
    Fc = {};

    tic
    parfor ii = 1:numPar
        adsPart = partition(adsTrain,numPar,ii);
        numFiles = numel(adsPart.Files);

        Npart = cell(1,numFiles);
        Fpart = cell(1,numFiles);
        for jj = 1:numFiles
            audioData = read(adsPart);

            % Extract features
            Y = helperFeatureExtraction(audioData,afe,normFactors);

            % Compute a posteriori log-likelihood
            logLikelihood = helperGMMLogLikelihood(Y,ubm);

            % Compute a posteriori normalized probability
            amax = max(logLikelihood,[],1);
            logLikelihoodSum = amax + log(sum(exp(logLikelihood-amax),1));
            gamma = exp(logLikelihood - logLikelihoodSum)';

            % Compute Baum-Welch statistics
            n = sum(gamma,1);
            f = Y * gamma;

            Npart{jj} = reshape(n,1,1,numComponents);
            Fpart{jj} = reshape(f,numFeatures,1,numComponents);
        end
        Nc = [Nc,Npart];
        Fc = [Fc,Fpart];
    end
    fprintf('Baum-Welch statistics completed (%0.0f seconds).\n',toc)
    
    N = Nc;
    F = Fc;
    muc = reshape(ubm.mu,numFeatures,1,[]);
    for s = 1:numSpeakers
        N{s} = repelem(reshape(Nc{s},1,[]),numFeatures);
        F{s} = reshape(Fc{s} - Nc{s}.*muc,[],1);
    end
    
end 

%%

%%
function [ubm, numComponents]  = universal_Background_Model(numFeatures, afe, normFactors)
    numComponents = 256; % args
    alpha = ones(1,numComponents)/numComponents;
    mu = randn(numFeatures,numComponents);
    vari = rand(numFeatures,numComponents) + eps;
    ubm = struct('ComponentProportion',alpha,'mu',mu,'sigma',vari);

    maxIter = 10;
    % Start timer
    tic
    for iter = 1:maxIter
        tic
        % EXPECTATION
        N = zeros(1,numComponents);
        F = zeros(numFeatures,numComponents);
        S = zeros(numFeatures,numComponents);
        L = 0;
        parfor ii = 1:numPar
            adsPart = partition(adsTrain,numPar,ii);
            while hasdata(adsPart)
                audioData = read(adsPart);

                % Extract features
                Y = helperFeatureExtraction(audioData,afe,normFactors);

                % Compute a posteriori log-liklihood
                logLikelihood = helperGMMLogLikelihood(Y,ubm);

                % Compute a posteriori normalized probability
                amax = max(logLikelihood,[],1);
                logLikelihoodSum = amax + log(sum(exp(logLikelihood-amax),1));
                gamma = exp(logLikelihood - logLikelihoodSum)';

                % Compute Baum-Welch statistics
                n = sum(gamma,1);
                f = Y * gamma;
                s = (Y.*Y) * gamma;

                % Update the sufficient statistics over utterances
                N = N + n;
                F = F + f;
                S = S + s;

                % Update the log-likelihood
                L = L + sum(logLikelihoodSum);
            end
        end

        % Print current log-likelihood
        % Stor timer
        fprintf('Training UBM: %d/%d complete (%0.0f seconds), Log-likelihood = %0.0f\n',iter,maxIter,toc,L)

        % MAXIMIZATION
        N = max(N,eps);
        ubm.ComponentProportion = max(N/sum(N),eps);
        ubm.ComponentProportion = ubm.ComponentProportion/sum(ubm.ComponentProportion);
        ubm.mu = F./N;
        ubm.sigma = max(S./N - ubm.mu.^2,eps);
    end

end

%%


%%
function [allFeatures, normFactors]  = feature_Normalization_Factors(adsTrain, numPar, afe)
    featuresAll = {};
    tic
    parfor ii = 1:numPar
        adsPart = partition(adsTrain,numPar,ii);
        featuresPart = cell(0,numel(adsPart.Files));
        for iii = 1:numel(adsPart.Files)
            audioData = read(adsPart);
            featuresPart{iii} = helperFeatureExtraction(audioData,afe,[]);
        end
        featuresAll = [featuresAll,featuresPart];
    end
    allFeatures = cat(2,featuresAll{:});
    % Calculate the global mean and standard deviation of each feature
    normFactors.Mean = mean(allFeatures,2,'omitnan');
    normFactors.STD = std(allFeatures,[],2,'omitnan');
    fprintf('Feature extraction from training set complete (%0.0f seconds).',toc)
end

%%

function [adsTrain, adsEnrollAndVerify] = reduce_data(adsTrain, adsEnrollAndVerify, validity, number)
    % reduce data 
    % reduceDataset = false;
    if validity
        adsTrain = splitEachLabel(adsTrain, number);
        adsEnrollAndVerify = splitEachLabel(adsEnrollAndVerify, number);
    end
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
    % Input:
    % audioData   - column vector of audio data
    % afe         - audioFeatureExtractor object
    % normFactors - mean and standard deviation of the features used for normalization. 
    %               If normFactors is empty, no normalization is applied.
    % Output
    % features    - matrix of features extracted
    % numFrames   - number of frames (feature vectors) returned
    % Normalize
    audioData = audioData/max(abs(audioData(:)));
    % Protect against NaNs
    audioData(isnan(audioData)) = 0;
    % Isolate speech segment
    idx = detectSpeech(audioData,afe.SampleRate);
    features = [];
    for ii = 1:size(idx,1)
        f = extract(afe,audioData(idx(ii,1):idx(ii,2)));
        features = [features;f]; %#ok<AGROW> 
    end
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

function L = helperGMMLogLikelihood(x,gmm)
    xMinusMu = repmat(x,1,1,numel(gmm.ComponentProportion)) - permute(gmm.mu,[1,3,2]);
    permuteSigma = permute(gmm.sigma,[1,3,2]);
    Lunweighted = -0.5*(sum(log(permuteSigma),1) + sum(xMinusMu.*(xMinusMu./permuteSigma),1) + size(gmm.mu,1)*log(2*pi));
    temp = squeeze(permute(Lunweighted,[1,3,2]));
    if size(temp,1)==1
        % If there is only one frame, the trailing singleton dimension was
        % removed in the permute. This accounts for that edge case.
        temp = temp';
    end
    
    L = temp + log(gmm.ComponentProportion)';
end





