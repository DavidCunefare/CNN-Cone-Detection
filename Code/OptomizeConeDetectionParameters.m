% Please cite this paper if you use any component of this software:
% D. Cunefare, L. Fang, R.F. Cooper, A. Dubra, J. Carroll, S. Farsiu, "Open source software for automatic detection of cone photoreceptors in adaptive optics ophthalmoscopy using convolutional neural networks," Scientific Reports, 7, 6620, 2017.
% Released under a GPL v2 license.


function OptomizeConeDetectionParameters(params)
% function for maximizing Dice's coefficent over a set of parameters

% Changes scaling between x and y (leave at 1)
MatchParams.PointMatchVerticalUpscale = 1.0;


% set parameters to search through
Sigma = params.opt.Sigma;
PMthresh = params.opt.PMthresh;
ExtMaxH = params.opt.ExtMaxH;

NumCombinations = length(Sigma)*length(PMthresh)*length(ExtMaxH);

% load in list of images 
ImageList = dir(fullfile( params.ImageDirTrain,['*' params.ImageExt])); 
ImageList =  {ImageList.name};

numFiles = length(ImageList);

AllDice = nan(length(Sigma),length(PMthresh),length(ExtMaxH));


% Brute force search through all combinations
i = 1;
for iS = 1:length(Sigma)
for iT = 1:length(PMthresh)
for iM = 1:length(ExtMaxH)

ProbParam.PMsigma = Sigma(iS);
ProbParam.PMthresh = PMthresh(iT);
ProbParam.ExtMaxH = ExtMaxH(iM);


% Initialize match counts
NumMatchesCNN = zeros(numFiles,1);
NumMissesCNN = zeros(numFiles,1);
NumFalseAlarmsCNN = zeros(numFiles,1);



% Loop through all files and evaluate
for iFile = 1:numFiles
        
    %%%%% Determine CNN cone locations %%%%%%
    % Load probability map
    [~,BaseName] = fileparts(ImageList{iFile});
    ProbPath = fullfile(params.Opt.ProbMapDirTrain,[BaseName '.mat']);
    load(ProbPath)
    
    % Determine cone locations
    [CNNPos] = ProbabilityMap_ConeLocations(Cone_Probability,ProbParam);
    
    if(isempty(CNNPos))
        % Any values so dice goes to 0
        NumMatchesCNN(iFile) = 0;
        NumMissesCNN(iFile) = 1;
        NumFalseAlarmsCNN(iFile) = 0;
        continue
    end
    
    %%%%% Compare to manual markings %%%%%%
    
    % Load in Manual Coordinates
    CoordPath = fullfile(params.ManualCoordDirTrain,[BaseName,params.CoordAdditionalText,params.CoordExt]);
    switch params.CoordExt
        case '.csv'
            ManualPos = csvread(CoordPath);
        case '.txt'
            [x,y] = textread(CoordPath);
            ManualPos = [x,y];
        otherwise
            error('Please select a known coord extension')     
    end

    % Match CNN to manual
    NNdistance = FindAllNNdistances(ManualPos);
    MatchParams.MaxDistance = median(NNdistance)*params.Opt.DistancePercent;

 
    [AutMatchCNN,ManualMatchCNN,AutIndepCNN,ManualIndepCNN] = FindManualConeMatches(CNNPos,ManualPos,MatchParams);

    % Remove cones pairs too close to border
    [IHeight, IWidth] = size(Cone_Probability);
    
    [AutMatchCNN,~,AutIndepCNN,ManualIndepCNN] = RemoveBorderCones_ConeCNN(AutMatchCNN,ManualMatchCNN,AutIndepCNN,ManualIndepCNN,IWidth,IHeight,params.Opt.BorderParams);
    
    % Save Results
    NumMatchesCNN(iFile) = size(AutMatchCNN,1);
    NumMissesCNN(iFile) = size(ManualIndepCNN,1);
    NumFalseAlarmsCNN(iFile) = size(AutIndepCNN,1);
    
end


% Compute statistics
FileSensCNN = NumMatchesCNN./(NumMatchesCNN+NumMissesCNN);
FileFDRCNN = NumFalseAlarmsCNN./(NumMatchesCNN+NumFalseAlarmsCNN);
FileDiceCNN = (2*NumMatchesCNN)./(2*NumMatchesCNN +NumFalseAlarmsCNN+NumMissesCNN);
  
ResultsAllCNN = [mean(FileSensCNN) mean(FileFDRCNN) mean(FileDiceCNN)];


AllDice(iS,iT,iM) = mean(FileDiceCNN);

disp(['Parameter Combination: ' num2str(i) '/' num2str(NumCombinations)])

i=i+1;

end
end
end



[MaxDiceTrain, MaxIdx] = max(AllDice(:));
[a, b, c] = ind2sub(size(AllDice),MaxIdx);
OptParam.MaxSigma = Sigma(a);
OptParam.MaxPMthresh = PMthresh(b);
OptParam.MaxExtMaxH = ExtMaxH(c);

save(params.Opt.SavePath,'AllDice','Sigma','PMthresh','ExtMaxH','MaxDiceTrain','OptParam');




