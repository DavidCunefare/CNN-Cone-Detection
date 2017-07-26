% Please cite this paper if you use any component of this software:
% D. Cunefare, L. Fang, R.F. Cooper, A. Dubra, J. Carroll, S. Farsiu, "Open source software for automatic detection of cone photoreceptors in adaptive optics ophthalmoscopy using convolutional neural networks," Scientific Reports, 7, 6620, 2017.
% Released under a GPL v2 license.


function SaveValidationCones(params)
% function for finding and saving cone locations from the validation data
% set

% Make the folder for saving data
if(~exist(params.Results.SaveDir,'dir'))
mkdir(params.Results.SaveDir);
end

% Set detection parameters based on optimization
load(params.Results.OptimizationPath)

ProbParam.PMsigma = OptParam.MaxSigma;
ProbParam.PMthresh = OptParam.MaxPMthresh;
ProbParam.ExtMaxH = OptParam.MaxExtMaxH;


% load in list of images
ImageList = dir(fullfile( params.ImageDirValidate,['*' params.ImageExt])); 
ImageList =  {ImageList.name};

numFiles = length(ImageList);


% Loop through all images in validation set
for iFile = 1:numFiles
    % Load probability map
    [~,BaseName] = fileparts(ImageList{iFile});
    ProbPath = fullfile(params.Results.ProbMapDirValidate,[BaseName '.mat']);
    load(ProbPath)
    
    % Determine cone locations
    [CNNPos] = ProbabilityMap_ConeLocations(Cone_Probability,ProbParam);
    
    % Save Data
    imageSize = size(Cone_Probability);
    SaveName = [BaseName '.mat'];
    save(fullfile(params.Results.SaveDir,SaveName),'CNNPos','imageSize');
end