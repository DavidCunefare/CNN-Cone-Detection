% Please cite this paper if you use any component of this software:
% D. Cunefare, L. Fang, R.F. Cooper, A. Dubra, J. Carroll, S. Farsiu, "Open source software for automatic detection of cone photoreceptors in adaptive optics ophthalmoscopy using convolutional neural networks," Scientific Reports, 7, 6620, 2017.
% Released under a GPL v2 license.


function SaveNewSetCones(params,NewDir,ImExtension,SaveDir)
% function to find cones using pretrained CNN on new images

% Get half patch size
HalfPatchSize = ceil((params.PatchSize-1)./2);



 % load in the Net
load(params.ProbMap.NetworkPath)

net = vl_simplenn_move(net, 'gpu');
net.layers{end}.type = 'softmax';


% Set detection parameters based on optimization
load(params.Results.OptimizationPath)

ProbParam.PMsigma = OptParam.MaxSigma;
ProbParam.PMthresh = OptParam.MaxPMthresh;
ProbParam.ExtMaxH = OptParam.MaxExtMaxH;
 
 
 
 % load in list of images
ImageList = dir(fullfile(NewDir,['*' ImExtension])); 
ImageList =  {ImageList.name};

numFiles = length(ImageList);


% Make the folder for saving data
if(~exist(SaveDir,'dir'))
mkdir(SaveDir);
end



% Loop through all images in training set
for iFile = 1:numFiles
  
    % Load Image
    Image = imread(fullfile(NewDir,ImageList{iFile}));

    % Get the cone positions; 
    [CNNPos]= GetConePosSingle(params,Image,net,ProbParam);

    
    
    % Save Data
    [~,BaseName] = fileparts(ImageList{iFile});
    imageSize = size(Image);
    SaveName = [BaseName '.mat'];
    save(fullfile(SaveDir,SaveName),'CNNPos','imageSize');

end