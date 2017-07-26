% Please cite this paper if you use any component of this software:
% D. Cunefare, L. Fang, R.F. Cooper, A. Dubra, J. Carroll, S. Farsiu, "Open source software for automatic detection of cone photoreceptors in adaptive optics ophthalmoscopy using convolutional neural networks," Scientific Reports, 7, 6620, 2017.
% Released under a GPL v2 license.


function SaveProbabilityMaps(params,TrainFlag,ValidateFlag)
% Function to save probability Maps from CNN to allow for quicker
% optimization of detection parameters
% Set TrainFlag/ValidateFlag to 0 to not generate probability maps


if nargin<2
    TrainFlag = 1;
end
if nargin<3
    ValidateFlag = 1;
end


%%%%%% Initialize Parameters 

% Get half patch size
HalfPatchSize = ceil((params.PatchSize-1)./2);


% load in the Net
load(params.ProbMap.NetworkPath)

net = vl_simplenn_move(net, 'gpu');
net.layers{end}.type = 'softmax';



if(TrainFlag)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Save Training Maps %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Make the Folder for saving files
if(~exist(params.ProbMap.SaveDirTrain,'dir'))
mkdir(params.ProbMap.SaveDirTrain);
end


% load in list of images
ImageList = dir(fullfile( params.ImageDirTrain,['*' params.ImageExt])); 
ImageList =  {ImageList.name};

numFiles = length(ImageList);


% Loop through all images in training set
for iFile = 1:numFiles
  
    % Load Image
    Image = imread(fullfile(params.ImageDirTrain,ImageList{iFile}));

    % Perform preprocessing
    Image = normalizeValues(Image,0,255);
    
    % Padimage    
    PadImage = padarray(Image,[HalfPatchSize HalfPatchSize],'symmetric');
    
    % Get patches
    [test_patches] = im2patches(PadImage,[params.PatchSize  params.PatchSize],params.ProbMap.PatchDistance);
    
    % Resize patches to be same as used for network
    test_patches = single(reshape(test_patches, size(test_patches,1),size(test_patches,2),1,size(test_patches,3)));

    % Use CNN to find probability for each patch 
    NumPatches = size(test_patches,4);
    Test_Probability = [];
    for Iter_num = 1:params.ProbMap.batchsize:NumPatches
        batchStart = Iter_num ;
        batchEnd = min(Iter_num+params.ProbMap.batchsize-1,NumPatches) ;
        batch = batchStart : 1 : batchEnd ;
        res_temp = vl_simplenn(net, gpuArray(single(test_patches(:,:,:,batch))),[],[],'mode','test');
        Prob_temp = squeeze(gather(res_temp(end).x)) ;
        Prob_temp(3:end,:) = [];
        Test_Probability = [Test_Probability Prob_temp ];
    end
    
    % Get Probability of being a cone
    Cone_Probability = Test_Probability(1,:);
    Cone_Probability = reshape(Cone_Probability',size(Image));
    
    
    % Save Map
    [~,BaseName] = fileparts(ImageList{iFile});
    SaveName = [BaseName '.mat'];
    save(fullfile(params.ProbMap.SaveDirTrain,SaveName),'Cone_Probability');
    
    disp(['Training File: ' num2str(iFile) '/' num2str(numFiles)])
end
end



if(ValidateFlag)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Save Validation Maps %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Make the Folder for saving files
if(~exist(params.ProbMap.SaveDirValidate,'dir'))
mkdir(params.ProbMap.SaveDirValidate);
end


% load in list of images
ImageList = dir(fullfile( params.ImageDirValidate,['*' params.ImageExt])); 
ImageList =  {ImageList.name};

numFiles = length(ImageList);


% Loop through all images in training set
for iFile = 1:numFiles
  
    % Load Image
    Image = imread(fullfile(params.ImageDirValidate,ImageList{iFile}));

    % Perform preprocessing
    Image = normalizeValues(Image,0,255);
    
    % Padimage    
    PadImage = padarray(Image,[HalfPatchSize HalfPatchSize],'symmetric');
    
    % Get patches
    [test_patches] = im2patches(PadImage,[params.PatchSize  params.PatchSize],params.ProbMap.PatchDistance);
    
    % Resize patches to be same as used for network
    test_patches = single(reshape(test_patches, size(test_patches,1),size(test_patches,2),1,size(test_patches,3)));

    % Use CNN to find probability for each patch 
    NumPatches = size(test_patches,4);
    Test_Probability = [];
    for Iter_num = 1:params.ProbMap.batchsize:NumPatches
        batchStart = Iter_num ;
        batchEnd = min(Iter_num+params.ProbMap.batchsize-1,NumPatches) ;
        batch = batchStart : 1 : batchEnd ;
        res_temp = vl_simplenn(net, gpuArray(single(test_patches(:,:,:,batch))),[],[],'mode','test');
        Prob_temp = squeeze(gather(res_temp(end).x)) ;
        Prob_temp(3:end,:) = [];
        Test_Probability = [Test_Probability Prob_temp ];
    end
    
    % Get Probability of being a cone
    Cone_Probability = Test_Probability(1,:);
    Cone_Probability = reshape(Cone_Probability',size(Image));
    
    
    % Save Map
    [~,BaseName] = fileparts(ImageList{iFile});
    SaveName = [BaseName '.mat'];
    save(fullfile(params.ProbMap.SaveDirValidate,SaveName),'Cone_Probability');
    
    disp(['Validation File: ' num2str(iFile) '/' num2str(numFiles)])
end
end