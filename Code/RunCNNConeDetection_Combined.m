% Please cite this paper if you use any component of this software:
% D. Cunefare, L. Fang, R.F. Cooper, A. Dubra, J. Carroll, S. Farsiu, "Open source software for automatic detection of cone photoreceptors in adaptive optics ophthalmoscopy using convolutional neural networks," Scientific Reports, 7, 6620, 2017.
% Released under a GPL v2 license.


% code to run CNN based cone detection from beginning to end

% make sure matconvnet is installed and setup (see
% http://www.vlfeat.org/matconvnet/quick/)

% Make sure matconvnet is in the search path (can be added manually or by
% running vl_setupnn.m in MatConvNet\matconvnet-1.0-beta23\matconvnet-1.0-beta23\matlab)

clear
close all


% Set-up MatConVNetPaths
BasePath = GetRootPath();
MatConvNetPath = fullfile(BasePath,'matconvnet-1.0-beta23');
run(fullfile(MatConvNetPath,'matlab','vl_setupnn.m'))


% Choose Data Set ('split detector' and 'confocal' use data sets in Cunefare 
% et. al 2017) to use a different data set add new class and modify called
% functions


DataSet = 'confocal_combined cnn';


% load in parameters
 params = get_parameters_Cone_CNN_Combined(DataSet);


%% Create image patch database
% Note: Combined the seperate confocal and split detector databases used in
% paper

BasePath = GetRootPath();
ConfocalIMDBPath = fullfile(BasePath,'Images and Results','Confocal','imdb-Confocal-ConeCNN.mat');
SplitIMDBPath = fullfile(BasePath,'Images and Results','Split Detector','imdb-SplitDetector-ConeCNN.mat');

% load confocal imdb
load(ConfocalIMDBPath)

temp = images;

% combine with split detector and save
load(SplitIMDBPath)

images.labels = cat(2, images.labels, temp.labels);
images.data = cat(4, images.data, temp.data);
images.set = cat(2, images.set, temp.set);

save (params.CNN.imdbPath,'-v7.3','images','meta')



%% train network
% number of gpus to use
gpus = 1; 
 
cnn_Cones(gpus,params);

%% Save Probability Maps for training and validation data sets
TrainFlag = 1; % Save training data probability maps
ValidateFlag = 1; % Save Validation data probability maps (necesary for SaveValidationCones.m)

SaveProbabilityMaps(params,TrainFlag,ValidateFlag)

%% Find best combination of detection parameters
OptomizeConeDetectionParameters(params)




%% Run split detector

clear
close all

% Choose Data Set ('split detector' and 'confocal' use data sets in Cunefare 
% et. al 2017) to use a different data set add new class and modify called
% functions

DataSet = 'split detector_combined cnn';



% load in parameters
 params = get_parameters_Cone_CNN_Combined(DataSet);


%% Save Probability Maps for training and validation data sets
TrainFlag = 1; % Save training data probability maps
ValidateFlag = 1; % Save Validation data probability maps (necesary for SaveValidationCones.m)

SaveProbabilityMaps(params,TrainFlag,ValidateFlag)

%% Find best combination of detection parameters
OptomizeConeDetectionParameters(params)




%% Get best parameters from combined sets
% load the split detector optimization results
DataSet = 'split detector_combined cnn';
params = get_parameters_Cone_CNN_Combined(DataSet);
load(params.Opt.SavePath)

SplitDice = AllDice;

% load the confocal  optimization results
DataSet = 'confocal_combined cnn';
params = get_parameters_Cone_CNN_Combined(DataSet);
load(params.Opt.SavePath)

ConfocalDice = AllDice;


% Determine the (weighted) average Dice's coefficients between the two sets
AllDice = ((SplitDice.*184) + (ConfocalDice.*200))./(384);

[MaxDiceTrain, MaxIdx] = max(AllDice(:));
[a, b, c] = ind2sub(size(AllDice),MaxIdx);
OptParam.MaxSigma = Sigma(a);
OptParam.MaxPMthresh = PMthresh(b);
OptParam.MaxExtMaxH = ExtMaxH(c);

save(params.Results.OptimizationPath,'AllDice','Sigma','PMthresh','ExtMaxH','MaxDiceTrain','OptParam');


%% Find results on test data sets
DataSet = 'confocal_combined cnn';
params = get_parameters_Cone_CNN_Combined(DataSet);

SaveValidationCones(params)

DataSet = 'split detector_combined cnn';
params = get_parameters_Cone_CNN_Combined(DataSet);

SaveValidationCones(params)
