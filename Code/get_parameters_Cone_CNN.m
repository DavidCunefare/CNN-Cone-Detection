% Please cite this paper if you use any component of this software:
% D. Cunefare, L. Fang, R.F. Cooper, A. Dubra, J. Carroll, S. Farsiu, "Open source software for automatic detection of cone photoreceptors in adaptive optics ophthalmoscopy using convolutional neural networks," Scientific Reports, 7, 6620, 2017.
% Released under a GPL v2 license.


function params = get_parameters_Cone_CNN(DataSet)
% function to return parameters for running cone CNN and paths to load and
% save data

%%%% General parameters
params.PatchSize = 33; % size of image patch


%%%% CNN training parameters
% Which network, 'lenet' default
params.CNNtrain.modeltype = 'lenet';

% Type of network, 'simplenn' (use for default) or 'dagnn' (not currently set up)
params.CNNtrain.networkType = 'simplenn' ;


%%%% Proabability map parameters
 % distance between patches extracted (currently only works for 1)
params.ProbMap.PatchDistance = 1;

% Number of batches to run at once, lower if running out of memory
params.ProbMap.batchsize = 2000;


%%%% Optimization parameters
params.opt.Sigma = [.1 .2 .3 .4 .5 .6 .7 .8 .9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2];
params.opt.PMthresh = [0 .1 .2 .3 .4 .5 .6 .7 .8 .9 1];
params.opt.ExtMaxH = [0 .05 .1 .15 .2 .25 .3 .4 .5];

% Percent of median manually marked distance between cones to search for
% matches
params.Opt.DistancePercent = .75;

% Number of pixels to remove from the sides of images when matching
params.Opt.BorderParams.HorizontalBorder = 7; 
params.Opt.BorderParams.VerticalBorder = 7;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Data Set Specific Parameters %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get Base FolderName
BasePath = GetRootPath();

% Set parameters based on data set
switch lower(DataSet)
    case 'split detector'
        
        %%%% General parameters
        % location of images and coordinate files
        params.ImageDirTrain = fullfile(BasePath,'Images and Results','Split Detector','Training Images');
        params.ManualCoordDirTrain = fullfile(BasePath,'Images and Results','Split Detector','Training Manual Coord');
        
        params.ImageDirValidate = fullfile(BasePath,'Images and Results','Split Detector','Validation Images');
        params.ManualCoordDirValidate = fullfile(BasePath,'Images and Results','Split Detector','Validation Manual Coord');
        
        % Extension on Images
        params.ImageExt ='.tif';
        
        % Text to add on to base of image names for coord files
        params.CoordAdditionalText = '_coords';
        
        % Format of coord file
        params.CoordExt = '.csv';
        
                
        %%%%% Parameters for imdb 
        % Set path to save imdb
        SaveName = 'imdb-SplitDetector-ConeCNN.mat';
        params.imdb.SavePath = fullfile(BasePath,'Images and Results','Split Detector',SaveName);
        
        
        %%%%% CNN training parameters
        % Set path to load imdb
        params.CNN.imdbPath =  params.imdb.SavePath;
        
        % Set path to save network training steps
        SaveNameNetTrain = 'CNN Training-Split Detector';
        params.CNN.TrainExpDir = fullfile(BasePath,'Images and Results','Split Detector',SaveNameNetTrain);
        
        % Set path to save final network
        SaveNameFinalNet = 'net-epoch-45-SplitDetector-ConeCNN.mat';
        params.CNN.NetworkSavePath = fullfile(BasePath,'Images and Results','Split Detector',SaveNameFinalNet);
        
        
        %%%% Proabability map parameters
        % Set path to load network
        params.ProbMap.NetworkPath =  params.CNN.NetworkSavePath;
        
        % Set paths to save probability maps
        params.ProbMap.SaveDirTrain  = fullfile(BasePath,'Images and Results','Split Detector','Probability Maps','Training');
        params.ProbMap.SaveDirValidate  = fullfile(BasePath,'Images and Results','Split Detector','Probability Maps','Validation');
        
        
        %%%% Optimization parameters
        % Set path to load training probability maps
        params.Opt.ProbMapDirTrain = params.ProbMap.SaveDirTrain;
        
        % Set path to save the optimization results
        SaveNameOpt = 'DetectionOptimization-SplitDetector-ConeCNN.mat';
        params.Opt.SavePath = fullfile(BasePath,'Images and Results','Split Detector',SaveNameOpt);
        
        
        %%%%% Validation result parameters
        % Set path for loading optimization results
        params.Results.OptimizationPath = params.Opt.SavePath;
        
        % Set path for saving detected cones from validation set
        SaveNameVal = 'Validation CNN Coord';
        params.Results.SaveDir = fullfile(BasePath,'Images and Results','Split Detector',SaveNameVal);
        
        % Set path to load validation probability maps
        params.Results.ProbMapDirValidate = params.ProbMap.SaveDirValidate;
        
        
    case 'confocal'
        
        %%%% General parameters
        % location of images and coordinate files
        params.ImageDirTrain = fullfile(BasePath,'Images and Results','Confocal','Training Images');
        params.ManualCoordDirTrain = fullfile(BasePath,'Images and Results','Confocal','Training Manual Coord');
        
        params.ImageDirValidate = fullfile(BasePath,'Images and Results','Confocal','Validation Images');
        params.ManualCoordDirValidate = fullfile(BasePath,'Images and Results','Confocal','Validation Manual Coord');
        
        % Extension on Images
        params.ImageExt ='.tif';
        
        % Text to add on to base of image names for coord files
        params.CoordAdditionalText = '_manualcoord';
        
        % Format of coord file
        params.CoordExt = '.txt';
        
        
        %%%%% Parameters for imdb 
        % Set path to save imdb
        SaveName = 'imdb-Confocal-ConeCNN.mat';
        params.imdb.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveName);
        
        
        %%%%% CNN training parameters
        % Set path to load imdb
        params.CNN.imdbPath =  params.imdb.SavePath;
        
        % Set path to save network training steps
        SaveNameNetTrain = 'CNN Training-Confocal';
        params.CNN.TrainExpDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameNetTrain);
        
        % Set path to save final network
        SaveNameFinalNet = 'net-epoch-45-Confocal-ConeCNN.mat';
        params.CNN.NetworkSavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameFinalNet);
        
        
        %%%% Proabability map parameters
        % Set path to load network
        params.ProbMap.NetworkPath =  params.CNN.NetworkSavePath;
        
        % Set paths to save probability maps
        params.ProbMap.SaveDirTrain  = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Training');
        params.ProbMap.SaveDirValidate  = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Validation');
        
        
        %%%% Optimization parameters
        % Set path to load training probability maps
        params.Opt.ProbMapDirTrain = params.ProbMap.SaveDirTrain;
        
        % Set path to save the optimization results
        SaveNameOpt = 'DetectionOptimization-Confocal-ConeCNN.mat';
        params.Opt.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameOpt);
        
        
        %%%%% Validation result parameters
        % Set path for loading optimization results
        params.Results.OptimizationPath = params.Opt.SavePath;
        
        % Set path for saving detected cones from validation set
        SaveNameVal = 'Validation CNN Coord';
        params.Results.SaveDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameVal);
        
        % Set path to load validation probability maps
        params.Results.ProbMapDirValidate = params.ProbMap.SaveDirValidate;
        
    otherwise
        error('Please select a known data set or add your own case')      
end


end