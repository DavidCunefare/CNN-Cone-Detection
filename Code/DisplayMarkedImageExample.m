% Please cite this paper if you use any component of this software:
% D. Cunefare, L. Fang, R.F. Cooper, A. Dubra, J. Carroll, S. Farsiu, "Open source software for automatic detection of cone photoreceptors in adaptive optics ophthalmoscopy using convolutional neural networks," Scientific Reports, 7, 6620, 2017.
% Released under a GPL v2 license.


% Example code for Getting cone position using trained network and
% displaying it


% Set-up MatConVNetPaths
BasePath = GetRootPath();
MatConvNetPath = fullfile(BasePath,'matconvnet-1.0-beta23');
run(fullfile(MatConvNetPath,'matlab','vl_setupnn.m'))



% choose dataset with already trained cnn and detection parameters
DataSet = 'split detector';
% load in parameters
 params = get_parameters_Cone_CNN(DataSet);
 
 
 
% Load in an image
ImageList = dir(fullfile( params.ImageDirValidate,['*' params.ImageExt])); 
ImageList =  {ImageList.name};

iFile = 32;
Image = imread(fullfile(params.ImageDirValidate,ImageList{iFile}));





% Get cone positions
[CNNPos]=GetConePosSingle(params,Image);


% display marked image
figure; imagesc(Image); colormap gray; axis image; axis off
hold on
scatter(CNNPos(:,1),CNNPos(:,2),40,'*','g','LineWidth',1.5);
hold off
