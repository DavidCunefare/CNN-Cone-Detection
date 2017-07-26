% Please cite this paper if you use any component of this software:
% D. Cunefare, L. Fang, R.F. Cooper, A. Dubra, J. Carroll, S. Farsiu, "Open source software for automatic detection of cone photoreceptors in adaptive optics ophthalmoscopy using convolutional neural networks," Scientific Reports, 7, 6620, 2017.
% Released under a GPL v2 license.


% Code to Find cone positions in a new set of images using a pretrained
% network and parameters


% Set-up MatConVNetPaths
BasePath = GetRootPath();
MatConvNetPath = fullfile(BasePath,'matconvnet-1.0-beta23');
run(fullfile(MatConvNetPath,'matlab','vl_setupnn.m'))

% choose dataset with already trained cnn and detection parameters
DataSet = 'split detector';
% load in parameters
 params = get_parameters_Cone_CNN(DataSet);

 
 % Choose Folder of images to detect cones in
ImageDir =  'D:\Testing\Images';

% format of images (must be readable by imread, must be 2D/grayscale format)
ImExtension = '.tif';

% Choose Folder to save coordinate
SaveDir = 'D:\Testing\Coord';



 
 SaveNewSetCones(params,ImageDir,ImExtension,SaveDir)