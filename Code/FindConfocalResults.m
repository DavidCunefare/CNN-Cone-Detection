% Please cite this paper if you use any component of this software:
% D. Cunefare, L. Fang, R.F. Cooper, A. Dubra, J. Carroll, S. Farsiu, "Open source software for automatic detection of cone photoreceptors in adaptive optics ophthalmoscopy using convolutional neural networks," Scientific Reports, 7, 6620, 2017.
% Released under a GPL v2 license.


% Generate results of the CNN based cone detection method on the confocal
% validation set

clear
close all
clc

% Save image flag
SaveFlag = 0;

% Set Dir for saving figures
BasePath = GetRootPath();
SaveDirFig = fullfile(BasePath,'Figure Images');

% Save Position of GTDP director
GTDPDir = fullfile(BasePath,'Images and Results','Confocal','Validation GTDP Coord');




% load in parameters (choose between standard and combined)
DataSet = 'confocal';
params = get_parameters_Cone_CNN(DataSet);

% DataSet = 'confocal_combined cnn';
% params = get_parameters_Cone_CNN_Combined(DataSet);






%%%%% Parameters %%%%%
MatchParams.PointMatchVerticalUpscale = 1.0;
DistancePercent = params.Opt.DistancePercent;
BorderParams = params.Opt.BorderParams;




% load in list of images 
CNNfileList = dir(fullfile(params.Results.SaveDir,'*.mat')); 
CNNfileList =  {CNNfileList.name};

numFiles = length(CNNfileList);


% Initialize Parameters
NumMatchesGTDP = zeros(numFiles,1);
NumMissesGTDP = zeros(numFiles,1);
NumFalseAlarmsGTDP = zeros(numFiles,1);

NumMatchesCNN = zeros(numFiles,1);
NumMissesCNN = zeros(numFiles,1);
NumFalseAlarmsCNN = zeros(numFiles,1);


NumCNN= zeros(numFiles,1);
NumMan = zeros(numFiles,1);
NumGTDP = zeros(numFiles,1);


PixeltoUm = zeros(numFiles,1);
CroppedWidths = zeros(numFiles,1);
CroppedHeights = zeros(numFiles,1);



%----------------------------------------------------------------------
% Load the resolution scaleInfo
%----------------------------------------------------------------------

scaleInfoFile = fullfile(BasePath,'Images and Results','Confocal','scale_info_Confocal.csv');

fid = fopen(scaleInfoFile,'r');
scaleInfo = textscan(fid,'%s%f','Delimiter','",','MultipleDelimsAsOne',1);
fclose(fid);


% Loop through each file
for iFile = 1:numFiles
    
    %%%%% Determine CNN cone locations %%%%%%
    
    [~,BaseName] = fileparts(CNNfileList{iFile});
    
    % Load CNN Coordinates
    load(fullfile(params.Results.SaveDir,CNNfileList{iFile}))
    
   % Save Dimensions (accouting for cropping) for later
   CroppedWidths(iFile) = imageSize(2) - 2*(params.Opt.BorderParams.HorizontalBorder);
   CroppedHeights(iFile) = imageSize(1) - 2*(params.Opt.BorderParams.VerticalBorder);
   
   % Determine pixel scaling
    file = BaseName;
    endIndex = find(file == '_',2);
    endIndex = endIndex(2) - 1;
    name = file(1:endIndex);
    patientIndex = strcmpi(scaleInfo{1},name);
    if isempty(patientIndex) || sum(patientIndex) ~= 1
        error('Scale scaleInfo is not present for %s',file);
    end
    micronsPerPixel = scaleInfo{2}(patientIndex);
    
    PixeltoUm(iFile) = micronsPerPixel;
   
   
   %%%%% Compare to manual markings %%%%%%
   % Load in Manual Coordinates
    CoordPath = fullfile(params.ManualCoordDirValidate,[BaseName,params.CoordAdditionalText,params.CoordExt]);
    [x,y] = textread(CoordPath);
    ManualPos = [x,y];

    % Match CNN to manual
    NNdistance = FindAllNNdistances(ManualPos);
    MatchParams.MaxDistance = median(NNdistance)*DistancePercent;

    [AutMatchCNN,ManualMatchCNN,AutIndepCNN,ManualIndepCNN] = FindManualConeMatches(CNNPos,ManualPos,MatchParams);

    % Remove cones pairs too close to border    
    [AutMatchCNN,ManualMatchCNN,AutIndepCNN,ManualIndepCNN] = RemoveBorderCones_ConeCNN(AutMatchCNN,ManualMatchCNN,AutIndepCNN,ManualIndepCNN,imageSize(2),imageSize(1),BorderParams);
    
    % Save Results
    NumMatchesCNN(iFile) = size(AutMatchCNN,1);
    NumMissesCNN(iFile) = size(ManualIndepCNN,1);
    NumFalseAlarmsCNN(iFile) = size(AutIndepCNN,1);
    
    
    
    %%%%% Find GTDP performance as well %%%%%%
    
    % load in GTDP coordinates
    [x,y] = textread(fullfile(GTDPDir,[BaseName,'.txt']));
    GTDPPos = [x,y];
    
    % Match GTDP to manual
    [AutMatchGTDP,ManualMatchGTDP,AutIndepGTDP,ManualIndepGTDP] = FindManualConeMatches(GTDPPos,ManualPos,MatchParams);
%      
     % Remove cones pairs too close to border
     [AutMatchGTDP,ManualMatchGTDP,AutIndepGTDP,ManualIndepGTDP] = RemoveBorderCones_ConeCNN(AutMatchGTDP,ManualMatchGTDP,AutIndepGTDP,ManualIndepGTDP,imageSize(2),imageSize(1),BorderParams);
%     
    % Save Results
    NumMatchesGTDP(iFile) = size(AutMatchGTDP,1);
    NumMissesGTDP(iFile) = size(ManualIndepGTDP,1);
    NumFalseAlarmsGTDP(iFile) = size(AutIndepGTDP,1);
    
    
    % Save Number of cones to find density (only using position and not matching)
     NumCNN(iFile) = size(RemoveBorderCones_Density(CNNPos,imageSize(2),imageSize(1),BorderParams),1);
     NumGTDP(iFile) = size(RemoveBorderCones_Density(GTDPPos,imageSize(2),imageSize(1),BorderParams),1);
     NumMan(iFile) = size(RemoveBorderCones_Density(ManualPos,imageSize(2),imageSize(1),BorderParams),1);
    
    
end


% Compute statistics
FileSensCNN = NumMatchesCNN./(NumMatchesCNN+NumMissesCNN);
FileFDRCNN = NumFalseAlarmsCNN./(NumMatchesCNN+NumFalseAlarmsCNN);
FileDiceCNN = (2*NumMatchesCNN)./(2*NumMatchesCNN +NumFalseAlarmsCNN+NumMissesCNN);

disp('Sensitivity     FDR     Dice')
disp('----------------------------')
  
MeanResultsCNN = [mean(FileSensCNN) mean(FileFDRCNN) mean(FileDiceCNN)]
StdResultsCNN = [std(FileSensCNN) std(FileFDRCNN) std(FileDiceCNN)]


FileSensGTDP = NumMatchesGTDP./(NumMatchesGTDP+NumMissesGTDP);
FileFDRGTDP = NumFalseAlarmsGTDP./(NumMatchesGTDP+NumFalseAlarmsGTDP);
FileDiceGTDP = (2*NumMatchesGTDP)./(2*NumMatchesGTDP +NumFalseAlarmsGTDP+NumMissesGTDP);
  
MeanResultsGTDP = [mean(FileSensGTDP) mean(FileFDRGTDP) mean(FileDiceGTDP)]
StdResultsGTDP = [std(FileSensGTDP) std(FileFDRGTDP) std(FileDiceGTDP)]

MedianDiceCNN = median(FileDiceCNN)
MedianDiceGTDP = median(FileDiceGTDP)



% % Check for signifigance
% WilSens =signrank(FileSensCNN,FileSensGTDP);
% WilFDR = signrank(FileFDRCNN,FileFDRGTDP);
% WilDice = signrank(FileDiceCNN,FileDiceGTDP);
% 
% PVals = [WilSens WilFDR WilDice]



%%%%%% Bland-Altman Analysis
DensityCNN = NumCNN./(CroppedWidths.*CroppedHeights.*PixeltoUm.^2).*1000^2;
DensityGTDP = NumGTDP./(CroppedWidths.*CroppedHeights.*PixeltoUm.^2).*1000^2;
DensityMan = NumMan./(CroppedWidths.*CroppedHeights.*PixeltoUm.^2).*1000^2;

% CNN vs Manual
DiffManMinusCNN = (DensityMan - DensityCNN);
AveManCNN = (DensityMan + DensityCNN)/2;

MeanManCNN = mean(DiffManMinusCNN);
StdManCNN = std(DiffManMinusCNN);

X = linspace(40000,120000,400);
figure; plot(AveManCNN,DiffManMinusCNN,'k.')
hold on
plot(X,ones(1,400).*MeanManCNN,'k');
plot(X,ones(1,400).*(MeanManCNN+1.96*StdManCNN),':','LineWidth',1.75,'Color',[.5 .5 .5]);
plot(X,ones(1,400).*(MeanManCNN-1.96*StdManCNN),':','LineWidth',1.75,'Color',[.5 .5 .5]);
hold off
set(gca,'FontSize',14)
xlabel('Mean Density (cones/mm^{2})','FontSize',16)
if(strcmp(DataSet,'confocal'))
ylabel('Manual - C-CNN (cones/mm^{2})','FontSize',16)
else
ylabel('Manual - M-CNN (cones/mm^{2})','FontSize',16)  
end
axis([40000 120000 -15000 15000])
set(gca,'YTick',[-15000 -12500 -10000 -7500 -5000 -2500 0 2500 5000 7500 10000 12500 15000])

if(SaveFlag==1)
    if(strcmp(DataSet,'confocal'))
        saveas(gcf,fullfile(SaveDirFig,'BlandAFig_Confocal_C-CNN.tif'));
    else
        saveas(gcf,fullfile(SaveDirFig,'BlandAFig_Confocal_M-CNN.tif'));
    end
end




% GTDP vs Manual
DiffManMinusGTDP = (DensityMan - DensityGTDP);
AveManGTDP = (DensityMan + DensityGTDP)/2;

MeanManGTDP = mean(DiffManMinusGTDP);
StdManGTDP = std(DiffManMinusGTDP);

X = linspace(40000,120000,400);
figure; plot(AveManGTDP,DiffManMinusGTDP,'k.')
hold on
plot(X,ones(1,400).*MeanManGTDP,'k');
plot(X,ones(1,400).*(MeanManGTDP+1.96*StdManGTDP),':','LineWidth',1.75,'Color',[.5 .5 .5]);
plot(X,ones(1,400).*(MeanManGTDP-1.96*StdManGTDP),':','LineWidth',1.75,'Color',[.5 .5 .5]);
hold off
set(gca,'FontSize',14)
xlabel('Mean Density (cones/mm^{2})','FontSize',16)
ylabel('Manual - GTDP (cones/mm^{2})','FontSize',16)
axis([40000 120000 -15000 15000])
set(gca,'YTick',[-15000 -12500 -10000 -7500 -5000 -2500 0 2500 5000 7500 10000 12500 15000])


if(SaveFlag==1)
saveas(gcf,fullfile(SaveDirFig,'BlandAFig_ConfocalGTDP.tif'));
end