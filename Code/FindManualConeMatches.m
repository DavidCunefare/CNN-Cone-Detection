% Please cite this paper if you use any component of this software:
% D. Cunefare, L. Fang, R.F. Cooper, A. Dubra, J. Carroll, S. Farsiu, "Open source software for automatic detection of cone photoreceptors in adaptive optics ophthalmoscopy using convolutional neural networks," Scientific Reports, 7, 6620, 2017.
% Released under a GPL v2 license.


% David Cunefare
% 2/24/2015

function   [AutmaticMatch, ManualMatch, AutmaticUnMatch, ManualUnMatch]=FindManualConeMatches(AutmaticCenteredCoord,ManualCoord,Params)
% Function for Finding matches between automatic and manually marked cone
% positions
%%%%%%
% Outputs :
%
% AutmaticMatch: 2 column matrix containing [X,Y] positions of the Centered
%           automatic segmentations with a match
%
% ManualMatch: 2 column matrix containing [X,Y] positions of the manual
%           segmentaions with a match
%
% AutmaticUnMatch: 2 column matrix containing [X,Y] positions of the Centered
%           automatic segmentations with no matches found
%
% ManualUnMatch: 2 column matrix containing [X,Y] positions of the manual
%           segmentaions with no matches found
%
%
%%%%%%
% Inputs:
% AutmaticCenteredCoord: 2 column matrix containing [X,Y] positions of the Centered
%           automatic segmentations to be matched
%
%
% ManualCoord:  2 column matrix containing [X,Y] positions of the
%           manual segmentations to be matched
%
% Params: Paramater matrix

NumManual = size(ManualCoord,1);
NumAutomatic = size(AutmaticCenteredCoord,1);

% Upscale the vertical direction to prioritize horizontal matching
ManualPos = [Params.PointMatchVerticalUpscale.*ManualCoord(:,2) ManualCoord(:,1)];
AutomaticPos = [Params.PointMatchVerticalUpscale.*AutmaticCenteredCoord(:,2) AutmaticCenteredCoord(:,1)];

CorrespondingIndices = zeros(NumManual,1);
CorrespondingDistances = zeros(NumManual,1);

AutmaticUnMatchIndices = zeros(NumAutomatic,1);
% Find the closest automatic point for each manual
for iManual = 1:NumManual
    CurrentManual = ManualPos(iManual,:);  
    [k,d] = dsearchn(AutomaticPos,CurrentManual);
    CorrespondingIndices(iManual) = k;
    CorrespondingDistances(iManual) =d; 
end
% If Nan distance set the correpondance to nan as well
CorrespondingIndices(isnan(CorrespondingDistances)) = nan;
% If distance greater than the maximum distance set the correpondance to nan 
CorrespondingIndices(CorrespondingDistances>Params.MaxDistance) = nan;

% Make sure there is a 1 to 1 correspondance between points
for iAutomatic = 1:NumAutomatic
    OverlappingMatchIndices = find(CorrespondingIndices == iAutomatic);
    [~, iMinDis] = min(CorrespondingDistances(OverlappingMatchIndices));
    
    CorrespondingIndices(OverlappingMatchIndices) = nan;
    CorrespondingIndices(OverlappingMatchIndices(iMinDis)) = iAutomatic;
    
    % Record Indices where an automatic segmentation had no match
    if(isempty(OverlappingMatchIndices))
        AutmaticUnMatchIndices(iAutomatic) =1;
    end
end

ManualMatch = ManualCoord(~isnan(CorrespondingIndices),:);
AutmaticMatch = AutmaticCenteredCoord(CorrespondingIndices(~isnan(CorrespondingIndices)),:);

AutmaticUnMatch = AutmaticCenteredCoord(AutmaticUnMatchIndices==1,:);
ManualUnMatch = ManualCoord(isnan(CorrespondingIndices),:);
end