% Please cite this paper if you use any component of this software:
% D. Cunefare, L. Fang, R.F. Cooper, A. Dubra, J. Carroll, S. Farsiu, "Open source software for automatic detection of cone photoreceptors in adaptive optics ophthalmoscopy using convolutional neural networks," Scientific Reports, 7, 6620, 2017.
% Released under a GPL v2 license.


% David Cunefare
% 3/11/2015

function [AutMatch,ManualMatch,AutIndep,ManualIndep,Ignored] = RemoveBorderCones_ConeCNN(AutMatch,ManualMatch,AutIndep,ManualIndep,IWidth,IHeight,BorderParams)
% Function for renoving marked cones(including both marks in a matched
% pair) that are too close to the boundary of the image
%%%%%%
% Outputs :
%
% AutmaticMatch: 2 column matrix containing [X,Y] positions of the Centered
%           automatic segmentations with a match
%
% ManualMatch: 2 column matrix containing [X,Y] positions of the manual
%           segmentaions with a match
%
% AutmaticIndep: 2 column matrix containing [X,Y] positions of the Centered
%           automatic segmentations with no matches found
%
% ManualIndep: 2 column matrix containing [X,Y] positions of the manual
%           segmentaions with no matches found
%
% Ignored: 2 column matrix containing [X,Y] positions of the ignored
%           segmentaions on the borders of the image
%
%
%%%%%%
% Inputs:
%
% AutmaticMatch: 2 column matrix containing [X,Y] positions of the Centered
%           automatic segmentations with a match
%
% ManualMatch: 2 column matrix containing [X,Y] positions of the manual
%           segmentaions with a match
%
% AutmaticIndep: 2 column matrix containing [X,Y] positions of the Centered
%           automatic segmentations with no matches found
%
% ManualIndep: 2 column matrix containing [X,Y] positions of the manual
%           segmentaions with no matches found
%
% IWidth: Image width(scalar)
%
% IHeight: Image height (scalar)
%
% BorderParams: Paramater matrix

%%% For Dice

 MatchedIgnored(AutMatch(:,1)<=BorderParams.HorizontalBorder+.5) = 1;
 MatchedIgnored(ManualMatch(:,1)<=BorderParams.HorizontalBorder+.5) = 1;
 MatchedIgnored(AutMatch(:,1)>=IWidth-BorderParams.HorizontalBorder+.5) = 1;
 MatchedIgnored(ManualMatch(:,1)>=IWidth-BorderParams.HorizontalBorder+.5) = 1;
 
 MatchedIgnored(AutMatch(:,2)<=BorderParams.VerticalBorder+.5) = 1;
 MatchedIgnored(ManualMatch(:,2)<=BorderParams.VerticalBorder+.5) = 1;
 MatchedIgnored(AutMatch(:,2)>=IHeight-BorderParams.VerticalBorder+.5) = 1;
 MatchedIgnored(ManualMatch(:,2)>=IHeight-BorderParams.VerticalBorder+.5) = 1;

 % Automatic Only Cones
 AutIgnored(AutIndep(:,1)<=BorderParams.HorizontalBorder+.5) = 1;
 AutIgnored(AutIndep(:,1)>=IWidth-BorderParams.HorizontalBorder+.5) = 1;
 AutIgnored(AutIndep(:,2)<=BorderParams.VerticalBorder+.5) = 1;
 AutIgnored(AutIndep(:,2)>=IHeight-BorderParams.VerticalBorder+.5) = 1;
 
 % Manual Only Cones
 ManualIgnored(ManualIndep(:,1)<=BorderParams.HorizontalBorder+.5) = 1;
 ManualIgnored(ManualIndep(:,1)>=IWidth-BorderParams.HorizontalBorder+.5) = 1;
 ManualIgnored(ManualIndep(:,2)<=BorderParams.VerticalBorder+.5) = 1;
 ManualIgnored(ManualIndep(:,2)>=IHeight-BorderParams.VerticalBorder+.5) = 1;
 
 % Remove ignored points and consolidate
 Ignored =[AutMatch(MatchedIgnored==1,:);ManualMatch(MatchedIgnored==1,:);...
           AutIndep(AutIgnored==1,:); ManualIndep(ManualIgnored==1,:)];
       
 AutMatch(MatchedIgnored==1,:) = [];     
 ManualMatch(MatchedIgnored==1,:) = [];
 AutIndep(AutIgnored==1,:) = [];
 ManualIndep(ManualIgnored==1,:) = [];


end