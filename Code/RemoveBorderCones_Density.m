% Please cite this paper if you use any component of this software:
% D. Cunefare, L. Fang, R.F. Cooper, A. Dubra, J. Carroll, S. Farsiu, "Open source software for automatic detection of cone photoreceptors in adaptive optics ophthalmoscopy using convolutional neural networks," Scientific Reports, 7, 6620, 2017.
% Released under a GPL v2 license.


% David Cunefare
% 3/11/2015

function [Cones,Ignored] = RemoveBorderCones_Density(Cones,IWidth,IHeight,BorderParams)
% Function for renoving marked cones( that are too close to the boundary of
% the image without considering pairing
%%%%%%


 
 % Automatic Only Cones
 IgnoredPos(Cones(:,1)<=BorderParams.HorizontalBorder+.5) = 1;
 IgnoredPos(Cones(:,1)>=IWidth-BorderParams.HorizontalBorder+.5) = 1;
 IgnoredPos(Cones(:,2)<=BorderParams.VerticalBorder+.5) = 1;
 IgnoredPos(Cones(:,2)>=IHeight-BorderParams.VerticalBorder+.5) = 1;
 

 
 % Remove ignored points and consolidate
 Ignored =[Cones(IgnoredPos==1,:)];
       
 Cones(IgnoredPos==1,:) = [];   
 
end