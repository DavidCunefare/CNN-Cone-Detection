% Please cite this paper if you use any component of this software:
% D. Cunefare, L. Fang, R.F. Cooper, A. Dubra, J. Carroll, S. Farsiu, "Open source software for automatic detection of cone photoreceptors in adaptive optics ophthalmoscopy using convolutional neural networks," Scientific Reports, 7, 6620, 2017.
% Released under a GPL v2 license.


function [Distances] = FindAllNNdistances(Points)
% Function for finding Nearest Neighbor distances for all points in a set
%%%%%%
% Outputs :
%
% Distances: vector containing euclidian distance to the nearest neighbor
%       for each point
%
%%%%%%
% Inputs:
%
% Points:  2 column matrix containing [X,Y] positions of points in the set

[~,Temp] = knnsearch(Points,Points,'K',2);

Distances = Temp(:,2);
end

