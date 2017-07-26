% Please cite this paper if you use any component of this software:
% D. Cunefare, L. Fang, R.F. Cooper, A. Dubra, J. Carroll, S. Farsiu, "Open source software for automatic detection of cone photoreceptors in adaptive optics ophthalmoscopy using convolutional neural networks," Scientific Reports, 7, 6620, 2017.
% Released under a GPL v2 license.


% Code for reasembling the original patch databases used for network
% training (Seperated due to file size).

BasePath = GetRootPath();

% Confocal

load(fullfile(BasePath,'IMDB components','imdb-Confocal-ConeCNN-2.mat'))

Temp = images;

load(fullfile(BasePath,'IMDB components','imdb-Confocal-ConeCNN-1.mat'))

images.labels = cat(2, images.labels,Temp.labels);
images.data = cat(4,images.data, Temp.data);
images.set = cat(2,images.set, Temp.set);

save (fullfile(BasePath,'Images and Results','Confocal','imdb-Confocal-ConeCNN.mat'),'-v7.3','images','meta')

% Combined

load(fullfile(BasePath,'IMDB components','imdb-Combined-ConeCNN-2.mat'))

Temp = images;

load(fullfile(BasePath,'IMDB components','imdb-Combined-ConeCNN-1.mat'))

images.labels = cat(2, images.labels,Temp.labels);
images.data = cat(4,images.data, Temp.data);
images.set = cat(2,images.set, Temp.set);

save (fullfile(BasePath,'Images and Results','Combined CNN Results','imdb-Combined-ConeCNN.mat'),'-v7.3','images','meta')
