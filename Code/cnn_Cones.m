function [net, info] = cnn_Cones(gpus,params)
% Code for training a CNN classifier for cone and non-cone patches

% This file was modified from the MatConvNet cnn_cifar.m file by David
% Cunefare 3-24-2017, for 
% D. Cunefare, L. Fang, R.F. Cooper, A. Dubra, J. Carroll, S. Farsiu, "Open source software for automatic detection of cone photoreceptors in adaptive optics ophthalmoscopy using convolutional neural networks," Scientific Reports, 7, 6620, 2017.


% Copyright (c) 2014-16 The MatConvNet team.
% All rights reserved.
% 
% Redistribution and use in source and binary forms are permitted
% provided that the above copyright notice and this paragraph are
% duplicated in all such forms and that any documentation,
% advertising materials, and other materials related to such
% distribution and use acknowledge that the software was developed
% by the <organization>. The name of the <organization> may not be
% used to endorse or promote products derived from this software
% without specific prior written permission.  THIS SOFTWARE IS
% PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES,
% INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES OF
% MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

% Get everything set up
% run(fullfile(vl_rootnn, 'matlab', 'vl_setupnn.m')) ;

opts.modelType = params.CNNtrain.modeltype; 
opts.networkType = params.CNNtrain.networkType;
opts.train = struct() ;
opts.train.gpus = gpus;

opts.imdbPath = params.CNN.imdbPath;
opts.expDir = params.CNN.TrainExpDir;
      
 
% -------------------------------------------------------------------------
%                                                    Prepare model and data
% -------------------------------------------------------------------------

switch opts.modelType
  case 'lenet'
    net = cnn_Cones_init('networkType', opts.networkType) ;
  otherwise
    error('Unknown model type ''%s''.', opts.modelType) ;
end

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
    disp('Could not find IMDB file')
    return;
end

net.meta.classes.name = imdb.meta.classes(:)' ;


% Randomise Patches
RandIndx = randperm(size(imdb.images.labels,2));
imdb.images.labels = imdb.images.labels(:,RandIndx);
imdb.images.data = imdb.images.data(:,:,:,RandIndx);
imdb.images.set = imdb.images.set(:,RandIndx);
 
% -------------------------------------------------------------------------
%                                                                     Train
% -------------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainfn = @cnn_train_Cones ;
%   case 'dagnn', trainfn = @cnn_train_dag ;
end

[net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3)) ;

% Save the final network in new location
stats = info;
save(params.CNN.NetworkSavePath, 'net', 'stats') 

% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
%   case 'dagnn'
%     bopts = struct('numGpus', numel(opts.train.gpus)) ;
%     fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% -------------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;


% % -------------------------------------------------------------------------
% function inputs = getDagNNBatch(opts, imdb, batch)
% % -------------------------------------------------------------------------
% images = imdb.images.data(:,:,:,batch) ;
% labels = imdb.images.labels(1,batch) ;
% if opts.numGpus > 0
%   images = gpuArray(images) ;
% end
% inputs = {'input', images, 'label', labels} ;

