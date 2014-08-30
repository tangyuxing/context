function d = rcnn_load_cached_context_features(cache_name, imdb_name, id)
% d = rcnn_load_cached_context_features(cache_name, imdb_name, id)
%   loads cached context features from:
%   feat_cache/[cache_name]/[imdb_name]/[id]_context.mat

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------


file = sprintf('./feat_cache/%s/%s/%s', cache_name, imdb_name, id);

if exist([file '_context.mat'], 'file')
  d = load([file '_context.mat']);
  % concatenate image-level global context with window-level local context
  global_context = d.global_context;
  local_context = d.local_context;
  global_context = repmat(global_context, size(local_context, 1), 1);
  d.context = cat(2, global_context, local_context);
else
  warning('could not load: %s', file);
  d = create_empty_context();
end

% standardize boxes to double (for overlap calculations, etc.)
d.boxes = double(d.boxes);


% ------------------------------------------------------------------------
function d = create_empty_context()
% ------------------------------------------------------------------------
d.gt = logical([]);
d.overlap = single([]);
d.boxes = single([]);
d.context = single([]);
d.class = uint8([]);
