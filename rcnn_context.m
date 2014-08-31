function [global_context, local_context] = rcnn_context(im, boxes, ...
    context_model)
% [global_context, local_context] = rcnn_context(im, boxes, context_model)
%   Compute contextual features on a set of boxes.
%
%   im is an image in RGB order as returned by imread
%   boxes are in [x1 y1 x2 y2] format with one box per row
%   rcnn_model specifies the CNN Caffe net file to use.

% scales
context_scales = [1 2 3];
context_scale_num = length(context_scales);

% concatenate all boxes together
% multiscale_boxes(1:end-1, :) are local boxes
% multiscale_boxes(end, :) is global box
multiscale_boxes = [];

% get multiscale boxes
box_num = size(boxes, 1);
box_centers = (boxes(:, [1 2 1 2]) + boxes(:, [3 4 3 4])) / 2;
for s = 1:context_scale_num
  scaled_boxes = (boxes - box_centers) * context_scales(s) + box_centers;
  multiscale_boxes = cat(1, multiscale_boxes, scaled_boxes);
end

% get the box of the whole image
[h, w, ~] = size(im);
box_whole_image = [1 1 w h];
multiscale_boxes = cat(1, multiscale_boxes, box_whole_image);

% make sure the boxes fit into the image region
multiscale_boxes(:, 1) = max(1, multiscale_boxes(:, 1));
multiscale_boxes(:, 2) = max(1, multiscale_boxes(:, 2));
multiscale_boxes(:, 3) = min(w, multiscale_boxes(:, 3));
multiscale_boxes(:, 4) = min(h, multiscale_boxes(:, 4));

% use spp_features to extract context features
raw_context = spp_features(im, multiscale_boxes, context_model);

% get global context feature for the whole image
global_context = raw_context(end, :);

% get local context feature for each box
raw_local_context = raw_context(1:end-1, :);
assert(size(raw_local_context, 1) == context_scale_num * box_num);
local_context = [];
for s = 1:context_scale_num
  ind = ((s-1)*box_num+1):(s*box_num);
  local_context = cat(2, local_context, raw_local_context(ind, :));
end
