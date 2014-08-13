function [global_context, local_context] = rcnn_context(im, ...
    boxes, context_model)

% PARAMETER OF THE NETWORK
% TODO: move these parameters into rcnn_model
% NOTE: if you change any of these parameters, you must also change the
% corresponding network prototext file
context_stride = 32;
max_proposal_num = 2500;
% 5 Scale
% fixed_sizes = [640, 768, 917, 1152, 1600]';
% fixed_context_scale = 2; % middle scale (0-indexed)
% 1 Scale
fixed_sizes = [917]';
fixed_context_scale = 0; % middle scale (0-indexed)

[global_context, local_context, response_maps] = rcnn_context_forward(im, ...
    boxes, context_model, fixed_sizes, context_stride, fixed_context_scale, ...
    max_proposal_num);

end

function [global_context, local_context, response_maps] = ...
    rcnn_context_forward(im, boxes, context_model, fixed_sizes, ...
    context_stride, fixed_context_scale, max_proposal_num)
% extract image-level global contextual feature and window-level local
% contextual features
%
%   im is an image in RGB order as returned by imread
%   boxes are in [x1 y1 x2 y2] format with one box per row
%   rcnn_model specifies the CNN Caffe net file to use.

% get the channel (BGR) mean
channel_mean = context_model.cnn.channel_mean;

% input size is the size of image used for network input
input_size = max(fixed_sizes);
scale_num = size(fixed_sizes, 1);
proposal_num = size(boxes, 1);

% calculate zooming factor
[image_h, image_w, ~] = size(im);
image_l = max(image_h, image_w);
zoom_factors = fixed_sizes / image_l;
fixed_hs = round(min(image_h * zoom_factors, fixed_sizes));
fixed_ws = round(min(image_w * zoom_factors, fixed_sizes));

% fixed the context scale to be the pre-defined scale
whole_scales = zeros(scale_num, 1, 'single');
whole_scales(:) = (1:scale_num) - 1;
context_scales = zeros(proposal_num, 1, 'single');
context_scales(:) = fixed_context_scale;

% calculate the multiscale image data and context windows
multiscale_image_data = zeros(input_size, input_size, 3, scale_num, 'single');
multiscale_whole_window = zeros(4, scale_num, 'single');
multiscale_context_windows = zeros(4, proposal_num, 'single');
response_maps.sizes = zeros(2, scale_num);
for scale = 1:scale_num
  % resize the image to a fixed scale, turn off antialiasing to make it
  % similar to imresize in OpenCV
  resized_im = imresize(im, [fixed_hs(scale), fixed_ws(scale)], 'bilinear', ...
      'antialiasing', false);
  % convert from RGB channels to BGR channels
  image_data = single(resized_im(:, :, [3, 2, 1]));
  % mean subtraction
  for c = 1:3
    image_data(:, :, c) = image_data(:, :, c) - channel_mean(c);
  end
  % set width to be the fastest dimension
  image_data = permute(image_data, [2, 1, 3]);
  multiscale_image_data(1:fixed_ws(scale), 1:fixed_hs(scale), :, scale) = ...
      image_data;

  % resize the image and calculate the size of feature maps
  resized_im_h = image_h * zoom_factors(scale);
  resized_im_w = image_w * zoom_factors(scale);
  pool5_h = round(resized_im_h / context_stride + 1);
  pool5_w = round(resized_im_w / context_stride + 1);
  fc8_h = pool5_h - 5;
  fc8_w = pool5_w - 5;
  response_maps.sizes(:, scale) = [fc8_h, fc8_w];
  whole_window = single([0, 0, fc8_h - 1, fc8_w - 1]);

  % resize the boxes and change it to [y1 x1 y2 x2], 0-indexed
  resized_boxes = (boxes(:, [2 1 4 3]) - 1) * zoom_factors(scale); % no adding 1
  % calculate the conv5 windows ([y1 x1 y2 x2], 0-indexed)
  pool5_windows = resized_boxes / context_stride;
  % 6x6 pool5 -> 6x6 fc8-conv, with a shift of 2.5
  context_windows = single(round(pool5_windows - 2.5));
  
  % make sure y1, x1, y2, x2 >= 0 & y1, y2 < fc8_height & x1, x3 < fc8_width
  context_windows = max(context_windows, 0);
  context_windows(:, [1 3]) = min(context_windows(:, [1 3]), fc8_h - 1);
  context_windows(:, [2 4]) = min(context_windows(:, [2 4]), fc8_w - 1);
  
  % add 1 to the ends
  context_windows(:, [3, 4]) = context_windows(:, [3, 4]) + 1;
  whole_window(:, [3, 4]) = whole_window(:, [3, 4]) + 1;
  
  % set width to be the fastest dimension
  whole_window = permute(whole_window, [2, 1]);
  multiscale_whole_window(:, scale) = whole_window;
  context_windows = permute(context_windows, [2, 1]);
  is_matched = (scale - 1 == context_scales);
  multiscale_context_windows(:, is_matched) = context_windows(:, is_matched);
end

% forward image data, context windows and context scales into caffe to get
% features
% split the windows into batches when window_num exceeds max_window_num
local_context = [];
for start_id = 1:max_proposal_num:proposal_num
  end_id = min(proposal_num, start_id + max_proposal_num - 1);
  context_windows_batch = zeros(4, max_proposal_num, 'single');
  context_windows_batch(:, 1:end_id-start_id+1) = ...
      multiscale_context_windows(:, start_id:end_id);
  context_scales_batch = zeros(max_proposal_num, 1, 'single');
  context_scales_batch(1:end_id-start_id+1) = context_scales(start_id:end_id);
  batch = {multiscale_image_data; ...
      multiscale_whole_window; whole_scales; ...
      context_windows_batch; context_scales_batch};
  caffe_output = caffe('forward', batch);
  % get features and set num to be the fastest dimension
  global_context = squeeze(caffe_output{1})';
  local_context = cat(1, local_context, squeeze(caffe_output{2})');
  % get response map and set height to be the fastest dimension
  response_maps.raw_map = permute(caffe_output{3}, [2 1 3 4]);
end

% postprocess results
local_context = local_context(1:proposal_num, :);

end
