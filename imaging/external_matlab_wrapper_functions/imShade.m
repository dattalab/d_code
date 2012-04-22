function [shadeimg] = imShade (img, mask1, mask2, mask3)
%IMSHADECELLS Shade grayscale image with mask
% [shadeimg] = imShadeCells (img, mask1, chan, alpha)
% img: input 2-D img   (N x M)
% mask1:  binary img with same dimensions, class double
% 
% based on shadecellsCR by Clay Reid 9/23/04
% 
% 19-Mar-2008 VB Removed ifnorm (not useful), ifplot (not used)
%             VB added argument chan



% always normalize
shadeimg = imScale(img, [], [0 1], 'double');

shadeimg = repmat(shadeimg, [1 1 3]);  % add 3rd dim

shadeimg(:,:,1) = shadeimg(:,:,1) .* (1-.5*double(mask1));  % delete channel

if nargin > 2
    shadeimg(:,:,2) = shadeimg(:,:,2) .* (1-.5*double(mask2));  % delete channel
end
   
if nargin > 3
    shadeimg(:,:,3) = shadeimg(:,:,3) .* (1-.5*double(mask3));  % delete channel
end



return;