function [out,inRange,outRange] = imScale(varargin)
%IMSCALE Rescale image pixel values as desired
% OUT = IMSCALE(IMG) Rescales image IMG with default parameters.
% OUT = IMSCALE(IMG, INRANGE) Specifies input range INRANGE. 
%       Default is [min(img(:)),max(img(:)]
% OUT = IMSCALE(IMG, INRANGE, OUTRANGE) Specifies output range OUTRANGE. 
%       Default is [0,1] for floats and [intmin,intmax] for integers
% OUT = IMSCALE(..., TYPE) Returns OUT as TYPE. Default is 'double'
%
% IMG should be size (nRows, nCols, nFrames, nPlanes)
%   where nPlanes == 1 or 3
%
% If TYPE not specified then OUT is cast to a double.
%
% If INRANGE not specified, compute it, lumping over all frames, but
%   separately for each color plane
%
% 12/5/08 MH update docs
% 18/3/08 vb added TYPE argument
%         vb removed floor (did not work for singles and doubles)
%         MH autocompute inRange, support 4d/RGB images recursively
%
%$Id: imScale.m 610 2010-05-04 14:29:42Z vincent $
%

[reg,prop]=parseparams(varargin);
nreg = length(reg);nprop = length(prop);
if nprop==1; type = prop{1};else;type= 'double';end;

img = varargin{1};
if nreg > 1; inRange = varargin{2};else inRange = [];end;
if nreg > 2; outRange = varargin{3};else outRange = [];end;
 
if isempty(outRange) % default outRange, default inRange done later
    dummy = zeros(1,1,type);
    if isinteger(dummy)
        outRange = double([intmin(type),intmax(type)]);
    else
        outRange = [0,1];
    end
end

% sanity check
if ~isnumeric(inRange) || ~isnumeric(outRange)
    error('Arguments INRANGE and OUTRANGE must be numeric');
end

[nRows nCols nFrames nPlanes] = size(img);

% if an RGB image, call this sep for each plane
if nPlanes > 1
    assert(nPlanes == 3);
    out = zeros([nRows nCols nFrames nPlanes], type);
    for iP = 1:nPlanes
        out(:,:,:,iP) = imScale(squeeze(img(:,:,:,iP)), ...
                                   [], outRange, type);
    end
    return
end

% do inRange after we know what the image will be (one plane)
if isempty(inRange) || ischar(inRange) 
    inRange = double([ min(img(:)) max(img(:)) ]); 
    if range(inRange) == 0
        % this can happen with one plane of a multi-plane image; so don't
        % complain, just return
        out = img;
        return
    end
end

inRange = double(inRange);
outRange = double(outRange);
scaleFact = (outRange(2)-outRange(1)) / (inRange(2)-inRange(1));
out = ( (double(img) - inRange(1)) * scaleFact + outRange(1) );
out(out > outRange(2)) = outRange(2);
out(out < outRange(1)) = outRange(1);

% use cast and turn warning off here, instead of using floor, because it
% is likely faster
ws = warning('off', 'MATLAB:intConvertNonIntVal');
out = cast(out,type);
warning(ws);

return;

%% test 
img= randn(128,128);
figure;
out = imScale(img);
imshow(out);colorbar % should range between 0 and 1
title(class(out))
figure;
out = imScale(img,'uint16');
imshow(out);colorbar % should range between 0 and 2^16-1
title(class(out))
figure;
out = imScale(img,[0,3]);
imshow(out);colorbar % should be darker and range between 0 and 1
title(class(out))
figure;
out = imScale(img,[0,3],'uint16');
imshow(out);colorbar % should be darker and range between 0 and 2^16-1
title(class(out))
figure;
out = imScale(img,[0,3],[0,10]);
imagesc(out);colorbar;% should be darker and range between 0 and 10
colormap gray; axis square;
title(class(out))
figure;
out = imScale(img,[0,3],[0,10],'uint16');
imagesc(out);colorbar;% should be darker and range between 0 and 10
colormap gray; axis square;
title(class(out))

