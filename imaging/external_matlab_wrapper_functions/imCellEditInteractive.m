function bwOut = imCellEditInteractive(fovImg, bwCell, zoomToAxis, ...
                                       cellRadius, seedPoints)
%IMCELLEDITINTERACTIVE Add cells to a cell mask interactively
%  BWOUT = IMCELLEDITINTERACTIVE(FOVIMG, BWCELL, ZOOMTOAXIS, CELLRADIUS)
%
%   Interactively add/del cells to a mask by clicking on them.
%
%   Saves mask at each step to the UserData field of the figure: if this
%   crashes you can pull it out manually.
%   ud = get(gcf, 'UserData');
%   recoveredMask = ud.bwMask;
%
%   params:
%       fovImg: input to imFindCells
%       bwCell: bw cell mask returned by imFindCells
%       zoomToAxis: 4-element vector to be passed to AXIS, sets image zoom
%           default: []
%       cellRadius: how large of a spot to fill cells with
%           default: 1; usually you should not need to change this
%             (if you find that clicks are too sensitive to exact pixel
%             position it can help to increase this to 2)
%             set to empty to auto-compute from existing cells; this is
%             rarely helpful
%
%   Hints for using:
%      Contrast threshold (fract of mean) should be between 0.9 and 1.05,
%          usually between 0.95 and 1.0.  Higher values mean more
%          restrictive/smaller masks, and avoid 'spillover'. 
%          I usually use 0.95 for dim cells and 1.0 for bright cells.
%      Dilation disk size in pix: 1-3, depending on cell sizes.  For cell
%          radii of ~10 pix (512pix,360um FOV), I use 3
%          for 3-5 pix radii, I use 1
%
%      * If two cells are nearby, you can often avoid spillover by clicking
%        the bright cell before the dim cell; likewise you can sometimes use
%        bright stuff between two cells to segment them, then delete it
%        afterwards.
%      * If you're not getting a big enough mask for a cell whose center is
%        bright, try clicking a bit off center on the dimmer edge.

%   by Mark Histed based on code from Aaron Kerlin
%   MH 5/1/08: add delete option; ui for editing theshold and disk size
%
%$Id: imCellEditInteractive.m 517 2009-05-01 21:17:19Z histed $

   
[nRows nCols] = size(fovImg);

if nargin < 2
    bwCell =[];
end
if isempty(bwCell), bwCell = false([nRows,nCols]); end

labCell = bwlabel(bwCell,8);

%% measure existing cells
stats = regionprops(labCell, 'Area');
cellAreas = cat(1, stats.Area);
nCells = length(cellAreas);

%% process arguments
if nargin < 3 || isempty(zoomToAxis), 
    % add 0.5 margin at each edge to emulate imshow / imagesc
    zoomToAxis = [1 nCols 1 nRows] + 0.5+[-1 1 -1 1];  
end
if nargin < 4, cellRadius = 1; end

%% auto-computer the cell area dilation disk radius, only if requested
if isempty(cellRadius)
    if nCells < 5
        error('Too few cells to measure cell radius (%d): ', nCells);
    else
        cellRadius = sqrt(mean(cellAreas)/pi);
    end
end
cellRadius = round(cellRadius);

if nargin < 5
    seedPoints = []
end

% draw initial figure
figH = figure;
imH = subDrawShaded(fovImg, bwCell, zoomToAxis);
axH = get(imH, 'Parent');
set(axH, 'Position', [0.06 0.06 0.88 0.88])

%title(sprintf('\\bf%s\\rm: add objects to mask', mfilename));
% set up UI
figPos = get(figH, 'Position');
figColor = get(figH, 'Color');
set(axH, 'Units', 'normalized');
axPos = get(axH, 'Position');
utlH = uicontrol('Style', 'text', ...
                 'String', {'Contrast threshold:', ' fract of mean'}, ...
                 'Position', [5 figPos(4)-40, 80, 40], ...
                 'BackgroundColor', figColor, ...
                 'HorizontalAlignment', 'left');
utH = uicontrol('Style', 'edit', ...
                'String', '0.95', ...   % default cThresh
                'Units', 'pixels', ...
                'Position', [5 figPos(4)-70, 60, 30]);

udlH = uicontrol('Style', 'text', ...
                 'String', {'Dilation disk:', ' size in pix'}, ...
                 'Position', [5 figPos(4)-130, 80, 30], ...
                 'BackgroundColor', figColor, ...
                 'HorizontalAlignment', 'left');
udH = uicontrol('Style', 'edit', ...
                'String', '1', ...    % default diskR
                'Units', 'pixels', ...
                'Position', [5 figPos(4)-160, 60, 30]);
cdH = uicontrol('Style', 'text', ...
                'String', { 'Cell radius:', ...
                            ' (initial disk)', ...
                            sprintf(' %5.2fpix', cellRadius) }, ...
                'Units', 'pixels', ...
                'Position', [5 figPos(4)-260, 60, 60], ...
                'BackgroundColor', figColor, ...
                'HorizontalAlignment', 'left');

% title str
tStr = { [mfilename ': Click on a cell region to add'], ...
         '       Shift-click to del, RET to finish, Z to undo' };
tHeight = (1-axPos(4))-axPos(2);
tlH = uicontrol('Style', 'text', ...
                'String', tStr, ...
                'Units', 'normalized', ...
                'Position', [0.25 1-tHeight, 0.5, tHeight*0.8], ...
                'BackgroundColor', figColor, ...
                'HorizontalAlignment', 'left');


%% iterate: get a point, add it, display it
bwCurr = bwCell;
nActions = 0;
nTotal = nCells;
saveTotal = {};

%% add seed points first
if seedPoints
    oldTotal = 0;
    for i = 1:size(seedPoints)
        %% do add
        [bwNew bwCell] = subAddCell(bwCurr, fovImg, seedPoints(i,1), seedPoints(i,2), cellRadius, 0.95, 1);

        if sum(bwCell(:)) > 5 && sum(bwCell(:)) < 75
            
            nTotal = nTotal+1;
            fprintf(1, 'Added object #%d: %d pix\n', ...
                    nTotal, sum(bwCell(:)));

            %% save old mask and update
            bwSave{nActions+1} = bwCurr;  
            saveTotal{nActions+1} = oldTotal;
            nActions = nActions+1;
            bwCurr = bwNew;
        end
    end
end
subDrawShaded(fovImg, bwCurr, zoomToAxis);


while 1  % till broken out of

    % interactively get clicks
    [X Y selectionType] = getAPoint(gca);

    if isnan(X)
        key = lower(Y);
        switch key
          case char(13) % return
            break;  % out of loop, done
          case 'z' 
            % undo
            if nActions == 0
                fprintf(1, '** Cannot undo: no cells added yet\n');
                continue;
            end
            bwCurr = bwSave{nActions};
            nTotal = saveTotal{nActions};
            nActions = nActions-1;

            % draw the new version
            subDrawShaded(fovImg, bwCurr, zoomToAxis);
            fprintf(1, 'Undo!  %d cells total now\n', nTotal);
        end
        continue
    end

    %% validate XY point
    X = round(X);
    Y = round(Y);
    if X <= 0 || X >= nCols || Y <= 0 || Y >= nRows    
        % clicked outside axes, repeat 
        fprintf(1, 'Click was outside image axes, try again\n');
        continue;
    end


    %%% get ui data
    diskR = str2double(get(udH, 'String'));
    if isnan(diskR)
        fprintf(1, 'Error reading disk radius: %s, try again\n', ...
                get(udH, 'String'));
        continue
    end
    if (diskR < 1) || diskR > 50
        fprintf(1, 'Disk too small or too big, try again\n');
        continue
    elseif ~iswithintol(round(diskR), diskR, 10^2*eps)
        fprintf(1, 'Disk radius must be an integer, try again\n');
        continue
    end
    
    
    cThresh = str2double(get(utH, 'String'));    
    if isnan(cThresh)
        fprintf(1, 'Error reading threshold: %s, try again\n', ...
                get(utH, 'String'));
        continue
    end
    if cThresh <= 0 || cThresh >= 100
        fprintf(1, 'Threshold too small or too big, try again\n');
        continue
    end



    %%% what kind of mouse click?
    oldTotal = nTotal;
    switch lower(selectionType)
      case 'extend'    % shift-left: delete
        % make sure we're in a cell
        if bwCurr(Y,X) ~= 1
            fprintf(1, '** Trying to delete, not on a cell, try again\n');
         
            continue;
        end
        
        %% do delete
        bwNew = subDeleteCell(bwCurr, X, Y);
        nTotal = nTotal-1;
        fprintf(1, 'Deleted object, %d total remain\n', nTotal);
      case 'normal'    % left-click: add
        % make sure not in a cell
        if bwCurr(Y,X) == 1
            fprintf(1, '** Trying to add in existing cell region, try again\n');
            continue;
        end

        %% do add
        [bwNew bwCell] = subAddCell(bwCurr, fovImg, X, Y, ...
                                    cellRadius, cThresh, diskR);

        nTotal = nTotal+1;
        fprintf(1, 'Added object #%d: %d pix\n', ...
                nTotal, sum(bwCell(:)));
      otherwise
        % other type of click, just continue without comment
        %keyboard
        fprintf(1, 'Unrecognized click occurred (matlab bug?) %s, %g %g\n', ...
            selectionType, X, Y);
        continue
    end

    %% save old mask and update
    bwSave{nActions+1} = bwCurr;  
    saveTotal{nActions+1} = oldTotal;
    nActions = nActions+1;
    bwCurr = bwNew;
    
    % draw the new version
    subDrawShaded(fovImg, bwCurr, zoomToAxis);
end

bwOut = bwCurr;
fprintf(1, '%s: done, %d objects total).\n', ...
        mfilename, nTotal);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function imH = subDrawShaded(fovImg, bw, zoomToAxis)

shadeimg = imShade(fovImg, bw);
cla;
imH = imagesc(shadeimg);
set(gca, 'XGrid', 'on', ...
         'YGrid', 'on', ...
         'Visible', 'on', ...
         'YDir', 'reverse', ...
         'DataAspectRatio', [1 1 1]);
axis(zoomToAxis);  %[397 468 212 283]);
drawnow;

ud.fovImg = fovImg;
ud.bwMask = bw;
set(gcf, 'UserData', ud);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function bwNew = subDeleteCell(bwCurr, X, Y);

tBwCell = bwselect(bwCurr, round(X), round(Y), 4);  % only 4-connected
                                                    % objs
bwNew = bwCurr & ~tBwCell;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [bwNew bwCell] = subAddCell(bwCurr, fovImg, X, Y, ...
                                     cellRadius, cThresh, diskR);

%%% Real cell-adding logic is here
%keyboard
[nRows nCols] = size(bwCurr);

%% set up dilation disks:  use 4-connected regions for all
%se = strel('disk',1,8);  % for cells-to-avoid
se = strel('square',3);  % for cells-to-avoid: all 9 pix in a square;
                       % avoid diagonally-connected cells.
se2 = strel('disk',round(cellRadius),4);  % for region to find mean over
seJunk = strel('disk', max(round(cellRadius/4), 1), 4);  % remove thin junk
seExpand = strel('disk', diskR, 4);  % expand thresholded region


% add a disk around each point, non-overlapping with adj cells
tempmask = false(nRows, nCols);
dilateorg = imdilate(bwCurr,se);
tempmask(Y, X) = 1;
tempmask = imdilate(tempmask,se2);
tempmask = tempmask & ~dilateorg;

% fill region around disk of similar intensity, combine with disk
cellMean = mean(fovImg(tempmask == 1),1);
allMeanBw = fovImg >= cellMean.*cThresh;  % threshold by intensity
connMeanBw = bwselect(allMeanBw &~dilateorg, X, Y, 4);
connMeanBw = connMeanBw |tempmask & ~dilateorg; 

% erode then dilate filled to remove sharp things
erMean = imerode(connMeanBw, seJunk);
dilateMean = imdilate(erMean, seJunk);
dilateMean = imdilate(dilateMean, seExpand); % (thresh is conservative)
bwCell = dilateMean & ~dilateorg;

bwNew = bwCurr | bwCell;
%keyboard
