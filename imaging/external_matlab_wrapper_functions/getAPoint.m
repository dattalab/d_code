function [x,y,selectionType] = getAPoint(varargin)
%GETAPOINT (ca-tools): retrieve a single point from an axes
%
%   [x y selectionType] = getAPoint(axH);
%
%   Waits for a click OR a keystroke.  If a key is pressed,
%   [NaN key] is returned (i.e. x = NaN)
%   selectionType tells you the type of click (shift-, right-, etc)
%
%   Doesn't use waitforbuttonpress because it is buggy (crashes MATLAB if
%   you hit Ctrl-C/break)
%   Leaves figure props (including UserData) same on return
%
%   See also: GETPTS (more complicated version of this)
%$Id: getAPoint.m 332 2008-09-09 20:22:02Z aaron $

global GETAPOINT_DATA

if nargin == 0 || (nargin == 1 && ishandle(varargin{1}))
    % called from command line
    if nargin < 1, 
        axH = gca;
    else 
        axH = varargin{1};
    end

    %% set up callback    
    figH = get(axH, 'Parent');
    % save state
    state = uisuspend(figH);
    
    % set up callbacks, pointer
    set(figH, 'WindowButtonDownFcn', { @getAPoint 'ButtonDown' }, ...
             'KeyPressFcn', { @getAPoint 'Keypress' }, ...
             'Pointer', 'crosshair');

    % save userdata
    GETAPOINT_DATA.axH = axH;
    figure(figH);
    
    %% wait for click
    drawnow;
    try
        waitfor(figH, 'WindowButtonDownFcn', '');  % nulled when work done
    catch
        if ishandle(figH)  % not closed
            uirestore(state);
        end
        rethrow(lasterror);
    end
    
    %% got a click, return it  (if it was a key, x and y still set as usual)
    x = GETAPOINT_DATA.x;
    y = GETAPOINT_DATA.y;
    selectionType = GETAPOINT_DATA.selectionType;
    uirestore(state);
    return
    
elseif nargin == 3
    % called as a callback
    figH = varargin{1};
    eventdata = varargin{2};
    idStr = varargin{3};
    assert(ishandle(figH) && ischar(idStr), ...
           'bug: Invalid parameters to callback');

    
    %% read user data
    axH = GETAPOINT_DATA.axH;
    
    switch idStr
      case 'ButtonDown'
        %% get current point and return it
        tPt = get(axH, 'CurrentPoint');
        GETAPOINT_DATA.x = tPt(1,1);
        GETAPOINT_DATA.y = tPt(1,2);
        GETAPOINT_DATA.selectionType = get(figH, 'SelectionType');
      case 'Keypress'
        %% return current key
        key = get(figH, 'CurrentCharacter');
        GETAPOINT_DATA.x = NaN;
        GETAPOINT_DATA.y = key;
        GETAPOINT_DATA.selectionType = NaN;
      otherwise
        error('bug: Invalid callback string');
    end
    
    set(figH, 'WindowButtonDownFcn', '');  % this is the cue to do work
    return
end











%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

% not used now, using 'cross' pointer
function [pointerShape, pointerHotSpot] = subCreatePointer

pointerHotSpot = [8 8];
pointerShape = [ ...
            NaN NaN NaN NaN NaN NaN   1   2   1 NaN NaN NaN NaN NaN NaN NaN
            NaN NaN NaN NaN NaN NaN   1   2   1 NaN NaN NaN NaN NaN NaN NaN
            NaN NaN NaN NaN NaN NaN   1   2   1 NaN NaN NaN NaN NaN NaN NaN
            NaN NaN NaN NaN NaN NaN   1   2   1 NaN NaN NaN NaN NaN NaN NaN
            NaN NaN NaN NaN NaN NaN   1   2   1 NaN NaN NaN NaN NaN NaN NaN
            NaN NaN NaN NaN NaN NaN   1   2   1 NaN NaN NaN NaN NaN NaN NaN
              1   1   1   1   1   1   1   2   1   1   1   1   1   1   1   1
              2   2   2   2   2   2   2   2   2   2   2   2   2   2   2   2
              1   1   1   1   1   1   1   2   1   1   1   1   1   1   1   1
            NaN NaN NaN NaN NaN NaN   1   2   1 NaN NaN NaN NaN NaN NaN NaN
            NaN NaN NaN NaN NaN NaN   1   2   1 NaN NaN NaN NaN NaN NaN NaN
            NaN NaN NaN NaN NaN NaN   1   2   1 NaN NaN NaN NaN NaN NaN NaN
            NaN NaN NaN NaN NaN NaN   1   2   1 NaN NaN NaN NaN NaN NaN NaN
            NaN NaN NaN NaN NaN NaN   1   2   1 NaN NaN NaN NaN NaN NaN NaN
            NaN NaN NaN NaN NaN NaN   1   2   1 NaN NaN NaN NaN NaN NaN NaN
            NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN];

