function regStack = stackRegisterJavaAG(stack,target,mode)
%STACKREGISTERJAVA (calcium): register a stack (single-threaded)
% REGSTACK = STACKREGISTERJAVA(STACK)
% REGSTACK = STACKREGISTERJAVA(STACK,TARGET)
%   Essentially a matlab wrapper around IJAlign_AK.java
%
%   histed 080505: create from RegMcoreSlave.m (aaron)
%   bonin 100107 added optional target argument
%$Id$

[nRows nCols nFrames] = size(stack);
if nargin < 3
    mode = 'translation'
end
if nargin < 2
    nAvg = min(100, floor(nFrames/15));
    avgFrNs = 1:nAvg;
    target = mean(stack(:,:,avgFrNs), 3);
end

fprintf(1, 'Moving source data to Java... ');
targetImg = ijarray2plus(target, 'single');
sourceStack = ijarray2plus(stack, 'single');
al = IJAlign_AK;
clear('stack');
fprintf(1, 'done.\n');


%% set up options
cropping = sprintf('%d %d %d %d', [5 5 nCols-6 nRows-6]);

rowOffset = round(nRows * 0.20);
colOffset = round(nCols * 0.20);

landmarks_1=num2str(round([nCols nRows nCols nRows]/2));

landmarks_2=num2str(round([nCols nRows nCols nRows nCols nRows nCols nRows]/2) + ...
                    [-colOffset 0 -colOffset 0 colOffset 0 colOffset 0]);

landmarks_3=num2str(round([nCols nRows nCols nRows nCols nRows nCols nRows nCols nRows nCols nRows]/2) + ...
                    [-colOffset -rowOffset -colOffset -rowOffset ...
                     -colOffset +rowOffset -colOffset +rowOffset ...
                     +colOffset -rowOffset +colOffset -rowOffset]);

switch mode
  case 0
    landmarks = landmarks_1;
    mode = 'translation'
  case 1
    landmarks = landmarks_2;
    mode = 'scaledRotation'
  case 2
    landmarks = landmarks_3;
    mode='rigidBody'    
  case 3
    landmarks = landmarks_3;
    mode='affine'    
  otherwise
    landmarks=landmarks_1;
    mode='translation'    
end

%landmarks = [center,' ',center,' ',center,' ',center];
%landmarks = center;
cmdstr = sprintf(['-align -window s %s -window t %s ' ...
                  '-' mode ' %s -hideOutput'], ...
                 cropping, cropping, landmarks);

% run it
tic;
fprintf(1, 'Doing alignment of %d frames... ', nFrames);
resultStack = al.doAlign(cmdstr, sourceStack, targetImg);
elT = toc;
fprintf(1, 'done in %ds, %4.2ffps\n', ...
        chop(elT,2), nFrames./elT);


%% decode output

% dummy object to disable automatic scaling in StackConverter
fprintf(1, 'Moving result data back to MATLAB... ');
% dummy = ij.process.ImageConverter(resultStack);
% dummy.setDoScaling(0); % this static property is used by StackConverter
% 
% converter = ij.process.StackConverter(resultStack); % don't use 
%                                                    %ImageConverter for stacks!
% converter.convertToGray8;

regStack = ijplus2array(resultStack, 'uint16');
fprintf(1, 'done.\n');


