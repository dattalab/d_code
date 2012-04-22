function bool = iswithintol(float1, float2, epsilon)
%ISWITHINTOL (ps-utils): are two floating point numbers equal?
%   BOOL = ISWITHINTOL(FLOAT1, FLOAT2, EPSILON)
%   returns 1 if arguments differ by less than epsilon
%
%   EPSILON defaults to EPS: the distance from 1.0 to the next largest
%   floating point number (see help EPS).
%
%$Id: iswithintol.m 125 2008-03-20 20:19:22Z vincent $
if nargin < 3, epsilon = eps; end

bool = ( abs(float1 - float2) < epsilon*10 );

