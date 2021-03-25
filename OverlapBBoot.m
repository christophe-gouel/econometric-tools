function Out = OverlapBBoot(Y,l)
% Overlapping Block Bootstrap procedure for a vector time series

%  File modified from Enrique M. Quilis Macroeconomic Research Department
%  Fiscal Authority for Fiscal Responsibility (AIReF)
% (https://fr.mathworks.com/matlabcentral/fileexchange/53701-bootstrapping-time-series)

% Input
% Y is the data to sample from
% l is the block length (optional)

% Output
% Out is the bootstrapped sample

% Dimension of the matrix to be bootstrapped
[T,dimy] = size(Y);

% Number of blocks
if nargin < 2 || isempty(l)
  l = round(T^(1/3)); % Criterion for block length
end
k = fix(T/l);

% ------------------------------------------------------------------------------
% INDEX SELECTION
% ------------------------------------------------------------------------------
I =  round(1+(T-l)*rand(1,k));

% ------------------------------------------------------------------------------
% BOOTSTRAP REPLICATION
% ------------------------------------------------------------------------------
Out = NaN(k*l,dimy);
for i=1:k
  Out(1+(i-1)*l:i*l,:) = Y(I(i):I(i)+l-1,:);
end
