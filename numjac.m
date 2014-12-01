function J = numjac(FUN,x,options,varargin)
% NUMJAC Computes finite differences Jacobian
%
% J = NUMJAC(FUN,X) computes the Jacobian J by central finite differences
% applied to function FUN (function name or function handle) that accepts a vector
% X as input and return a vector F. The Jacobian is evaluated at X.
%
% J = NUMJAC(FUN,X,options) uses the structure OPTIONS to calculate the
% Jacobian. The fields of the structure are
%  FinDiffType    : {'central'} or 'forward' for central or forward finite
%                   differences
%  FinDiffRelStep : scalar or vector step size factor. For central finite
%                   differences, the default value is FinDiffRelStep = eps^(1/3),
%                   and for forward finite differences FinDiffRelStep = sqrt(eps).
%  SerialProblem  : true or false (default), if true considers the problem as a
%                   serial problem: X is a n-by-d matrix and FUN(X) outputs a
%                   n-by-d matrix, such that each line is independent from the
%                   other. In this case, NUMJAC returns a sparse block diagonal
%                   nxd-by-nxd Jacobian.
%  Vectorized     : 'on' or {'off'} to specify whether FUN is vectorized

% Copyright (C) 2011-2014 Christophe Gouel
% Licensed under the Expat license, see LICENSE.txt

%% Initialization
if isa(FUN,'char')
  FUN = str2func(FUN);
elseif ~isa(FUN,'function_handle')
  error('FUN must be either a string or a function handle')
end

defaultopt = struct(             ...
    'FinDiffType'    , 'central',...
    'FinDiffRelStep' , []       ,...
    'SerialProblem'  , false    ,...
    'Vectorized'     , 'off');

if nargin <= 2
  options = defaultopt;
else
  warning('off','catstruct:DuplicatesFound')
  options = catstruct(defaultopt,options);
end

epsilon = options.FinDiffRelStep;

switch lower(options.FinDiffType)
  case 'central'
    central = true;
  case 'forward'
    central = false;
  otherwise
    error(['Invalid value for OPTIONS field FinDiffType: must be ' ...
           '''forward'' or ''central''']);
end

if isempty(epsilon)
  if central
    epsilon = eps^(1/3);
  else
    epsilon = sqrt(eps);
  end
end

x     = x(:);
step  = epsilon(:).*max(abs(x),1);
x1    = x+step;
if central
  x0  = x-step;
  h   = x1-x0;
else
  f0  = FUN(x,varargin{:});
  h   = x1-x;
end

%% Computation of finite difference Jacobian
if ~options.SerialProblem
  switch lower(options.Vectorized)
    case 'off'
      %% Traditional problem
      d         = length(x);
      for j=1:d
        xx      = x;
        xx(j)   = x1(j);
        f1      = FUN(xx,varargin{:});
        if central
          xx(j) = x0(j);
          f0    = FUN(xx,varargin{:});
        end
        if j==1, J = zeros(length(f0),d); end
        J(:,j)  = (f1-f0)/h(j);
      end

    case 'on'
      %% Vectorized problem
      d    = length(x);
      if central, xx = x(:,ones(2*d,1));
      else        xx = x(:,ones(d,1));
      end
      xx(sub2ind([d d],1:d,1:d))                 = x1;
      if central, xx(d^2+sub2ind([d d],1:d,1:d)) = x0; end
      ff  = FUN(xx,varargin{:});
      if central
        J = ff(:,1:d)-ff(:,(d+1):(2*d));
      else
        J = bsxfun(@minus,ff,f0);
      end
      J = bsxfun(@rdivide,J,h');

    otherwise
      error(['Invalid value for OPTIONS field Vectorized: must be ' ...
             '''on'' or ''off''']);
  end
else
  %% Serial problem
  [n,d]       = size(x);
  J           = zeros(n,d,d);
  for j=1:d
    xx        = x;
    xx(:,j)   = x1(:,j);
    f1        = FUN(xx,varargin{:});
    if central
      xx(:,j) = x0(:,j);
      f0      = FUN(xx,varargin{:});
    end
    J(:,:,j)  = (f1-f0)./h(:,j);
  end
  J = spblkdiag(permute(J,[2 3 1]));

end

