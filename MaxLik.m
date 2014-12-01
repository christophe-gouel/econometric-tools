function [params,ML,vcov,g,H,exitflag,output] = MaxLik(loglikfun,params,obs,options,varargin)
% MAXLIK Maximizes a log-likelihood function
%
% PARAMS = MAXLIK(LOGLIKFUN,PARAMS,OBS)
%
% PARAMS = MAXLIK(LOGLIKFUN,PARAMS,OBS,OPTIONS) maximizes the log-likelihood
% function with the parameters defined by the structure OPTIONS. The fields of
% the structure are
%   ActiveParams          :
%   cov                   : method to calculate the covariance matrix of the
%                           parameters, 1 for inverse hessian, 2 for the
%                           cross-product of the first-order derivatives, and 3
%                           (default) for a covariance matrix based on the
%                           hessian and the first-order derivatives.
%   numhessianoptions     : structure of options to be passed to the function
%                           numhessian that can be used to calculate the hessian
%                           for the covariance matrix of the parameters.
%   numjacoptions         : structure of options to be passed to the function
%                           numjac that can be used to calculate the jacobian
%                           for the covariance matrix of the parameters.
%   ParamsTransform       :
%   ParamsTransformInv    :
%   ParamsTransformInvDer :
%   solver                : 'fmincon', 'fminunc' (default), 'fminsearch', or
%                           'patternsearch'
%   solveroptions         : options to be passed to the solver maximizing the
%                           likelihood.
%
% PARAMS = MAXLIK(LOGLIKFUN,PARAMS,OBS,OPTIONS,VARARGIN) provides additional
% arguments for LOGLIKFUN, which, in this case, takes the following form:
% LOGLIKFUN(PARAMS,OBS,VARARGIN).
%
% [PARAMS,ML] = MAXLIK(LOGLIKFUN,PARAMS,OBS,...) returns the normalized
% log-likelihood at the solution: \sum_{i=1}^n log f(params,obs_i)/n.
%
% [PARAMS,ML,VCOV] = MAXLIK(LOGLIKFUN,PARAMS,OBS,...)
%
% [PARAMS,ML,VCOV,G] = MAXLIK(LOGLIKFUN,PARAMS,OBS,...) returns the gradient
% with respect to the parameters of the normalized log-likelihood at the
% solution.
%
% [PARAMS,ML,VCOV,G,H] = MAXLIK(LOGLIKFUN,PARAMS,OBS,...) returns the hessian
% with respect to the parameters of the normalized log-likelihood at the
% solution.
%
% [PARAMS,ML,VCOV,G,H,EXITFLAG] = MAXLIK(LOGLIKFUN,PARAMS,OBS,...) returns the
% exitflag from the optimization solver.
%
% [PARAMS,ML,VCOV,G,H,EXITFLAG,OUTPUT] = MAXLIK(LOGLIKFUN,PARAMS,OBS,...)
% returns a structure OUTPUT from the optimization solver that contains
% information about the optimization.
%
% See also FMINSEARCH, FMINUNC, NUMHESSIAN, NUMJAC.

% Copyright (C) 2014 Christophe Gouel
% Licensed under the Expat license

%% Initialization

nparams = size(params,1);

defaultopt = struct('ActiveParams'         , [],...
                    'bounds'               , struct('lb',-inf(nparams,1),...
                                                    'ub', inf(nparams,1)),...
                    'cov'                  , 3,...
                    'numhessianoptions'    , struct(),...
                    'numjacoptions'        , struct(),...
                    'ParamsTransform'      , @(P) P,...
                    'ParamsTransformInv'   , @(P) P,...
                    'ParamsTransformInvDer', @(P) ones(size(P)),...
                    'solver'               , 'fminunc',...
                    'solveroptions'        , struct(),...
                    'TestParamsTransform'  , true);
if nargin < 4 || isempty(options)
  options = defaultopt;
else
  warning('off','catstruct:DuplicatesFound')
  if isfield(options,'bounds')
    options.bounds = catstruct(defaultopt.bounds,options.bounds);
  end
  if isfield(options,'numhessianoptions')
    options.numhessianoptions = catstruct(defaultopt.numhessianoptions,...
                                          options.numhessianoptions);
  end
  if isfield(options,'numjacoptions')
    options.numjacoptions = catstruct(defaultopt.numjacoptions,...
                                      options.numjacoptions);
  end
  options = catstruct(defaultopt,options);
end
ActiveParams          = options.ActiveParams;
cov                   = options.cov;
ParamsTransform       = options.ParamsTransform;
ParamsTransformInv    = options.ParamsTransformInv;
ParamsTransformInvDer = options.ParamsTransformInvDer;
solver                = options.solver;
solveroptions         = options.solveroptions;

validateattributes(loglikfun,{'char','function_handle'},{},1)
validateattributes(params,{'numeric'},{'2d'},2)

nobs = size(obs,1);

if options.TestParamsTransform && ...
      norm(ParamsTransformInv(ParamsTransform(params(:,1)))-params(:,1))>=sqrt(eps)
  error('Functions to transform parameters are not inverse of each other')
end

if options.TestParamsTransform && ...
      norm(diag(numjac(@(P) ParamsTransformInv(P),ParamsTransform(params(:,1))))...
           -ParamsTransformInvDer(ParamsTransform(params(:,1))))>=1E-6
  error(['The function to differentiate transformed parameters does not correspond ' ...
         'to its finite difference gradient.'])
end

if isa(loglikfun,'char'), loglikfun = str2func(loglikfun); end

if isempty(ActiveParams)
  ActiveParams = true(nparams);
else
  validateattributes(ActiveParams,{'logical','numeric'},{'vector','numel',nparams})
  ActiveParams = ActiveParams(:)~=zeros(nparams,1);
end

%% Functions and matrices to extract active parameters for the estimation
SelectParamsMat = zeros(nparams,sum(ActiveParams));
ind             = 1:nparams;
SelectParamsMat(sub2ind(size(SelectParamsMat),ind(ActiveParams),1:sum(ActiveParams))) = 1;
FixedParams     = ParamsTransform(params(:,1)).*(~ActiveParams);
SelectParams    = @(P) FixedParams(:,ones(size(P,2),1))+SelectParamsMat*P;

lb = -inf(nparams,1);
ub = +inf(nparams,1);
lbtrans = min(ParamsTransform(options.bounds.lb),ParamsTransform(options.bounds.ub));
ubtrans = max(ParamsTransform(options.bounds.lb),ParamsTransform(options.bounds.ub));
for i=1:nparams
  if any(isfinite([options.bounds.lb(i) options.bounds.ub(i)]))
    lb(i) = lbtrans(i);
    ub(i) = ubtrans(i);
  end
end

%% Maximization of the log-likelihood
Objective = @(P) -sum(loglikfun(ParamsTransformInv(SelectParams(P)),obs,varargin{:}),1)/nobs;
params    = SelectParamsMat'*ParamsTransform(params);

try
  switch lower(solver)
    case {'fmincon','fminunc','fminsearch','patternsearch'}
      %% MATLAB solvers
      problem = struct('objective', Objective,...
                       'x0'       , params,...
                       'solver'   , solver,...
                       'lb'       , lb(ActiveParams),...
                       'ub'       , ub(ActiveParams),...
                       'options'  , solveroptions);
      [PARAMS,ML,exitflag,output] = feval(solver,problem);

    case 'pswarm'
      %% PSwarm
      problem = struct('Variables'  , sum(ActiveParams),...
                       'ObjFunction', Objective,...
                       'LB'         , lb(ActiveParams),...
                       'UB'         , ub(ActiveParams));
      for it=1:size(params,2), InitialPopulation(it).x = params(:,it); end
      [PARAMS,ML,output] = PSwarm(problem,InitialPopulation,solveroptions);
      exitflag = 1;
  end
catch err
  params   = NaN(length(ActiveParams),1);
  ML       = NaN;
  vcov     = NaN(length(ActiveParams));
  g        = NaN(length(ActiveParams),1);
  H        = NaN(length(ActiveParams));
  exitflag = 0;
  output   = err;
  return
end
params                      = ParamsTransformInv(SelectParams(PARAMS));
ML                          = -ML;

%% Covariance and hessian of parameters
if nargout>=4 || (nargout>=3 && any(cov==[2 3]))
  G   = numjac(@(P) loglikfun(ParamsTransformInv(SelectParams(P)),obs,varargin{:}),...
               PARAMS,options.numjacoptions);
  g   = -sum(G,1)'/nobs;
end

if nargout>=5 || (nargout>=3 && any(cov==[1 3]))
  H = numhessian(Objective,PARAMS,options.numhessianoptions);
  if all(isfinite(H(:)))
    if ~all(eig(H)>=0)
      warning('Hessian is not positive definite')
    end
  else
    warning('Hessian has infinite elements')
  end
end

if nargout>=3
  D = diag(ParamsTransformInvDer(SelectParams(PARAMS)));
  D = D(ActiveParams,ActiveParams);
  switch cov
    case 1
      vcov = D'*(H\D)/nobs;
    case 2
      vcov = D'*((G'*G)\D);
    case 3
      vcov = D'*(H\(G'*G)/H)*D/nobs^2;
    otherwise
      vcov = [];
  end
end
