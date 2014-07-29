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
defaultopt = struct('ActiveParams'         , [],...
                    'bounds'               , struct('lb',-inf(size(params)),...
                                                    'ub', inf(size(params))),...
                    'cov'                  , 3,...
                    'numhessianoptions'    , struct(),...
                    'numjacoptions'        , struct(),...
                    'ParamsTransform'      , @(P) P,...
                    'ParamsTransformInv'   , @(P) P,...
                    'ParamsTransformInvDer', @(P) ones(size(P)),...
                    'solver'               , 'fminunc',...
                    'solveroptions'        , struct());
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

validateattributes(loglikfun,{'char','function_handle'},{},1)
validateattributes(params,{'numeric'},{'column','nonempty'},2)

nobs = size(obs,1);

if norm(ParamsTransformInv(ParamsTransform(params))-params)>=sqrt(eps)
  error('Functions to transform parameters are not inverse of each other')
end

if norm(diag(numjac(@(P) ParamsTransformInv(P),ParamsTransform(params)))...
        -ParamsTransformInvDer(ParamsTransform(params)))>=1E-6
  error(['The function to differentiate transformed parameters does not correspond ' ...
         'to its finite difference gradient.'])
end  
  
if isa(loglikfun,'char'), loglikfun = str2func(loglikfun); end

if isempty(ActiveParams)
  ActiveParams = true(size(params));
else
  validateattributes(ActiveParams,{'logical','numeric'},{'vector','numel',numel(params)})
  ActiveParams = ActiveParams(:)~=zeros(size(params));
end

%% Functions and matrices to extract active parameters for the estimation
SelectParamsMat = zeros(length(params),sum(ActiveParams));
ind             = 1:length(params);
SelectParamsMat(sub2ind(size(SelectParamsMat),ind(ActiveParams),1:sum(ActiveParams))) = 1;
SelectParams    = @(P) ParamsTransform(params).*(~ActiveParams)+SelectParamsMat*P;

%% Maximization of the log-likelihood
Objective = @(P) -sum(loglikfun(ParamsTransformInv(SelectParams(P)),obs,varargin{:}))/nobs;
problem = struct('objective', Objective,...
                 'x0'       , SelectParamsMat'*ParamsTransform(params),...
                 'solver'   , solver,...
                 'lb'       , options.bounds.lb(ActiveParams),...
                 'ub'       , options.bounds.ub(ActiveParams),...
                 'options'  , options.solveroptions);
try
  [PARAMS,ML,exitflag,output] = feval(solver,problem);
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
