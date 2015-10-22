function [params,ML,vcov,g,H,ModelCriterion,exitflag,output] = MaxLik(loglikfun,params,obs,options,varargin)
% MAXLIK Maximizes a log-likelihood function
%
% PARAMS = MAXLIK(LOGLIKFUN,PARAMS,OBS) maximizes the log-likelihood function
% LOGLIKFUN with respect to parameters with initial values PARAMS and using the
% observables OBS. PARAMS is either a matrix or a table with parameters in rows.
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
%   solver                : a string or a cell array of string that indicates
%                           the optimization solvers to use successively.
%                           Possible values are 'fmincon', 'fminunc' (default),
%                           'fminsearch', 'ga', 'particleswarm', 'patternsearch',
%                           or 'pswarm'
%   solveroptions         : options structure or cell array of options to be
%                           passed to the solvers maximizing the likelihood.
%   Vectorized            : 'on' or {'off'} to specify whether LOGLIKFUN is vectorized
%
% PARAMS = MAXLIK(LOGLIKFUN,PARAMS,OBS,OPTIONS,VARARGIN) provides additional
% arguments for LOGLIKFUN, which, in this case, takes the following form:
% LOGLIKFUN(PARAMS,OBS,VARARGIN).
%
% [PARAMS,ML] = MAXLIK(LOGLIKFUN,PARAMS,OBS,...) returns the log-likelihood at
% the solution: \sum_{i=1}^n log f(params,obs_i).
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

% Copyright (C) 2014-2015 Christophe Gouel
% Licensed under the Expat license

%% Initialization

nparams = size(params,1);

defaultopt = struct('ActiveParams'         , []                          ,...
                    'bounds'               , struct('lb',-inf(nparams,1) ,...
                                                    'ub', inf(nparams,1)),...
                    'cov'                  , 3                           ,...
                    'numhessianoptions'    , struct()                    ,...
                    'numjacoptions'        , struct()                    ,...
                    'ParamsTransform'      , @(P) P                      ,...
                    'ParamsTransformInv'   , @(P) P                      ,...
                    'ParamsTransformInvDer', @(P) ones(size(P))          ,...
                    'solver'               , {'fminunc'}                 ,...
                    'solveroptions'        , {struct()}                  ,...
                    'TestParamsTransform'  , true                        ,...
                    'Vectorized'           , 'off');
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
if strcmpi(options.Vectorized,'on')
  options.numjacoptions.Vectorized = 'on';
end

ActiveParams          = options.ActiveParams;
cov                   = options.cov;
ParamsTransform       = options.ParamsTransform;
ParamsTransformInv    = options.ParamsTransformInv;
ParamsTransformInvDer = options.ParamsTransformInvDer;
if ischar(options.solver)
  solver              = {options.solver};
else
  solver              = options.solver;

end
if isstruct(options.solveroptions)
  solveroptions       = {options.solveroptions};
else
  solveroptions       = options.solveroptions;
end
validateattributes(solveroptions,{'cell'},{'numel',numel(solver)})

validateattributes(loglikfun,{'char','function_handle'},{},1)
validateattributes(params,{'numeric','table'},{'2d'},2)

if isa(params,'table')
  CoefficientNames = params.Properties.RowNames;
  params = params{:,:};
  ToTable = @(Estimate) table(Estimate,'RowNames',CoefficientNames);
else
  ToTable = @(P) P;
end

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
  ActiveParams = true(nparams,1);
else
  validateattributes(ActiveParams,{'logical','numeric'},{'vector','numel',nparams})
  ActiveParams = ActiveParams(:)~=zeros(nparams,1);
end
ActiveParams0 = ActiveParams;
k             = sum(ActiveParams0); % # estimated parameters

%% Functions and matrices to extract active parameters for the estimation
SelectParamsMat = zeros(nparams,k);
ind             = 1:nparams;
SelectParamsMat(sub2ind(size(SelectParamsMat),ind(ActiveParams),1:k)) = 1;
FixedParams     = ParamsTransform(params(:,1)).*(~ActiveParams);
SelectParams    = @(P) FixedParams(:,ones(size(P,2),1))+SelectParamsMat*P;
PARAMS          = SelectParamsMat'*ParamsTransform(params);

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

%% Default values in case of errors
ModelCriterion = struct('AIC'  , NaN ,...
                        'AICc' , NaN ,...
                        'BIC'  , NaN ,...
                        'CAIC' , NaN);
vcov = NaN(k,k);
g    = NaN(length(ActiveParams),1);
H    = NaN(length(ActiveParams));

%% Maximization of the log-likelihood
try
  for i=1:length(solver)
    switch lower(solver{i})
      case {'fmincon','fminunc','fminsearch','patternsearch'}
        %% MATLAB solvers
        Objective = @(P) -sum(loglikfun(ToTable(ParamsTransformInv(SelectParams(P))),...
                                        obs,varargin{:}),1)/nobs;
        problem = struct('objective', Objective,...
                         'x0'       , PARAMS,...
                         'solver'   , solver{i},...
                         'lb'       , lb(ActiveParams),...
                         'ub'       , ub(ActiveParams),...
                         'options'  , solveroptions{i});
        [PARAMS,ML,exitflag,output] = feval(solver{i},problem);

      case 'particleswarm'
        Objective = @(P) -sum(loglikfun(ToTable(ParamsTransformInv(SelectParams(P'))),...
                                        obs,varargin{:}),1)'/nobs;
        solveroptions{i}.InitialSwarm = PARAMS';
        problem = struct('solver'   , solver{i},...
                         'objective', Objective,...
                         'nvars'    , k,...
                         'lb'       , lb(ActiveParams),...
                         'ub'       , ub(ActiveParams),...
                         'options'  , solveroptions{i});
        [PARAMS,ML,exitflag,output] = feval(solver{i},problem);
        PARAMS = PARAMS';

      case 'ga'
        Objective = @(P) -sum(loglikfun(ToTable(ParamsTransformInv(SelectParams(P'))),...
                                        obs,varargin{:}),1)'/nobs;
        solveroptions{i}.InitialPopulation = PARAMS';
        problem = struct('solver'    , solver{i},...
                         'fitnessfcn', Objective,...
                         'nvars'     , k,...
                         'lb'        , lb(ActiveParams),...
                         'ub'        , ub(ActiveParams),...
                         'Aineq'     , [],...
                         'Bineq'     , [],...
                         'Aeq'       , [],...
                         'Beq'       , [],...
                         'nonlcon'   , [],...
                         'intcon'    , [],...
                         'options'   , solveroptions{i});
        [PARAMS,ML,exitflag,output] = feval(solver{i},problem);
        PARAMS = PARAMS';

      case 'pswarm'
        %% PSwarm
        Objective = @(P) -sum(loglikfun(ToTable(ParamsTransformInv(SelectParams(P))),...
                                        obs,varargin{:}),1)/nobs;
        if strcmpi(options.Vectorized,'on'), solveroptions{i}.Vectorized = 1; end
        problem = struct('Variables'  , k,...
                         'ObjFunction', Objective,...
                         'LB'         , lb(ActiveParams),...
                         'UB'         , ub(ActiveParams));
        InitialPopulation = cell2struct(mat2cell(PARAMS,...
                                                 size(PARAMS,1),...
                                                 ones(1,size(PARAMS,2))),...
                                        'x');
        [PARAMS,ML,output] = PSwarm(problem,InitialPopulation,solveroptions{i});
        exitflag = 1;

      otherwise
        error(['Invalid value for OPTIONS field solver: must be ' ...
               '''fmincon'', ''fminunc'', ''fminsearch'', ''ga'', ''particleswarm'', ' ...
               '''patternsearch'', or ''pswarm''']);
    end
  end
catch err
  %% Values in case of error
  params   = NaN(length(ActiveParams),1);
  ML       = NaN;
  exitflag = 0;
  output   = err;
  return

end
params                      = ParamsTransformInv(SelectParams(PARAMS));
ML                          = -ML*nobs;

%% Covariance and hessian of parameters

% Covariance and hessian are only calculated for parameters that are not at their bounds
AtBounds               = params==options.bounds.lb | params==options.bounds.ub;
ActiveParams(AtBounds) = 0;
SelectParamsMat        = zeros(nparams,sum(ActiveParams));
SelectParamsMat(sub2ind(size(SelectParamsMat),ind(ActiveParams),1:sum(ActiveParams))) = 1;
FixedParams            = ParamsTransform(params).*(~ActiveParams);
SelectParams           = @(P) FixedParams(:,ones(size(P,2),1))+SelectParamsMat*P;
PARAMS                 = SelectParamsMat'*ParamsTransform(params);
Objective = @(P) -sum(loglikfun(ToTable(ParamsTransformInv(SelectParams(P))),...
                                obs,varargin{:}),1)/nobs;

% Gradient
if nargout>=4 || (nargout>=3 && any(cov==[2 3]))
  try
    G   = numjac(@(P) loglikfun(ToTable(ParamsTransformInv(SelectParams(P))),...
                                obs,varargin{:}),...
                 PARAMS,options.numjacoptions);
    g   = -sum(G,1)'/nobs;
  catch err
    %% Values in case of error
    output   = err;
    return

  end
end

% Hessian
if nargout>=5 || (nargout>=3 && any(cov==[1 3]))
  try
    H = numhessian(Objective,PARAMS,options.numhessianoptions);
  catch err
    %% Values in case of error
    output   = err;
    return

  end
  if all(isfinite(H(:)))
    if ~all(eig(H)>=0)
      warning('MaxLik:HessNotPosDef','Hessian is not positive definite')
    end
  else
    warning('MaxLik:HessInfElements','Hessian has infinite elements')
  end
end

% Covariance
if nargout>=3
  D   = diag(ParamsTransformInvDer(SelectParams(PARAMS)));
  D   = D(ActiveParams,ActiveParams);
  ind = ActiveParams(ActiveParams0);
  switch cov
    case 1
      vcov(ind,ind) = D'*(H\D)/nobs;
    case 2
      vcov(ind,ind) = D'*((G'*G)\D);
    case 3
      vcov(ind,ind) = D'*(H\(G'*G)/H)*D/nobs^2;
    otherwise
      vcov = [];
  end
end

if exist('CoefficientNames','var')
  SE = NaN(length(ActiveParams0),1);
  SE(ActiveParams0) = sqrt(diag(vcov));
  params = table(params,SE,params./SE,...
                 'VariableNames',{'Estimate' 'SE' 'tStat'},...
                 'RowNames',CoefficientNames);
end

ModelCriterion = struct('AIC'  , -2*ML+k*2                 ,...
                        'AICc' , -2*ML+k*2*nobs/(nobs-k-1) ,...
                        'BIC'  , -2*ML+k*log(nobs)         ,...
                        'CAIC' , -2*ML+k*(log(nobs)+1));