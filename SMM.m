function [params,Obj,vcov,G,exitflag,output] = SMM(model,params,obs,options,varargin)
% SMM Maximizes a log-likelihood function
%
% PARAMS = SMM(MODEL,PARAMS,OBS) maximizes the log-likelihood function
% MODEL with respect to parameters with initial values PARAMS and using the
% observables OBS. PARAMS is either a matrix or a table with parameters in rows.
%
% PARAMS = SMM(MODEL,PARAMS,OBS,OPTIONS) maximizes the log-likelihood
% function with the parameters defined by the structure OPTIONS. The fields of
% the structure are
%   ActiveParams          :
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
%
% PARAMS = SMM(MODEL,PARAMS,OBS,OPTIONS,VARARGIN) provides additional
% arguments for MODEL, which, in this case, takes the following form:
% MODEL(PARAMS,OBS,VARARGIN).
%
% [PARAMS,OBJ] = SMM(MODEL,PARAMS,OBS,...) returns the value of the
% objective at the solution.
%
% [PARAMS,OBJ,VCOV] = SMM(MODEL,PARAMS,OBS,...)
%
% [PARAMS,OBJ,VCOV,G] = SMM(MODEL,PARAMS,OBS,...) returns the gradient
% with respect to the parameters of the log-likelihood at the solution.
%
% [PARAMS,OBJ,VCOV,G,EXITFLAG] = SMM(MODEL,PARAMS,OBS,...) returns the
% exitflag from the optimization solver.
%
% [PARAMS,OBJ,VCOV,G,EXITFLAG,OUTPUT] = SMM(MODEL,PARAMS,OBS,...)
% returns a structure OUTPUT that contains the overidentification statistics and
% information about the optimization from the optimization solver.
%
% See also FMINSEARCH, FMINUNC, NUMJAC.

% Copyright (C) 2019-2020 Christophe Gouel
% Licensed under the Expat license

%% Initialization

nparams = size(params,1);

defaultopt = struct('ActiveParams'          , []                          ,...
                    'bounds'                , struct('lb',-inf(nparams,1) ,...
                                                     'ub', inf(nparams,1)),...
                    'modeltype'             , 'smm'                       ,...
                    'nlag'                  , 0                           ,...
                    'nrep'                  , 10                          ,...
                    'numjacoptions'         , struct()                    ,...
                    'ParamsTransform'       , @(P) P                      ,...
                    'ParamsTransformInv'    , @(P) P                      ,...
                    'ParamsTransformInvDer' , @(P) ones(size(P))          ,...
                    'solver'                , {'fminunc'}                 ,...
                    'solveroptions'         , {struct()}                  ,...
                    'TestParamsTransform'   , true                        ,...
                    'W'                     , []                          ,...
                    'weightingmatrixoptions', struct('wtype' , 'i',...
                                                     'wlags' , 0  ,...
                                                     'center', 1 ,...
                                                     'diagonly', 0));
if nargin < 4 || isempty(options)
  options = defaultopt;
else
  warning('off','catstruct:DuplicatesFound')
  if isfield(options,'bounds')
    options.bounds = catstruct(defaultopt.bounds,options.bounds);
  end
  if isfield(options,'numjacoptions')
    options.numjacoptions = catstruct(defaultopt.numjacoptions,...
                                      options.numjacoptions);
  end
  if isfield(options,'weightingmatrixoptions')
    options.weightingmatrixoptions = catstruct(defaultopt.weightingmatrixoptions,...
                                               options.weightingmatrixoptions);
  end
  options = catstruct(defaultopt,options);
end

ActiveParams          = options.ActiveParams;
ParamsTransform       = options.ParamsTransform;
ParamsTransformInv    = options.ParamsTransformInv;
ParamsTransformInvDer = options.ParamsTransformInvDer;
if ischar(options.solver)
  solver              = {options.solver};
else
  solver              = options.solver;
end
if iscell(options.solveroptions)
  solveroptions       = options.solveroptions;
else
  solveroptions       = {options.solveroptions};
end
validateattributes(solveroptions,{'cell'},{'numel',numel(solver)})
weightingmatrixoptions = options.weightingmatrixoptions;

validateattributes(model,{'struct'},{},1)
validateattributes(params,{'numeric','table'},{'2d'},2)

if isa(params,'table')
  CoefficientNames = params.Properties.RowNames;
  params = params{:,1};
  ToTable = @(Estimate) table(Estimate,'RowNames',CoefficientNames);
else
  ToTable = @(P) P;
end

nobs0 = size(obs,1);
nrep  = options.nrep;

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

if isempty(ActiveParams)
  ActiveParams = true(nparams,1);
else
  validateattributes(ActiveParams,{'logical','numeric'},{'vector','numel',nparams})
  ActiveParams = ActiveParams(:)~=zeros(nparams,1);
end
ActiveParams0 = ActiveParams;
nactparams    = sum(ActiveParams0); % # estimated parameters

%% Functions and matrices to extract active parameters for the estimation
SelectParamsMat = zeros(nparams,nactparams);
ind             = 1:nparams;
SelectParamsMat(sub2ind(size(SelectParamsMat),ind(ActiveParams),1:nactparams)) = 1;
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

%% Observed moments
moments_fun  = model.moments_fun;
moments_obs  = moments_fun(obs);
if isempty(options.W)
  switch lower(weightingmatrixoptions.wtype)
    case 'bb' % for block bootstrap
      W = WeightingMatrixBB(moments_fun,obs,...
                            weightingmatrixoptions.N_BB,...
                            weightingmatrixoptions.block_length,...
                            weightingmatrixoptions.diagonly);
    otherwise
      W = WeightingMatrix(moments_obs,...
                          weightingmatrixoptions.wtype,...
                          weightingmatrixoptions.wlags,...
                          weightingmatrixoptions.center,...
                          weightingmatrixoptions.diagonly);
  end
else
  W = options.W;
end
try
  W = pinv(W);
catch err
  %% Values in case of error
  params   = NaN(length(ActiveParams),1);
  M        = NaN;
  Obj      = NaN;
  exitflag = 0;
  vcov     = NaN(nactparams,nactparams);
  G        = NaN(size(W,1),nactparams);
  output   = err;
  return

end

Emoments_obs = mean(moments_obs,1); % (1,nmom)

[nobs1,nmom] = size(moments_obs);
if strcmpi(options.modeltype, 'smm')
  nsim = nobs1 * (nrep - 1) + nobs0;
  nlost = nobs0 - nobs1;
elseif strcmpi(options.modeltype, 'ind')
  nobs1 = nobs0 - nlag;
  nsim = nobs1 * nrep + nlag;
  nlost = nlag;
end

% SimMoments will have to be changed for indirect inference: parameters should
% be estimated on nrep subsamples
SimMoments   = @(P) moments_fun(model.simulate(P,nsim,varargin{:}));

%% Default values in case of errors
vcov = NaN(nactparams,nactparams);
G    = NaN(nmom,nactparams);

%% Minmization of the objective
MaxIter_names = {'MaxIter', 'MaxIterations'};
ismaxiter = isfield(solveroptions{1}, MaxIter_names);
if any(ismaxiter)
  tosolve = getfield(solveroptions{1}, MaxIter_names{ismaxiter}) ~= 0; %#ok
  if isempty(tosolve), tosolve = true; end
else
  tosolve = true;
end

if tosolve
  try
    for i=1:length(solver)
      switch lower(solver{i})
        case {'fmincon','fminunc','fminsearch','patternsearch'}
          %% MATLAB solvers except fminbnd
          Objective = @(P) SMMObj(ToTable(ParamsTransformInv(SelectParams(P))));
          problem = struct('objective', Objective,...
                           'x0'       , PARAMS,...
                           'solver'   , solver{i},...
                           'lb'       , lb(ActiveParams),...
                           'ub'       , ub(ActiveParams),...
                           'options'  , solveroptions{i});
          [PARAMS,Obj,exitflag,output] = feval(solver{i},problem);
        case 'fminbnd'
          %% fminbnd
          Objective = @(P) SMMObj(ToTable(ParamsTransformInv(SelectParams(P))));
          problem = struct('objective', Objective,...
                           'solver'   , solver{i},...
                           'x1'       , lb(ActiveParams),...
                           'x2'       , ub(ActiveParams),...
                           'options'  , solveroptions{i});
          [PARAMS,Obj,exitflag,output] = feval(solver{i},problem);

        case 'multistart'
          Objective = @(P) SMMObj(ToTable(ParamsTransformInv(SelectParams(P)')));
          problem = createOptimProblem(             options.subsolver,...
                                                    'x0'       , PARAMS(:,1)',...
                                                    'objective', Objective,...
                                                    'lb'       , lb(ActiveParams),...
                                                    'ub'       , ub(ActiveParams),...
                                                    'options'  , solveroptions{i});
          startpts = CustomStartPointSet(PARAMS');
          ms = MultiStart('StartPointsToRun','bounds-ineqs',...
                          'UseParallel',true);
          [PARAMS,Obj,exitflag,output,solutions] = run(ms,problem,startpts);
          output.solutions = solutions;
          PARAMS = PARAMS';

        case 'particleswarm'
          Objective = @(P) SMMObj(ToTable(ParamsTransformInv(SelectParams(P)')));
          solveroptions{i}.InitialSwarm = PARAMS';
          problem = struct('solver'   , solver{i},...
                           'objective', Objective,...
                           'nvars'    , nactparams,...
                           'lb'       , lb(ActiveParams),...
                           'ub'       , ub(ActiveParams),...
                           'options'  , solveroptions{i});
          [PARAMS,Obj,exitflag,output] = feval(solver{i},problem);
          PARAMS = PARAMS';

        case 'ga'
          Objective = @(P) SMMObj(ToTable(ParamsTransformInv(SelectParams(P)')));
          solveroptions{i}.InitialPopulation = PARAMS';
          problem = struct('solver'    , solver{i},...
                           'fitnessfcn', Objective,...
                           'nvars'     , nactparams,...
                           'lb'        , lb(ActiveParams),...
                           'ub'        , ub(ActiveParams),...
                           'Aineq'     , [],...
                           'Bineq'     , [],...
                           'Aeq'       , [],...
                           'Beq'       , [],...
                           'nonlcon'   , [],...
                           'intcon'    , [],...
                           'options'   , solveroptions{i});
          [PARAMS,Obj,exitflag,output] = feval(solver{i},problem);
          PARAMS = PARAMS';

        case 'pswarm'
          %% PSwarm
          Objective = @(P) SMMObj(ToTable(ParamsTransformInv(SelectParams(P))));
          problem = struct('Variables'  , nactparams,...
                           'ObjFunction', Objective,...
                           'LB'         , lb(ActiveParams),...
                           'UB'         , ub(ActiveParams));
          InitialPopulation = cell2struct(mat2cell(PARAMS,...
                                                   size(PARAMS,1),...
                                                   ones(1,size(PARAMS,2))),...
                                          'x');
          [PARAMS,Obj,output] = PSwarm(problem,InitialPopulation,solveroptions{i});
          exitflag = 1;

        case 'lesage'
          Objective = @(P) SMMObj(ToTable(ParamsTransformInv(SelectParams(P))));
          output = maxlik(Objective, PARAMS, solveroptions{i});
          PARAMS = output.b;
          Obj = output.f;
          exitflag = 1;

        case 'opti'
          %% OPTI toolbox (https://inverseproblem.co.nz/OPTI/index.php)
          Objective = @(P) SMMObj(ToTable(ParamsTransformInv(SelectParams(P))));
          problem = opti('fun'    ,Objective,...
                         'x0'     ,PARAMS,...
                         'bounds' ,lb(ActiveParams),ub(ActiveParams),...
                         'options',solveroptions{i});
          [PARAMS,Obj,exitflag,output] = solve(problem);
        case 'nlopt'
          Objective = @(P) SMMObj(ToTable(ParamsTransformInv(SelectParams(P'))));
          problem = struct('algorithm'    ,NLOPT_LN_NELDERMEAD,...
                           'lower_bounds' ,lb(ActiveParams),...
                           'upper_bounds' ,ub(ActiveParams),...
                           'min_objective',Objective);
          problem = catstruct(problem,solveroptions{i});
          [PARAMS,Obj,exitflag] = nlopt_optimize(problem,PARAMS);
          output = [];

        otherwise
          error(['Invalid value for OPTIONS field solver: must be ' ...
                 '''fmincon'', ''fminunc'', ''fminsearch'', ''ga'', ''particleswarm'', ' ...
                 '''patternsearch'', ''pswarm'', ''lesage'', or ''opti''']);
      end
    end
  catch err
    %% Values in case of error
    params   = NaN(length(ActiveParams),1);
    M        = NaN;
    Obj      = NaN;
    exitflag = 0;
    output   = err;
    return

  end
else
  Obj = SMMObj(ToTable(ParamsTransformInv(SelectParams(PARAMS))));
  exitflag = 1;
  output   = '';
end
params                      = ParamsTransformInv(SelectParams(PARAMS));

%% Export W
output.W = W;

%% Overidentification test
output.OID_stat   = Obj * nobs1 * nrep / (1 + nrep);
output.dof        = nmom - nactparams;
output.OID_pvalue = chi2pdf(output.OID_stat, output.dof);

%% Export moments
output.moments_obs = Emoments_obs;
output.moments_sim = mean(SimMoments(ToTable(ParamsTransformInv(SelectParams(PARAMS)))),1);

%% Covariance of parameters

% Covariance is only calculated for parameters that are not at their bounds
AtBounds               = params==options.bounds.lb | params==options.bounds.ub;
ActiveParams(AtBounds) = 0;
SelectParamsMat        = zeros(nparams,sum(ActiveParams));
SelectParamsMat(sub2ind(size(SelectParamsMat),ind(ActiveParams),1:sum(ActiveParams))) = 1;
FixedParams            = ParamsTransform(params).*(~ActiveParams);
SelectParams           = @(P) FixedParams(:,ones(size(P,2),1))+SelectParamsMat*P;
PARAMS                 = SelectParamsMat'*ParamsTransform(params);
nactparams             = sum(ActiveParams);

% Gradient
if nargout>=4 || nargout>=3
  try
    % Matrix of contributions to the gradient
    vec = @(y) y(:);
    G   = numjac(@(P) vec(SimMoments(ToTable(ParamsTransformInv(SelectParams(P))))),...
                 PARAMS,options.numjacoptions); % (nsim*nmom,nactparams)
    if strcmpi(options.modeltype, 'smm')
      G   = reshape(G,nsim-nlost,nmom,nactparams);      % (nsim,nmom,nactparams)
      J   = squeeze(mean(G,1));                   % (nmom,nactparams)
    elseif strcmpi(options.modeltype, 'ind')
      if numel(G) == (nmom * nactparams) % If auxilliary model on nrep * nobs data
        J = G;
      else                               % If nrep auxilliary model on nobs data
        G   = reshape(G,nrep,nmom,nactparams);      % (nrep,nmom,nactparams)
        J   = squeeze(mean(G,1));                   % (nmom,nactparams)
      end
    end
  catch err
    %% Values in case of error
    output   = err;
    return
  end
end

% Covariance
if nargout>=3
  D   = diag(ParamsTransformInvDer(SelectParams(PARAMS)));
  D   = D(ActiveParams,ActiveParams);
  ind = ActiveParams(ActiveParams0);
  if ~strcmpi(weightingmatrixoptions.wtype, 'i') || strcmpi(options.modeltype, 'ind')
    vcov(ind,ind) = (1 + 1 / nrep) * D' * inv(J' * W * J) * D / nobs1; %#ok
  else % Identity matrix for weighting
    S = WeightingMatrix(moments_obs,'b',floor(4 * (nobs1 / 100) ^(2 / 9)),1,0);
    vcov(ind,ind) = (1 + 1 / nrep) * D' * inv(J' * J) * (J' * S * J) * inv(J' * J) * D / nobs1; %#ok
  end
end

if exist('CoefficientNames','var')
  SE    = zeros(length(ActiveParams0),1);
  SE(ActiveParams0) = sqrt(diag(vcov));
  tStat = NaN(length(ActiveParams0),1);
  tStat(ActiveParams0) = params(ActiveParams0)./SE(ActiveParams0);
  params = table(params,SE,tStat,...
                 'VariableNames',{'Estimate' 'SE' 'tStat'},...
                 'RowNames',CoefficientNames);
end

  function Obj = SMMObj(par)

  if any(lb > par) || any(par > ub)
    Obj = Inf;
  else
    M   = Emoments_obs - mean(SimMoments(par),1);
    Obj = M * W * M';
  end

  end

end