function Out = OverlapBBoot2(Y, lb, type)
% Overlapping Block Bootstrap procedure for a vector time series

%  File modified from Enrique M. Quilis Macroeconomic Research Department
%  Fiscal Authority for Fiscal Responsibility (AIReF)
% (https://fr.mathworks.com/matlabcentral/fileexchange/53701-bootstrapping-time-series)

% Input
% Y is the data to sample from
% l is the block length (optional)

% Output
% Out is the bootstrapped sample
% BB type
if nargin<3, type = 'MBB'; end
% Number of blocks
if nargin<2
  Bl  = Opt_Block_Length(Y); % Criterion for block length
  lsb = round(Bl(1,:));
  lcb = round(Bl(2,:));
end

% Dimension of the matrix to be bootstrapped
[T,dimy] = size(Y);

% k = fix(T/lb);
% ------------------------------------------------------------
% INDEX SELECTION
% ------------------------------------------------------------
% I = round(1+(T-lb)*rand(1,k));
% ------------------------------------------------------------
% BOOTSTRAP REPLICATION
% ------------------------------------------------------------
Out = [];
for j=1:dimy
    switch type
        case 'SBB'
   Out = [Out SBBloop(Y(:,j),lsb(j))];
        case 'MBB'
   lb  = 5;%T^(1/3); %lsb(1);%  % Criterion for block length        
   Out = [Out MBBloop(Y(:,j),lb)];
        otherwise
   Out = [Out CBBloop(Y(:,j),lcb(j))];
    end
            
end
% ============================================================
% BBloop ==> UNIVARIATE BOOTSTRAP LOOP
% ============================================================

function yb = MBBloop(y,lb)
% Moving block bootstrap
% ceil(T/m) Uniform random numbers over 1...T-m+1
yRepl = [y;y];
T = length(y);
u = ceil((T-lb+1)*rand(ceil(T/lb),1));
% u = bsxfun(@plus,u,0:lb-1)';
u = bsxfun(@plus,u,0:ceil(lb)-1)';
% Transform to col vector, and remove excess
u = u(:); 
u = u(1:T);
% yb sample simulation
yb = yRepl(u);

function yb = CBBloop(y,lb)
% Circular block bootstrap
yRepl = [y;y];
T = length(y);
% ceil(T/m) Uniform random numbers over 1...T-m+1
u = ceil(T*rand(ceil(T/lb),1));
u = bsxfun(@plus,u,0:lb-1)';
% Transform to col vector, and remove excess
u = u(:); 
u = u(1:T);
% yb sample simulation
yb = yRepl(u);

function yb = SBBloop(y,lb)
% Stationary block bootstrap
yRepl = [y;y];
T = length(y);
u = zeros(T,1);
u(1) = ceil(T*rand);
for t=2:T
  if rand<1/lb
    u(t) = ceil(T*rand);
  else
    u(t) = u(t-1) + 1;
  end
end
% yb sample simulation
yb = yRepl(u);
