%% Generate data
rng(1)
par = [1 0.1 0.5 0.2]';
N = 10000;
X = [ones(N,1) randn(N,2)];
Y = X*par(1:end-1)+par(end)*randn(N,1);

%% OLS
MLfit = fitlm(X(:,2:end),Y);

moments = @(obs) [obs obs.*obs(:,1) obs(:,2:4).*obs(:,2) obs(:,3:4).*obs(3) obs(:,4).^2];
simulate_model = @(beta,T,ind,shocks) [X(ind,:)*beta(1:end-1)+beta(end)*shocks X(ind,:)];

rng(5)
nsim = 10 * N;
ind = randi(N,1,nsim);
shocks = randn(nsim,1);

model = struct('moments_fun', moments,...
               'simulate', @(beta,T) simulate_model(beta,T,ind,shocks));

options = struct('solveroptions',struct('Display','iter'),...
                 'weightingmatrixoptions',struct('wtype','p'));
[parhat,obj,vcov] = SMM(model,par,[Y X],options);

[parhat sqrt(diag(vcov))]
MLfit