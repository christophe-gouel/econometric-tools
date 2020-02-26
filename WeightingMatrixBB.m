function [W, m_BB] = WeightingMatrixBB(moments_fun,Obs,N_BB,block_length,diagonly)

% File inspired from Frankel and Westerhoff (2012, JEDC, Appendix B)

if nargin < 5 || isempty(diagonly), diagonly = 0; end
if nargin < 4, block_length = []; end
if nargin < 3 || isempty(N_BB), N_BB = 1E2*size(Obs,1); end

m0   = mean(moments_fun(Obs));
m_BB = NaN(N_BB,size(m0,2));
for i = 1:N_BB
  Ybb         = OverlapBBoot(Obs,block_length);
  m_BB(i,:)   = mean(moments_fun(Ybb));
end

m = m_BB - mean(m_BB);
W = (m'*m) / N_BB;

if diagonly, W = diag(diag(W)); end