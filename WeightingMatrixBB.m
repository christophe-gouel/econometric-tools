function [W, m_BB] = WeightingMatrixBB(model,Obs,N_BB)

% File ispired from Frankel and Westerhoff (Appendix B JEDC, 2012)


if nargin<3,  N_BB = 1E2*size(Obs,1); end

moments_fun  = model.moments_fun;
for i = 1:N_BB
  Ybb         = OverlapBBoot(Obs);  
  m_BB(i,:)   = mean(moments_fun(Ybb));
end

m = m_BB- mean(m_BB);
W = (m'*m) / N_BB;
