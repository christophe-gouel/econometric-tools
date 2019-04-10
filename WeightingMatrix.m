function W = WeightingMatrix(m,wtype,wlags,center)

% File modified from the SGMM MATLAB toolbox
% (http://www.yildiz.edu.tr/~tastan/SGMM.html)

if nargin<4 || isempty(center), center = 1; end
if nargin<3 || isempty(wlags),  wlags = 0; end
if nargin<2 || isempty(wtype),  wtype = 'i'; end

if center, m = m - mean(m); end

T = size(m,1);
switch wtype
  case 'b'                    % Bartlett
    w = 1 - (1:wlags) / (wlags+1);
  case 'p'                    % Parzen
    x = 1 - (1:wlags) / (wlags+1);
    w = 1-6*x.^2.*(1-x);
    ww = 2*(1-x).^3;
    w(x >= 0.5) = ww(x >= 0.5);
    w = fliplr(w);
  case 'i'
    W = eye(size(m,2));
    return
end

w = w / T;

W = (m'*m) / T;
for i = 1:wlags
  temp = m(1:end-i,:)' * m(i+1:end,:);
  W = W + w(i) * (temp + temp');
end
%W = inv(W);

