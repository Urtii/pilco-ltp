%% loss_mp.m
% *Summary:* Quadratic loss for Mathieu pendulum scenario.
%
% Implements expected value and variance of
%   c(x) = (x - target)' * W * (x - target)
% for x ~ N(m,s), matching PILCO loss function interfaces.
%
% Usage:
%   [L,dLdm,dLds,S,dSdm,dSds,C,dCdm,dCds] = loss_mp(cost,m,s)
%
% Required fields in cost:
%   cost.target   [D x 1] target state (here [0;0])
% Optional fields:
%   cost.W        [D x D] positive semi-definite weight matrix

function [L,dLdm,dLds,S,dSdm,dSds,C,dCdm,dCds] = loss_mp(cost,m,s)

D = length(m);

% Defaults and argument checks.
if isfield(cost,'W')
  W = cost.W;
else
  W = eye(D);
end

if isfield(cost,'target')
  z = cost.target;
elseif isfield(cost,'z')
  z = cost.z;
else
  z = zeros(D,1);
end

if ~isequal(size(W), [D D])
  error('loss_mp:InvalidW', 'cost.W must be a %d-by-%d matrix.', D, D);
end
if numel(z) ~= D
  error('loss_mp:InvalidTarget', 'cost.target must have %d elements.', D);
end

z = z(:);

% 1) Expected cost E[c(x)]
L = s(:)'*W(:) + (z-m)'*W*(z-m);

if nargout > 1
  dLdm = 2*(m-z)'*W;
  dLds = W';
end

% 2) Variance Var[c(x)]
if nargout > 3
  S = trace(W*s*(W + W')*s) + (z-m)'*(W + W')*s*(W + W')*(z-m);
  if S < 1e-12
    S = 0; % numerical cleanup
  end
end

if nargout > 4
  dSdm = -(2*(W+W')*s*(W+W')*(z-m))';
  dSds = W'*s'*(W + W')' + (W + W')'*s'*W' + ...
         (W + W')*(z-m)*((W + W')*(z-m))';
end

% 3) inv(s) times input-output covariance
if nargout > 6
  C = 2*W*(m-z);
  dCdm = 2*W;
  dCds = zeros(D,D^2);
end
