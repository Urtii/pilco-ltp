%% fit_floquet_ssm_em.m
% *Summary:* EM-style estimation of a Floquet-inspired periodic observation
% linear state-space model.
%
% Model:
%   z_{k+1} = A_F z_k + B_F u_k + w_k,    w_k ~ N(0,Q)
%   y_k     = C_{p(k)} z_k + v_k,          v_k ~ N(0,R)
%   p(k)    = mod(k-1,N) + 1
%
% Inputs:
%   y      [T x 1] observed scalar output
%   u      [T x m] input sequence (m=1 in current scenario)
%   N      scalar, number of phase bins
%   opts   struct (optional fields):
%            .latentDim   latent dimension (default 2)
%            .maxIter     EM iterations (default 25)
%            .ridge       ridge regularization (default 1e-6)
%            .verbose     display progress (default true)
%            .cSmooth     circular smoothing weight in [0,1] (default 0)
%
% Output:
%   model struct with fields:
%     .A, .B, .C{p}, .Q, .R, .N, .x0, .P0

function model = fit_floquet_ssm_em(y, u, N, opts)

if nargin < 4, opts = struct(); end
if nargin < 3 || isempty(N), N = 20; end

if ~isfield(opts,'latentDim'), opts.latentDim = 2; end
if ~isfield(opts,'maxIter'),   opts.maxIter = 25; end
if ~isfield(opts,'ridge'),     opts.ridge = 1e-6; end
if ~isfield(opts,'verbose'),   opts.verbose = true; end
if ~isfield(opts,'cSmooth'),   opts.cSmooth = 0.0; end

y = y(:);
[T, ny] = size(y);
if ny ~= 1
  error('fit_floquet_ssm_em:InvalidY', 'y must be scalar output [T x 1].');
end

if isempty(u)
  u = zeros(T,1);
end
if size(u,1) ~= T
  error('fit_floquet_ssm_em:LengthMismatch', 'u and y must have equal length.');
end

n = opts.latentDim;
nu = size(u,2);
ridge = opts.ridge;

% ------------------------------ Initialization -----------------------------
A = [1 1; 0 1];
if n ~= 2
  A = eye(n);
end
B = zeros(n,nu);
if n >= 2 && nu >= 1
  B(2,1) = 1;
elseif nu >= 1
  B(1,1) = 1;
end

Q = 1e-2*eye(n);
R = max(var(y), 1e-4);
x0 = zeros(n,1);
P0 = eye(n);

C = cell(1,N);
for p = 1:N
  C{p} = zeros(1,n);
  C{p}(1,1) = 1;
end

% ------------------------------- EM loop -----------------------------------
for it = 1:opts.maxIter
  % E-step: time-varying KF + RTS smoother
  stats = estep_kf_rts(y,u,A,B,C,Q,R,x0,P0,N);

  m = stats.mSmooth;
  Ps = stats.PSmooth;
  Pcs = stats.PCross;

  % M-step for A,B via ridge least squares on expected sufficient statistics.
  S_xixi = zeros(n+nu,n+nu);
  S_zxi  = zeros(n,n+nu);
  for k = 2:T
    mk = m(:,k); mkm1 = m(:,k-1); ukm1 = u(k-1,:)';

    E_zk_zkm1 = Pcs(:,:,k) + mk*mkm1';
    E_zkm1    = Ps(:,:,k-1) + mkm1*mkm1';

    E_xi_xi = [E_zkm1,               mkm1*ukm1';
               ukm1*mkm1',           ukm1*ukm1'];
    E_z_xi  = [E_zk_zkm1, mk*ukm1'];

    S_xixi = S_xixi + E_xi_xi;
    S_zxi  = S_zxi  + E_z_xi;
  end

  Theta = S_zxi / (S_xixi + ridge*eye(n+nu));
  A = Theta(:,1:n);
  B = Theta(:,n+1:end);

  % M-step for Q.
  Qsum = zeros(n,n);
  for k = 2:T
    mk = m(:,k); mkm1 = m(:,k-1); ukm1 = u(k-1,:)';

    E_zz   = Ps(:,:,k) + mk*mk';
    E_zxi  = [Pcs(:,:,k) + mk*mkm1', mk*ukm1'];
    E_xixi = [Ps(:,:,k-1)+mkm1*mkm1', mkm1*ukm1';
              ukm1*mkm1',             ukm1*ukm1'];

    Qsum = Qsum + E_zz - Theta*E_zxi' - E_zxi*Theta' + Theta*E_xixi*Theta';
  end
  Q = Qsum/max(T-1,1);
  Q = (Q + Q')/2 + 1e-8*eye(n);

  % M-step for periodic C_p and scalar R.
  Cnum = cell(1,N);
  Cden = cell(1,N);
  for p = 1:N
    Cnum{p} = zeros(1,n);
    Cden{p} = ridge*eye(n);
  end

  Rsum = 0; count = 0;
  for k = 1:T
    p = mod(k-1,N) + 1;
    mk = m(:,k);
    Ezz = Ps(:,:,k) + mk*mk';

    Cnum{p} = Cnum{p} + y(k)*mk';
    Cden{p} = Cden{p} + Ezz;
  end

  for p = 1:N
    C{p} = Cnum{p} / Cden{p};
  end

  % Optional circular smoothing across phases.
  if opts.cSmooth > 0
    C = circular_smooth_C(C, opts.cSmooth);
  end

  for k = 1:T
    p = mod(k-1,N) + 1;
    mk = m(:,k);
    Ezz = Ps(:,:,k) + mk*mk';
    ck = C{p};
    Rsum = Rsum + y(k)^2 - 2*y(k)*(ck*mk) + ck*Ezz*ck';
    count = count + 1;
  end
  R = max(Rsum/max(count,1), 1e-8);

  % Update initial latent distribution from smoothed posterior.
  x0 = m(:,1);
  P0 = Ps(:,:,1) + 1e-8*eye(n);

  if opts.verbose
    fprintf('EM iter %02d/%02d, loglik = %.4f\n', it, opts.maxIter, stats.loglik);
  end
end

model.A = A;
model.B = B;
model.C = C;
model.Q = Q;
model.R = R;
model.N = N;
model.x0 = x0;
model.P0 = P0;
end

% -------------------------------------------------------------------------
function stats = estep_kf_rts(y,u,A,B,C,Q,R,x0,P0,N)
T = length(y);
n = size(A,1);

mPred = zeros(n,T);
PPred = zeros(n,n,T);
mFilt = zeros(n,T);
PFilt = zeros(n,n,T);

loglik = 0;

for k = 1:T
  if k == 1
    mp = x0;
    Pp = P0;
  else
    mp = A*mFilt(:,k-1) + B*u(k-1,:)';
    Pp = A*PFilt(:,:,k-1)*A' + Q;
    Pp = (Pp + Pp')/2;
  end

  p = mod(k-1,N) + 1;
  Hk = C{p}; % 1 x n

  S = Hk*Pp*Hk' + R;
  K = (Pp*Hk')/S;
  innov = y(k) - Hk*mp;

  mf = mp + K*innov;
  Pf = (eye(n)-K*Hk)*Pp;
  Pf = (Pf + Pf')/2;

  mPred(:,k) = mp; PPred(:,:,k) = Pp;
  mFilt(:,k) = mf; PFilt(:,:,k) = Pf;

  loglik = loglik - 0.5*(log(2*pi*S) + innov^2/S);
end

% RTS smoother
mSmooth = mFilt;
PSmooth = PFilt;
PCross = zeros(n,n,T); % PCross(:,:,k)=Cov(z_k, z_{k-1}|Y) for k>=2

for k = T-1:-1:1
  J = PFilt(:,:,k)*A'/PPred(:,:,k+1);
  mSmooth(:,k) = mFilt(:,k) + J*(mSmooth(:,k+1)-mPred(:,k+1));
  PSmooth(:,:,k) = PFilt(:,:,k) + J*(PSmooth(:,:,k+1)-PPred(:,:,k+1))*J';
  PSmooth(:,:,k) = (PSmooth(:,:,k) + PSmooth(:,:,k)')/2;

  % Cross-covariance recursion for EM sufficient statistics.
  if k == T-1
    PCross(:,:,k+1) = PSmooth(:,:,k+1)*J';
  else
    Jnext = PFilt(:,:,k+1)*A'/PPred(:,:,k+2);
    PCross(:,:,k+1) = PFilt(:,:,k+1)*J' + ...
      Jnext*(PCross(:,:,k+2)-A*PFilt(:,:,k+1))*J';
  end
end

stats.mSmooth = mSmooth;
stats.PSmooth = PSmooth;
stats.PCross = PCross;
stats.loglik = loglik;
end

% -------------------------------------------------------------------------
function Csm = circular_smooth_C(C, w)
% Smooth C_p along circular phase index with nearest-neighbor averaging.
N = numel(C);
Csm = C;
for p = 1:N
  pl = p-1; if pl < 1, pl = N; end
  pr = p+1; if pr > N, pr = 1; end
  Csm{p} = (1-w)*C{p} + 0.5*w*(C{pl}+C{pr});
end
end
