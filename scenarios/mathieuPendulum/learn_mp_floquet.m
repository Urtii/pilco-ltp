%% learn_mp_floquet.m
% *Summary:* Alternative learning entry point for the Mathieu pendulum using
% a Floquet-style periodic state-space model and simple CEM policy search.
%
% This script is intentionally independent of trainDynModel.m / learnPolicy.m.

clear all; close all;
settings_mp;

% Keep a dedicated log for the alternative pipeline.
floquetLog = struct();
floquetLog.model = cell(1,N);
floquetLog.params = cell(1,N);
floquetLog.realCost = cell(1,N);

% Number of phases for periodic output map C_p.
nPhase = max(10, round(2*pi/(0.05*3.0))); % heuristic tied to dynamics_mp params

% 1) Initial random rollouts using standard rollout().
rolloutCache = cell(0,1);
for jj = 1:J
  [xx, yy, ll, latent] = rollout(gaussian(mu0, S0), ...
    struct('maxU', policy.maxU), H, plant, cost);

  x = [x; xx]; y = [y; yy];

  seq = struct();
  seq.u = xx(:,end);
  seq.theta = latent(1:end-1,1); % scalar measurement y_k = theta_k
  seq.x0 = latent(1,1:2)';
  seq.cost = ll;
  rolloutCache{end+1} = seq;
end

% 2) Iterative alternative model-fit + policy search + rollout.
for it = 1:N
  % Build one long (u,y) sequence for EM from cached rollouts.
  [yTrain, uTrain] = flatten_sequences(rolloutCache);

  opts = struct('latentDim',2,'maxIter',20,'ridge',1e-5, ...
                'verbose',false,'cSmooth',0.2);
  model = fit_floquet_ssm_em(yTrain, uTrain, nPhase, opts);
  floquetLog.model{it} = model;

  % Policy: u = K*y + bias (scalar y, scalar u), optimized via CEM on model.
  [Kopt, bopt] = optimize_linear_policy_cem(model, mu0, H, policy.maxU, cost);
  floquetLog.params{it} = [Kopt, bopt];

  pol = struct();
  pol.maxU = policy.maxU;
  pol.p.K = Kopt;
  pol.p.bias = bopt;
  pol.fcn = @linear_output_feedback;

  [xx, yy, ll, latent] = rollout(gaussian(mu0, S0), pol, H, plant, cost);
  x = [x; xx]; y = [y; yy];

  seq = struct();
  seq.u = xx(:,end);
  seq.theta = latent(1:end-1,1);
  seq.x0 = latent(1,1:2)';
  seq.cost = ll;
  rolloutCache{end+1} = seq;

  floquetLog.realCost{it} = ll;
  fprintf('Floquet iteration %d/%d: rollout mean cost = %.4f\n', ...
    it, N, mean(ll));
end

%% ------------------------------ Local functions -------------------------
function u = linear_output_feedback(policy, m, s) %#ok<INUSD>
% m contains policy inputs; first entry is treated as scalar measurement y.
y = m(1);
u = policy.p.K*y + policy.p.bias;
u = max(min(u, policy.maxU), -policy.maxU);
end

function [yAll, uAll] = flatten_sequences(cache)
yAll = []; uAll = [];
for i = 1:numel(cache)
  yAll = [yAll; cache{i}.theta(:)];
  uAll = [uAll; cache{i}.u(:)];
end
end

function [Kbest, bbest] = optimize_linear_policy_cem(model, mu0, H, umax, cost)
% Small CEM optimizer over [K,b] using model rollout objective.
mu = [0; 0];
Sigma = diag([1.0, 0.5].^2);

nElite = 12;
nPop = 64;
nIter = 15;

bestJ = inf; bestTheta = [0;0];
for it = 1:nIter
  samples = mvnrnd(mu', Sigma, nPop)';
  J = zeros(1,nPop);
  for j = 1:nPop
    K = samples(1,j); b = samples(2,j);
    J(j) = model_rollout_cost(model, mu0, H, K, b, umax, cost);
  end

  [Jsorted, idx] = sort(J, 'ascend');
  elite = samples(:,idx(1:nElite));

  mu = mean(elite,2);
  Sigma = cov(elite') + 1e-6*eye(2);

  if Jsorted(1) < bestJ
    bestJ = Jsorted(1);
    bestTheta = samples(:,idx(1));
  end
end

Kbest = bestTheta(1);
bbest = bestTheta(2);
end

function J = model_rollout_cost(model, x0, H, K, b, umax, cost)
% Deterministic rollout on fitted model with periodic C_p observation map.
z = x0(:);
J = 0;
for k = 1:H
  p = mod(k-1, model.N) + 1;
  yk = model.C{p}*z;
  uk = K*yk + b;
  uk = max(min(uk, umax), -umax);

  xk = z(1:2);
  e = xk - cost.target(:);
  J = J + e' * cost.W * e;

  z = model.A*z + model.B*uk;
end
J = J/H;
end
