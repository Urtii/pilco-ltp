%% learn_mp.m
% *Summary:* Standard PILCO learning entry script for mathieuPendulum.
%
% This follows the same architecture used by existing scenarios:
%   settings -> random rollouts -> trainDynModel -> learnPolicy -> applyController

clear all; close all;
settings_mp;
basename = 'mathieuPendulum_';

% 1) Initial random rollouts.
for jj = 1:J
  [xx, yy, realCost{jj}, latent{jj}] = ...
    rollout(gaussian(mu0, S0), struct('maxU',policy.maxU), H, plant, cost);
  x = [x; xx]; y = [y; yy];
end

mu0Sim(odei,:) = mu0; S0Sim(odei,odei) = S0;
mu0Sim = mu0Sim(dyno); S0Sim = S0Sim(dyno,dyno);

% 2) Controlled learning loop.
for j = 1:N
  trainDynModel;
  learnPolicy;
  applyController;
  disp(['mathieuPendulum controlled trial # ' num2str(j)]);
end
