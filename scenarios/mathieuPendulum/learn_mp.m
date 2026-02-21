%% learn_mp.m
% *Summary:* Script to learn a controller for the Mathieu pendulum scenario.
%
% This script follows the same high-level learning flow as cartPole_learn.m:
% # Load settings
% # Run J random rollouts
% # Iterate trainDynModel / learnPolicy / applyController

% 1. Initialization
clear all; close all;
settings_mp;                             % load scenario-specific settings
basename = 'mathieuPendulum_';           % filename stem for saved artifacts

% 2. Initial J random rollouts
for jj = 1:J
  [xx, yy, realCost{jj}, latent{jj}] = ...
    rollout(gaussian(mu0, S0), struct('maxU',policy.maxU), H, plant, cost);
  x = [x; xx]; y = [y; yy];              % augment dynamics training sets

  if plotting.verbosity > 0              % visualize trajectory
    if ~ishandle(1); figure(1); else set(0,'CurrentFigure',1); end; clf(1);
    draw_rollout_mp;
  end
end

mu0Sim(odei,:) = mu0; S0Sim(odei,odei) = S0;
mu0Sim = mu0Sim(dyno); S0Sim = S0Sim(dyno,dyno);

% 3. Controlled learning (N iterations)
for j = 1:N
  trainDynModel;    % train (GP) dynamics model
  learnPolicy;      % optimize policy
  applyController;  % execute policy on real dynamics

  disp(['mathieuPendulum controlled trial # ' num2str(j)]);

  if plotting.verbosity > 0              % visualize trajectory
    if ~ishandle(1); figure(1); else set(0,'CurrentFigure',1); end; clf(1);
    draw_rollout_mp;
  end
end
