%% test_mathieu_floquet.m
% *Summary:* Basic validation for Floquet SSM fitting on Mathieu pendulum data.
%
% Test steps:
% 1) Simulate plant with random input sequence.
% 2) Fit periodic-output Floquet state-space model.
% 3) Compare one-step predicted output against true output.
% 4) Plot phase-dependent C_p coefficients.

clear all; close all;
settings_mp;

Tsim = 500;

% Random open-loop rollout data generation.
start = gaussian(mu0, S0);
policyRand = struct('maxU', policy.maxU);
[xx, yy, ~, latent] = rollout(start, policyRand, Tsim, plant, cost);

u = xx(:,end);
yTrue = latent(1:end-1,1); % scalar measured output: theta

% Fit Floquet model.
nPhase = 30;
opts = struct('latentDim',2,'maxIter',30,'ridge',1e-5,'verbose',true,'cSmooth',0.1);
model = fit_floquet_ssm_em(yTrue, u, nPhase, opts);

% One-step ahead prediction using filtered state recursion with true y updates.
z = model.x0;
yPred = zeros(Tsim,1);
for k = 1:Tsim
  p = mod(k-1, model.N) + 1;
  yPred(k) = model.C{p} * z;
  z = model.A*z + model.B*u(k);
end

rmse = sqrt(mean((yTrue - yPred).^2));
fprintf('Floquet model output RMSE (open-loop prediction): %.6f\n', rmse);

% Visualization.
figure(1); clf;
plot(yTrue, 'k', 'LineWidth', 1); hold on;
plot(yPred, 'r--', 'LineWidth', 1);
legend('True y=theta', 'Model prediction');
xlabel('k'); ylabel('output');
title(sprintf('Mathieu pendulum Floquet fit, RMSE=%.4g', rmse));
grid on;

% Phase-dependent C_p plot.
Cstack = zeros(model.N,2);
for p = 1:model.N
  Cstack(p,:) = model.C{p};
end

figure(2); clf;
plot(1:model.N, Cstack(:,1), 'b-o', 'LineWidth', 1); hold on;
plot(1:model.N, Cstack(:,2), 'm-s', 'LineWidth', 1);
legend('C_p(1)', 'C_p(2)');
xlabel('Phase index p'); ylabel('Observation coefficients');
title('Phase-dependent observation matrix C_p');
grid on;
