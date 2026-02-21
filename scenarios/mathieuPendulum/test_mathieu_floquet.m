%% test_mathieu_floquet.m
% *Summary:* Validation for Floquet/LTP identification on Mathieu pendulum data
% with additional baseline system-identification models.
%
% Test steps:
% 1) Simulate plant with random input sequence.
% 2) Identify an LTP Floquet SSM (periodic C_p).
% 3) Identify non-LTP baseline models using the same data:
%       a) LTI SSM baseline (same EM routine with N=1)
%       b) ARX(2,1) baseline in output space
% 4) Compare prediction RMSE across models.
% 5) Plot phase-dependent C_p and prediction overlays.

clear all; close all;
settings_mp;

Tsim = 5000;

% Random open-loop rollout data generation.
start = gaussian(mu0, S0);
policyRand = struct('maxU', policy.maxU);
[xx, ~, ~, latent] = rollout(start, policyRand, Tsim, plant, cost);

u = xx(:,end);
yTrue = latent(1:end-1,1); % scalar measured output: theta

if numel(yTrue) ~= Tsim || numel(u) ~= Tsim
  error('test_mathieu_floquet:DataShape', ...
    'Expected u and y to both have length Tsim.');
end

%% 1) LTP identification (Floquet model)
nPhase = 30;
optsLTP = struct('latentDim',2,'maxIter',30,'ridge',1e-5,'verbose',true,'cSmooth',0.1);
modelLTP = fit_floquet_ssm_em(yTrue, u, nPhase, optsLTP);
yPredLTP = ssm_openloop_predict(modelLTP, u);
rmseLTP = sqrt(mean((yTrue - yPredLTP).^2));

%% 2a) Own baseline model #1: LTI SSM identification (N = 1)
% This uses the same EM estimator but collapses periodicity to a single phase,
% producing a time-invariant observation map.
optsLTI = struct('latentDim',2,'maxIter',30,'ridge',1e-5,'verbose',true,'cSmooth',0.0);
modelLTI = fit_floquet_ssm_em(yTrue, u, 1, optsLTI);
yPredLTI = ssm_openloop_predict(modelLTI, u);
rmseLTI = sqrt(mean((yTrue - yPredLTI).^2));

%% 2b) Own baseline model #2: ARX(2,1) identification
a = fit_arx21(yTrue, u);
yPredARX = predict_arx21(yTrue, u, a);
rmseARX = sqrt(mean((yTrue - yPredARX).^2));

%% 3) Numerical comparison report
fprintf('\nPrediction RMSE comparison (lower is better):\n');
fprintf('  LTP Floquet SSM (N=%d): %.6f\n', nPhase, rmseLTP);
fprintf('  LTI SSM baseline   (N=1): %.6f\n', rmseLTI);
fprintf('  ARX(2,1) baseline       : %.6f\n\n', rmseARX);

% Print identified Floquet/LTI state transition and control matrices (A_F, B_F).
fprintf('Identified LTP model A_F:\n'); disp(modelLTP.A);
fprintf('Identified LTP model B_F:\n'); disp(modelLTP.B);
fprintf('Identified LTI model A_F:\n'); disp(modelLTI.A);
fprintf('Identified LTI model B_F:\n'); disp(modelLTI.B);

%% 4) Visualization: output predictions
figure(1); clf;
plot(yTrue, 'k', 'LineWidth', 1.0); hold on;
plot(yPredLTP, 'r--', 'LineWidth', 1.0);
plot(yPredLTI, 'b-.', 'LineWidth', 1.0);
plot(yPredARX, 'g:', 'LineWidth', 1.2);
legend('True y=theta', ...
       sprintf('LTP Floquet (RMSE=%.4g)', rmseLTP), ...
       sprintf('LTI SSM (RMSE=%.4g)', rmseLTI), ...
       sprintf('ARX(2,1) (RMSE=%.4g)', rmseARX), ...
       'Location','best');
xlabel('k'); ylabel('output');
title('Mathieu pendulum system ID: LTP vs own baseline models');
grid on;

%% 5) Visualization: phase-dependent C_p for LTP model
Cstack = zeros(modelLTP.N,2);
for p = 1:modelLTP.N
  Cstack(p,:) = modelLTP.C{p};
end

figure(2); clf;
plot(1:modelLTP.N, Cstack(:,1), 'b-o', 'LineWidth', 1); hold on;
plot(1:modelLTP.N, Cstack(:,2), 'm-s', 'LineWidth', 1);
legend('C_p(1)', 'C_p(2)', 'Location', 'best');
xlabel('Phase index p'); ylabel('Observation coefficients');
title('LTP identification: phase-dependent observation matrix C_p');
grid on;

%% 6) Visualization: simulated system dynamics traces
t = (0:Tsim-1)' * dt;
figure(3); clf;
subplot(3,1,1);
plot(t, latent(1:end-1,1), 'k', 'LineWidth', 1.0);
ylabel('theta [rad]');
title('Mathieu pendulum simulation traces');
grid on;

subplot(3,1,2);
plot(t, latent(1:end-1,2), 'b', 'LineWidth', 1.0);
ylabel('theta\_dot [rad/s]');
grid on;

subplot(3,1,3);
plot(t, u, 'r', 'LineWidth', 1.0);
xlabel('time [s]'); ylabel('u');
grid on;

%% ------------------------------ Local functions --------------------------
function yPred = ssm_openloop_predict(model, u)
% Open-loop prediction for identified SSM model.
T = numel(u);
z = model.x0(:);
yPred = zeros(T,1);

for k = 1:T
  p = mod(k-1, model.N) + 1;
  yPred(k) = model.C{p} * z;
  z = model.A*z + model.B*u(k);
end
end

function a = fit_arx21(y, u)
% Fit ARX(2,1): y(k) = a1*y(k-1) + a2*y(k-2) + b1*u(k-1) + e(k)
T = numel(y);
if T < 4
  error('fit_arx21:NotEnoughData', 'Need at least 4 samples for ARX(2,1).');
end

Phi = zeros(T-2, 3);
yt  = zeros(T-2, 1);
for k = 3:T
  Phi(k-2,:) = [y(k-1), y(k-2), u(k-1)];
  yt(k-2) = y(k);
end

ridge = 1e-8;
a = (Phi' * Phi + ridge*eye(3)) \ (Phi' * yt);
end

function yPred = predict_arx21(y, u, a)
% One-step recursive ARX prediction using true initial conditions y(1:2).
T = numel(y);
yPred = zeros(T,1);
yPred(1:2) = y(1:2);

for k = 3:T
  yPred(k) = a(1)*yPred(k-1) + a(2)*yPred(k-2) + a(3)*u(k-1);
end
end
