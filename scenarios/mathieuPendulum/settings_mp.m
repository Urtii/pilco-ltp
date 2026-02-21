%% settings_mp.m
% *Summary:* Scenario settings for a parametrically excited pendulum
% (Mathieu-type LTP system) using the standard PILCO architecture.
%
% The state is minimal with no augmentation:
%   x = [theta; theta_dot]

warning('off','all'); format short; format compact;
rand('seed',11); randn('seed',23);

% Include paths exactly as in standard scenario settings.
try
  rd = '../../';
  addpath([rd 'base'],[rd 'util'],[rd 'gp'],[rd 'control'],[rd 'loss']);
catch
end

% 1) Define state and index sets.
odei = [1 2];      % states propagated through ODE
augi = [];         % no augmentation
dyno = [1 2];      % states used for prediction/loss
dyni = [1 2];      % GP inputs from state only (no trig augmentation)
poli = [1 2];      % policy acts on raw state

% For compatibility with rollout/gTrig internals, no angle expansion is used.
angi = [];

% Learn differences for both state dimensions.
difi = [1 2];

% 2) Scenario horizon and initial uncertainty.
dt = 0.05;            % [s] sample time
Tpred = 4.0;          % [s] prediction horizon duration
H = ceil(Tpred/dt);   % prediction horizon in steps
mu0 = [0.2; 0.0];     % initial mean state
S0 = diag([0.05 0.10].^2);
N = 12;               % policy improvement iterations
J = 2;                % initial random rollouts
K = 1;                % number of start states for optimization
nc = 20;              % controller basis functions

% 3) Plant structure.
plant.dynamics = @dynamics_mp;
plant.noise = diag([0.01^2 0.02^2]);
plant.dt = dt;
plant.ctrl = @zoh;
plant.odei = odei;
plant.augi = augi;
plant.angi = angi;
plant.poli = poli;
plant.dyno = dyno;
plant.dyni = dyni;
plant.difi = difi;
plant.prop = @propagated;

% 4) Policy structure (standard GP+sat controller).
policy.fcn = @(policy,m,s)conCat(@congp,@gSat,policy,m,s);
policy.maxU = 3.0;

% No angle augmentation: initialize directly in raw state space.
policy.p.inputs = gaussian(mu0(poli), S0(poli,poli), nc)';
policy.p.targets = 0.05*randn(nc, length(policy.maxU));

% congp with D=2, U=1 -> hyp length D+2 = 4
% [ell_1 ell_2 sf sn]
policy.p.hyp = log([1 1 1 0.05])';

% 5) Cost structure (quadratic around upright equilibrium).
cost.fcn = @loss_mp;
cost.gamma = 1;
cost.target = [0; 0];
cost.W = diag([20 1]);

% 6) Dynamics model structure (default PILCO GP pipeline unchanged).
dynmodel.fcn = @gp1d;
dynmodel.train = @train;
dynmodel.induce = zeros(300,0,1);
trainOpt = [300 500];

% 7) Policy optimization parameters.
opt.length = 75;
opt.MFEPLS = 30;
opt.verbosity = 1;
opt.method = 'BFGS';

% 8) Plot verbosity.
plotting.verbosity = 1;

% 9) Scenario initializations.
x = []; y = [];
fantasy.mean = cell(1,N); fantasy.std = cell(1,N);
realCost = cell(1,N); M = cell(N,1); Sigma = cell(N,1);
