%% dynamics_mp.m
% *Summary:* Continuous-time parametrically excited pendulum (Mathieu-type LTP)
%
%   theta_dd + 2*zeta*w0*theta_d + w0^2*(1 + alpha*cos(omega*t))*theta = b*u
%
% State ordering (minimal state, no augmentation):
%   z(1) = theta
%   z(2) = theta_dot
%
% Function interface mirrors dynamics_cp.m:
%   dz = dynamics_mp(t, z, f)
%
% Inputs:
%   t   - current ODE time
%   z   - state [2x1]
%   f   - control signal function handle; typically f(t) from simulate.m
%
% Output:
%   dz  - state derivative [2x1]
%
% Notes:
%   * Deterministic dynamics (no process/measurement noise).
%   * Fully ODE45/RK-compatible.
%   * Parameters are intentionally defined locally for complete isolation.

function dz = dynamics_mp(t, z, f)

% --------- Physical/model parameters (kept local by design) ---------------
w0    = 2.0;   % natural frequency [rad/s]
zeta  = 0.05;  % damping ratio [-]
alpha = 0.35;  % parametric excitation amplitude [-]
omegaHz = 5.3;                % excitation frequency [Hz]
omega   = 2*pi*omegaHz;       % excitation frequency [rad/s]
b     = 1.0;   % input gain [rad/(s^2 * u)]

% --------------------------- Dimension checks ------------------------------
if numel(z) ~= 2
  error('dynamics_mp:InvalidStateDimension', ...
    'Expected state z to be length 2: [theta; theta_dot].');
end

% ------------------------------ Input force --------------------------------
if nargin < 3 || isempty(f)
  u = 0;
elseif isa(f, 'function_handle')
  u = f(t);
else
  % Compatible fallback if caller provides scalar directly.
  u = f;
end

% --------------------------- LTP state-space ODE ---------------------------
% theta_dot   = z2
% theta_ddot  = -2*zeta*w0*z2 - w0^2*(1 + alpha*cos(omega*t))*z1 + b*u

dz = zeros(2,1);
dz(1) = z(2);
dz(2) = -2*zeta*w0*z(2) - (w0^2)*(1 + alpha*cos(omega*t))*z(1) + b*u;
