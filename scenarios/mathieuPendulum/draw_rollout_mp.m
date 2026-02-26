%% draw_rollout_mp.m
% *Summary:* Draw the most recent Mathieu pendulum rollout.
%
% This script is intentionally lightweight and mirrors the role of
% draw_rollout_* scripts in other PILCO scenarios.
%
% Expected variables in caller workspace:
%   xx      rollout data [H x (state+obs+u)]
%   latent  latent trajectories cell
%   j/jj    controlled/random trial index
%   dt, x   scenario variables from settings/learn script

T = size(xx,1);
t = (0:T-1)'*dt;

if exist('j','var') && ~isempty(j)
  id = j + J;
  ttl = ['mathieuPendulum controlled trial # ' num2str(id)];
  lat = latent{j};
else
  id = jj;
  ttl = ['mathieuPendulum random trial # ' num2str(id)];
  lat = latent{jj};
end

subplot(3,1,1);
plot(t, lat(1:T,1), 'k', 'LineWidth', 1.2);
ylabel('\theta [rad]'); title(ttl); grid on;

subplot(3,1,2);
plot(t, lat(1:T,2), 'b', 'LineWidth', 1.2);
ylabel('\theta\_dot [rad/s]'); grid on;

subplot(3,1,3);
plot(t, xx(:,end), 'r', 'LineWidth', 1.2);
xlabel('time [s]'); ylabel('u'); grid on;

drawnow;
