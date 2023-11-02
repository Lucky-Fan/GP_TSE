clear,clc
close all
load('..//data//NGSIM//mat//speed_grid_full.mat')
seed = 3000;
hal.rho = 1e-6;
hal.max_rho = 1;
hal.max_iter = 200;
hal.beta = 1.1;
hal.tol = 0.001;
hal.plotf = 1;
hal.theta = 6;
hal.seed = seed;


% Construct the observed locations
s = nan*zeros(size(full_speed));
d_loc = [10, 100, 190];
s(d_loc, :) = full_speed(d_loc, :);


veh = zeros(size(s));  % The observed value
veh(s>0) = s(s>0);
q = (veh>0);  % The mask array
[N,T] = size(s);
% colormap
cm_jet= flipud(jet);
cm = flipud(jet);
cm_jet(1,:) = 1;            % speed 0 = white

% STH-LRTC parameters
% tau = [40,30];
tau = [40, 90];
sizeh = [tau N-tau(1)+1 T-tau(2)+1];
hal.sizeh = sizeh;

tic
[mat_hat, rmse, rmse_total, mae, mae_total] = STH_LRTC(veh, full_speed, q, tau, hal);
toc

fprintf('rmse: %.4f, rmse_total: %.4f, mae: %.4f, mae_total:%.4f \n',...
        rmse, rmse_total, mae, mae_total);
save(strcat('detector_NGSIM_LRTC_hat','_', num2str(tau(1)),'_',num2str(tau(2)),'.mat'),'mat_hat','rmse','rmse_total','mae','mae_total')


