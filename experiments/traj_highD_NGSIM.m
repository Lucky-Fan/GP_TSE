clear,clc
close all
load('..//data//HighD//mat//highD_full.mat')
full_speed = full;
seed = 3000;
hal.rho = 1e-6;
hal.max_rho = 1;
hal.max_iter = 200;
hal.beta = 1.1;
hal.tol = 0.001;
hal.plotf = 1;
hal.theta = 6;
hal.seed = seed;

for mr=[0.05,0.1,0.2,0.3,0.4,0.5]
% for mr =[0.05, 0.1]
%     for i = 1
    for i=0:1:9
        load(strcat('..//data//HighD//mat//highD_', num2str(mr),'_', num2str(i),'.mat'))
        veh = zeros(size(s));  % The observed value
        veh(s>0) = s(s>0);
        q = (veh>0);  % The mask array
        [N,T] = size(s);
        % colormap
        cm_jet= flipud(jet);
        cm = flipud(jet);
        cm_jet(1,:) = 1;            % speed 0 = white

        % STH-LRTC parameters
        tau = [40,30];
%         tau = [50, 50];
        sizeh = [tau N-tau(1)+1 T-tau(2)+1];
        hal.sizeh = sizeh;

        tic
        [mat_hat, rmse, rmse_total, mae, mae_total] = STH_LRTC(veh, full_speed, q, tau, hal);
        toc

        fprintf('missing: %.2f, iter: %d, rmse: %.4f, rmse_total: %.4f, mae: %.4f, mae_total:%.4f \n',...
            mr, i, rmse, rmse_total, mae, mae_total);
%         save(strcat('highD_hat_mat','_', num2str(mr),'_',num2str(i),'.mat'),'mat_hat','rmse','rmse_total','mae','mae_total')
    end
end


