function [Z, rmse_final, rmse_total_final,  mae, mae_total] = STH_LRTC(veh, truth, q, tau, opt)
%% STH_LRTC: spatiotemporal Hankel tensor with low-rank tensor completion

% Based on Tensor completion for estimating missing values in visual data (HaLRTC) @Ji Liu

% Input:
% veh: spatiotemporal matrix with missing values (H = mat2hankel(veh, tau, order))
% truth: ground truth matrix (veh = V.*q)
% q: index matrix (1: training data position, 0: testing data position)
% tau: delay embedding lengths (tau = [tau_s, tau_t])



% Output:
% Z: estimated full spaiotemporal matrix



rho = opt.rho;
plotf = opt.plotf;
max_iter = opt.max_iter;
beta = opt.beta;
max_rho = opt.max_rho;
tol = opt.tol;
sizeh = opt.sizeh;
theta = opt.theta;
seed = opt.seed;
[height, width] = size(veh);



% matrix to Hankel tensor
dim = sizeh; 
idx = ~q(:);        % testing index
normZ = norm(veh(:));

E = veh;
Z = veh;

tic
for iter = 1 : max_iter
    % update Xtensor
    Ztensor = mat2hankel(Z, tau); 
    Etensor = mat2hankel(E, tau); 
    
    Xtemp = reshape(Ztensor-Etensor/rho,[dim(1)*dim(2),dim(3)*dim(4)]); % tensor to a balanced matrix
    Xsquare = SVT_TNN(Xtemp, 1/rho,theta);                           % shrinkage singular value decomposition
    Xtensor = reshape(Xsquare,dim);                                            % reshape balanced matrix to tensor

    
    % update Z
    Zold = Z;
    X = hankel2mat(Xtensor);
    Zest = X - E/rho;
    Z = veh + Zest.* ~q;  % Put the exact observed to those positions
    
    % update E
    E = E + rho * (X - Z);
    
    % update rho
    rho = min (beta * rho, max_rho);
    
    % check convergence
    res(iter) = norm( Z(:) - Zold(:) )/ normZ ;
    
    
    % error on testing data   
    rmse(iter) = sqrt(norm(Z(idx) - truth(idx), 'fro' )^2/(sum(idx)));
    rmse_total(iter) = sqrt(norm(Z - truth, 'fro' )^2/(height*width));
    
%     if iter == 1 || mod(iter, 1) == 0
%         disp(['seed=' num2str(seed) ' tau= ' num2str(tau) ' theta= ' num2str(theta) ' iter= ' num2str(iter) ...
%             ', rho=' num2str(rho) ', rmse_test=' num2str(rmse(iter)) ', res=' num2str(res(iter))]);
%     end
    
    % plot figure
    if plotf && mod(iter,1) == 0
        subplot(311)
        imagesc(truth)
        colorbar
        caxis([0 80])
        
        subplot(312)
        imagesc(veh)
        colorbar
        caxis([0 80])
        
        subplot(313)
        imagesc(Z)
        caxis([0 80])
        colorbar
        drawnow()
    end
    
    if res(iter) < tol
        break
    end
end

% toc
rmse_final = rmse(iter);
rmse_total_final = rmse_total(iter);
mae = sum(abs(Z(idx) - truth(idx)),'all')/sum(idx);
mae_total = sum(abs(Z - truth),'all')/(height*width);

