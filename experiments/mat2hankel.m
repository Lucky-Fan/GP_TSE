function [H] = mat2hankel(X, tau)
%% delay embedding (matrix to tensor) 
% X: matrix
% tau: delay embedding length


[N,T] = size(X);

stau = tau(1);
ttau = tau(2);

H = zeros(stau, ttau, N-stau+1,T-ttau+1);   % original hankel tensor


for t=1:T-ttau+1
    for k = 1:N-tau+1
        H(:,:,k,t) = X(k:k+stau-1,t:t+ttau-1);
    end
end










  
  
  
  
  
  
  
  
  