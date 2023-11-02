function [X, n, Sigma2] = SVT_TNN(Z, tau, theta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% min: 1/2*||Z-X||^2 + ||X||_tr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% [S, V, D, Sigma2] = MySVDtau(Z, tau);
% V = max(diag(V) - tau, 0);
% n = sum(V > 0);
% X = S(:, 1:n) * diag(V(1:n)) * D(:, 1:n)';

%% new implementation
[m, nd] = size(Z);
if m < nd
    AAT = Z*Z';
   [S, Sigma2, ~] = svd(AAT);
%    [S, Sigma2, ~] = primme_svds(AAT,min(m,nd));
    %[S, Sigma2, ~] = rsvd(AAT, size(AAT,1));
    Sigma2 = diag(Sigma2);
    V = sqrt(Sigma2);
    vec = V;
    vec(theta+1:end) = vec(theta+1:end)-tau;
    n = sum(vec > 0);
    %tol = max(size(Z)) * eps(max(V));
    %n = sum(V > max(tol, tau));
    mid = vec(1:n) ./ V(1:n);
    X = (S(:, 1:n) * diag(mid)) * (S(:, 1:n)' * Z);
    return;
else
    [X, n, Sigma2] = SVT_TNN(Z', tau, theta);
    X = X';
    return;
end



