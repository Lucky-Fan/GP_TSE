function matrix = hankel2mat(tensor)

[stau, ttau, N, T] = size(tensor);

temp = zeros(N+stau-1, T+ttau-1);
    
    for t = 1:T
        for i = 1:N
            idx1 = i:i+stau-1;
            idx2 = t:t+ttau-1;            
            temp(idx1,idx2) = temp(idx1, idx2) + (tensor(:,:,i,t));           
        end
        
    end
    
    mv = ones(1, N+stau-1)*stau;
    mv(1:stau) = 1:stau;
    mv(end:-1:end-stau+1) = 1:stau;
    
    mv = repmat(mv,[T+ttau-1,1])';
    
    
    mv2 = ones(1, T+ttau-1)*ttau;
    mv2(1:ttau) = 1:ttau;
    mv2(end:-1:end-ttau+1) = 1:ttau;
    
    mv2 = repmat(mv2,[N+stau-1,1]);
    
    matrix = temp./(mv.*mv2);
     
end




