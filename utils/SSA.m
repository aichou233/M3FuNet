%==========================================================================
% Zabalza Jaime, Jinchang Ren, Zheng Wang, Marshall Stephen, Jun Wang.Singular Spectrum Analysis for Effective Feature Extraction in Hyperspectral Imaging

% Input:
% L              ！！ The embedding window L
% x              ！！ The input vector

% Output:
% Y              ！！ Feature vector after SSA
%==========================================================================


function [Y]=SSA(L,x)

%embedding
N=length(x);
if L>N/2     
    L=N-L;
end
K=N-L+1;
X=zeros(L,K);  
for i=1:L
    X(i,1:K)=x(i:K+i-1);
end

%SVD
S=X*X';
[U,autoval]=eigs(S,1);
[~,l]=sort(-diag(autoval));
U=U(:,l);
V=(X')*U;

%grouping
I=[1:1];
Vt=V';
rac_X=U(:,I)*Vt(I,:);  

%diagonal averaging
Y=zeros(1,N);
for n=1:L
    for j=1:n
        Y(:,n)=Y(:,n)+(1/n)*rac_X(j,n-j+1);
    end
end
for n=L+1:K-1
    for j=1:L
        Y(:,n)=Y(:,n)+(1/L)*rac_X(j,n-j+1);
    end
end
for n=K:N
    for j=n-K+1:L
        Y(:,n)=Y(:,n)+(1/(N-n+1))*rac_X(j,n-j+1);
    end
end

