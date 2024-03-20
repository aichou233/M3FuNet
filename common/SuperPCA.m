function [PC] = SuperPCA(data,num_PC,labels)
% 
[M,N,~]=size(data);
Results_segment= seg_im_class(data,labels); % 获得每一块超像素
Num=size(Results_segment.Y,2);

for i=1:Num
    [P] = Eigenface_f(Results_segment.Y{1,i}',num_PC);
    PC = Results_segment.Y{1,i}*P;
    X(Results_segment.index{1,i},:)=PC;
end
PC = reshape(X,M,N,num_PC);