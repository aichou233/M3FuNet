function [L,W]=Construct_L(X,gnd,knn)
options = [];

options.NeighborMode = 'Supervised';
options.gnd = gnd;
options.WeightMode = 'HeatKernel';
options.knn = knn;
W = constructW(X,options);

aa=sum(W);
DDD=diag(aa);
L=DDD-W;
