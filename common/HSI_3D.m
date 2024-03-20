hold on; close all;
ori = data3D_SV; % 输入3维HSI数据
B = size(ori,3);
for i = 1:B
s = surface([0.1*i 0.1*i; 0.1*i 0.1*i], [1 -1; 1 -1], [1 1; -1 -1], ... 
'FaceColor', 'texturemap', 'CData', ori(:,:,i)); 
s.EdgeColor = 'none';
end
% s = surface([0.1*i 0.1*i; 0.1*i 0.1*i], [-1 1; -1 1], [1 1; -1 -1], ... 
% s = surface([0.1*i 0.1*i; 0.1*i 0.1*i], [1 -1; 1 -1], [1 1; -1 -1], ... 
axis off;
view(3);