function [data3D_SV] = SV_handle(data3D,labels)
% data3D ：三维原始高光谱数据
% labels ：超像素分割标记图
% data3D_SV ：基于1-2D转换超向量修正的高光谱特征样本
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[M,N,B]=size(data3D);
seg_num = max(max(labels)); % 分割的区域数
data3D_SV = zeros(M*N,B);
for i = 0:seg_num
    %% 保留指定的标记i分割区域的具体位置
    d_super_pixels = labels; % 超像素分割标签赋值
    % 保留第i超像素区域
    d_super_pixels(find(d_super_pixels==i))=1000;
    no_1000 = find(d_super_pixels~=1000); % 第i超像素非标记位置
    yes_1000 = find(d_super_pixels==1000); % 第i超像素标记位置
    d_super_pixels(no_1000)=0; % 第i超像素非标记位置记为0
    %% 保留三维标记i超像素，生成对应的超向量super_vector
    data_super = data3D;
    for dd = 1:B % 给3D数据去除非i超像素标记区域
        data_temp = data_super(:,:,dd);
%         data_temp(no_1000)=0;
        data_temp(no_1000)=100000;
        data_super(:,:,dd) = data_temp;
    end
    data_super_1D = reshape(data_super,M*N*B,1);
    super_vector = [];pp = 1;
    for tt = 1:M*N*B
        if data_super_1D(tt)~=100000
            super_vector(pp) = data_super_1D(tt);
            pp=pp+1;
        end
    end % 此时的super_vector即为去0后的超向量：1xn double
    %     labels(no_1000)=0;
    %% 处理超向量矩阵super_matrix
    band_samples = size(super_vector,2)/B; % 每个波段贡献的样本点数
    super_matrix = zeros(B,band_samples); % 超向量矩阵
    ss = 1;
    for s = 1:B % 生成超向量矩阵
        super_matrix(s,:) = super_vector(ss:ss+band_samples-1);
        ss = ss+band_samples;
    end
    for s = 1:B % MAD替换离群值
        super_matrix(s,:) = filloutliers(super_matrix(s,:),'linear','movmedian',10); % MAD去除离群值
    end
%     for s = 1:B % MAD替换离群值
%         super_matrix(s,:) = filloutliers(super_matrix(s,:),'linear','movmedian',10); % MAD去除离群值
%     end
%     for s = 1:B % MAD替换离群值
%         super_matrix(s,:) = filloutliers(super_matrix(s,:),'linear','movmedian',10); % MAD去除离群值
%     end

    for s = 1:B % 均值化
        band_mean = mean(super_matrix(s,:));
        super_matrix(s,:) = (super_matrix(s,:)+band_mean)./2;
    end
    for s = 1:B % 均值化
        band_mean = mean(super_matrix(s,:));
        super_matrix(s,:) = (super_matrix(s,:)+band_mean)./2;
    end
    %% 修正极大偏移块
    s_shift_1 = []; % 每个波段的分布主数
    for s = 1:B
        s_shift_1(s) = (mean(super_matrix(s,:))+median(super_matrix(s,:)))/2;
    end
    MAD_shift = abs(0.5*median(abs(s_shift_1-median(s_shift_1))));
    s_shift_2 = filloutliers(s_shift_1,'linear','movmedian',10);% 每个波段的MAD分布主数
%     s_shift = [s_shift_1;s_shift_2]';%可视化两种分布主数区别
    s_shift_3 = s_shift_2 - s_shift_1; % 两者的变化差位置
    shift = []; % 偏离较大的波段数
    for s = 1:B
        if abs(s_shift_3(s))>MAD_shift
            shift = [shift,s];
        end
    end
    % 修正超向量super_vector和超向量矩阵super_matrix
    for j = 1:length(shift) % 用MAD分布主数替换偏离较大的波段
        super_matrix(shift(j),:) = s_shift_2(shift(j));
    end
    ss = 1;
    for s = 1:B
        super_vector(ss:ss+band_samples-1)=super_matrix(s,:);
        ss = ss+band_samples;
    end
    %% 超向量1-2D转换，转换回三维结构的特征样本
    vector_PCA = super_vector;
    ss = 1;
    for s = 1:B
        data3D_SV(yes_1000,s) = vector_PCA(ss:ss+band_samples-1);
        ss = ss+band_samples;
    end
end
    data3D_SV = reshape(data3D_SV,[M,N,B]);
end