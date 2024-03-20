function convImage = SingleConvolution( image, filter )
%功能：
%   单通道图像的卷积运算
%输入参数： 
%   image：扩展后的图像
%   filter：滤波器
%输出参数： 
%   imageConv：卷积后的图像
[Height, Width] = size( image );
[mF, nF] = size( filter );
for i = 1 : Height - mF + 1
    for j = 1 : Width - nF + 1
        localImage = [];
        localImage = image(i:i+mF-1, j:j+nF-1);
        convImage(i, j) = sum( sum( localImage .* filter ) );
    end
end
convImage = uint8( convImage );
end