function [confusion, accuracy, TPR, FPR,sample_num,kappa] = confusion_matrix_wei(class, c)
%
% class is the result of test data after classification
%          (1 x n)
%
% c is the label for testing data
%          (1 x len_c)
%
%

class = class.';
c = c.';

n = length(class);
c_len = length(c);

% if n ~= sum(c)
%     disp('WRANING:  wrong inputting!');
%     return;
% end


% confusion matrix
confusion = zeros(c_len, c_len);
sample_num = cell(c_len, c_len);
a = 0;
for i = 1: c_len
    b = 1;
    for j = (a + 1): (a + c(i))
        confusion(i, class(j)) = confusion(i, class(j)) + 1;
        if class(j)~=i
            sample_num{i,class(j)}(b)= j;
            b = b + 1;
        end
    end
    a = a + c(i);
end


% True_positive_rate + False_positive_rate + accuracy
TPR = zeros(1, c_len);
FPR = zeros(1, c_len);
sum_1 = 0;
sum_2 = 0;
for i = 1: c_len
  FPR(i) = confusion(i, i)/sum(confusion(:, i));
  TPR(i) = confusion(i, i)/sum(confusion(i, :));
  sum_1 = sum_1+confusion(i, i);
  sum_2 = sum_2+(sum(confusion(i, :))*sum(confusion(:, i)));
end
accuracy = sum(diag(confusion))/sum(c);

po = sum_1/n;
pe = sum_2/(n*n);
kappa = (po-pe)/(1-pe);
