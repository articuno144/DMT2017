d = reshape(csvread('desired.csv'),3712,5);
x = reshape(csvread('input.csv'),3712,300,5);
idx_pretrain = 1:3072;
x = x(idx_pretrain,:,:);
d = d(idx_pretrain,:);
x = reshape(x,[],1);
d = reshape(d,[],1);
csvwrite('x_pretrain.csv',x);
csvwrite('d_pretrain.csv',d);