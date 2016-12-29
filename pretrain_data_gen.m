d = reshape(csvread('desired.csv'),3712,5);
x = reshape(csvread('input.csv'),3712,300,5);
idx_pretrain = 1:3072;
idx_all = 1:3712;
idx_newgest = [];
idx_newpers = [];

for i = 3073:3712
    if ~isequal(d(i,:),d(i-1,:))
        idx_newgest = [idx_newgest,i];
        if isequal(d(i,:),[1,0,0,0,0])
            idx_newpers = [idx_newpers,i];
        end
    end
end
idx_newgest_test = idx_newgest+24;
idx_pers_train = zeros(1,5*24);
idx_pers_test = zeros(1,5*40);
tmp_idx = 1:24:120;
for i = 1:24
    idx_pers_train(tmp_idx+i-1) = idx_newgest(6:10)+i-1;
end
tmp_idx = 1:40:200;
for i = 1:40
    idx_pers_test(tmp_idx+i-1) = idx_newgest_test(6:10)+i-1;
end
x2 = zeros(idx_newpers(2)-1,300,5);
d2 = zeros(idx_newpers(2)-1,5);
x2(1:3072,:,:) = x(1:3072,:,:);
x2(3072+1:3072+120,:,:) = x(idx_pers_train,:,:);
x2(3072+121:end,:,:) = x(idx_pers_test,:,:);
d2(1:3072,:) = d(1:3072,:);
d2(3073:3192,:) = d(idx_pers_train,:);
d2(3193:end,:) = d(idx_pers_test,:);
x2 = reshape(x2,[],1);
d2 = reshape(d2,[],1);
csvwrite('x2.csv',x2);
csvwrite('d2.csv',d2);