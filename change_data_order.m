D = csvread('desired.csv');
I = csvread('input.csv');
len = length(D)/5;
D = reshape(D,[],5);
I = reshape(I,[],300,5);
Dnew = D;
Inew = I;
seq = [rand(len,1),[1:len]'];
seq = sortrows(seq);
seq = seq(:,2);
for i = 1:len
    Dnew(i,:) = D(seq(i),:);
    Inew(i,:,:) = I(seq(i),:,:);
end
Dnew = reshape(Dnew,[],1);
Inew = reshape(Inew,[],1);
csvwrite('d_rand_order.csv',Dnew);
csvwrite('i_rand_order.csv',Inew);