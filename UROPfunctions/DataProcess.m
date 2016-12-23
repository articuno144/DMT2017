function C = DataProcess(filename)
%takes the long dataset, shrink to 1/10 length, takes only the first column
A = csvread(filename);
[r,~] = size(A);
B = zeros(r,8);
for i = 1:8
    B(:,i) = A(:,18+2*i)*255 + A(:,19+2*i);
end
C(:,1) = B(1:20:end,1);
C(:,2) = B(1:20:end,8);
plot(1:length(C(:,1)'),C(:,1)',1:length(C(:,1)'),C(:,2)')