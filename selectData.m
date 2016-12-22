%selects the useful data
cd DMTdata
files = ls;
files = files(3:end,:);

[numData,~] = size(files);
for i = 1:numData
filename = files(i,:);
try
A = csvread(filename);
catch
    delete(filename);
end
[r,~] = size(A);
B = zeros(r,8);
C = zeros(ceil(r/20),2);
for j = 1:8
    B(:,j) = A(:,18+2*j)*255 + A(:,19+2*j);
end
C(:,1) = B(1:20:end,1);
C(:,2) = B(1:20:end,8);
plot(1:length(C(:,1)'),C(:,1)',1:length(C(:,1)'),C(:,2)')
disp(filename);
want = input('is this looking ok?');
if ~want
    delete(filename);
end
hold off
end
cd ..