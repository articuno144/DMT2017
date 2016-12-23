function status = DMTUIGenerateSetWithNoise()
%Generate all datasets of size b*600*5, where b is the batch size
cd DMTdata
files = ls;
files = files(3:end,:);
[numData,~] = size(files);
numData = numData*35; %extract 7 sets each each with 5 elements from the data set
d = zeros(numData,1);
x = zeros(numData,600,1);
A = DMTDataProcess(['DMTdata/',filename]);
plot(1:length(A(:,4)'),A(:,4)',1:length(A(:,5)'),A(:,5)')
[r,~]  = size(A);
p = ginput(7);
p = floor(p(:,1));
pos = zeros(1,35);
for j = 1:5
    pos(j:5:35+j) = p-1+j;
end
k = 1:r;
for m = pos
    k(m) = 0;
end
for i = 1:50
    B(1:30) = A(1,pos(i)-l:pos(i));
    B(31:60) = A(2,pos(i) - l:pos(i));
    file = ['data2\CN',int2str(batchNumber),'_',int2str(i),'.csv'];
    csvwrite(file,B');
end
for m = k(min(pos):max(pos))
    if m ~= 0
        disp(m)
    B(1:30) = A(1,m-l:m);
    B(31:60) = A(2,m - l:m);
    file = ['data2\NO_CN',int2str(batchNumber),'_',int2str(m),'.csv'];
    csvwrite(file,B');
    end
end
status = 'Done';