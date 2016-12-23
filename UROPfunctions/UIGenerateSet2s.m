function status = UIGenerateSet2s(batchNumber)
%Takes in filename string (without '.csv'),generate single datasets of length
%60, corresponding to 60ms.
A = DataProcess('data2\CN3.csv');
[r,~]  = size(A);
A = A';
l = 29;
plot(A(1,:))
% p = zeros(1,10);
% [p(1),~] = ginput(1);
% p(1) = floor(p(1));
% p(1:10) = p(1):100:(p(1)+900);
p = ginput(10);
p = floor(p(:,1));
disp(p)
pos = zeros(1,50);
for j = 1:5
    pos(j:5:45+j) = p-1+j;
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