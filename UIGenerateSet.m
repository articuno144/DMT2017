function status = UIGenerateSet(filename)
%Takes in filename string (without '.csv'),generate single datasets of length
%60, corresponding to 60ms.
A = DataProcess(['data\',filename,'.csv']);
%[r,~]  = size(A);
A = A';
l = 59;
plot(A)
p = floor(ginput(10));
p = p(:,1)';
disp(p)
pos = zeros(1,50);
for j = 1:5
    pos(j:5:45+j) = p-1+j;
end
for i = 1:50
    B = A(pos(i)-l:pos(i));
    file = ['data\',filename,'_',int2str(i),'.csv'];
    csvwrite(file,B');
end
status = 'Done';