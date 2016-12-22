function status = GenerateNoise()
%Takes in filename string (without '.csv'),generate single datasets of length
%60, corresponding to 60ms.
A = DataProcess('data2\NO2.csv');
%[r,~]  = size(A);
L = length(A);
A = A';
l = 29;
pos = 200:L;
%pos = pos(:,1)';
for i = 1:L-200
    B(1:30) = A(1,pos(i)-l:pos(i));
    B(31:60) = A(2,pos(i)-l:pos(i));
    file = ['NOC_',int2str(i-1),'.csv'];
    csvwrite(['data2\',file],B);
end
status = 'Done';