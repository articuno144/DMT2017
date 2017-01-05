function status = DMTUIGenerateSetNoNoiseHalfLength()
%Generate all datasets of size b*300*5, where b is the batch size =
%numData*64
x = zeros(160,300,5);
d = zeros(160,5);
ctr = 1;

for j = 1:5
A = DMTDataProcessQuarter(['sample9gesture',num2str(j),'.csv']);
plot(1:length(A(:,4)'),A(:,4)',1:length(A(:,5)'),A(:,5)')
p = ginput(4);
p = floor(p(:,1));
pos = zeros(1,32);
for i = 1:8
    pos(i:8:32-1+i) = p-1+i;
end
for i = 1:32
    try
    x(ctr,:,:) = A(pos(i)-299:pos(i),:);
    d(ctr,j)=1;
    ctr=ctr+1;
    catch
    end
end
end
x = reshape(x,[],1);
%x = reshape(x,[],300,5);
csvwrite('test_input.csv',x);
d = reshape(d,[],1);
%d = reshape(d,[],5);
csvwrite('test_desired.csv',d);
status = 'Done';