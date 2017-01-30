function status = DMTUIGenerateSetNoNoiseHalfLength()
%Generate all datasets of size b*300*5, where b is the batch size =
%numData*64
numData = 320;
ctr=1;
d = zeros(numData,6);
x = zeros(numData,300,5);
for j = 1:numData/64
filename = ['myt',num2str(j-1),'.csv'];
[~,l_filename] =size(filename); 
A = DMTDataProcessQuarter(filename);
plot(1:length(A(:,4)'),A(:,4)',1:length(A(:,5)'),A(:,5)')
p = ginput(8);
p = floor(p(:,1));
pos = zeros(1,64);
for i = 1:8
    pos(i:8:64-1+i) = p-1+i;
end
for i = 1:64
    try
    x(ctr,:,:) = A(pos(i)-299:pos(i),:);
    d(ctr,1+str2num(filename(l_filename-4)))=1;
    ctr=ctr+1;
    catch
    end
end
end
x = reshape(x,[],1);
%x = reshape(x,[],300,5);
csvwrite('input_myt0130t.csv',x);
d = reshape(d,[],1);
%d = reshape(d,[],6);
csvwrite('desired_myt0130t.csv',d);
status = 'Done';