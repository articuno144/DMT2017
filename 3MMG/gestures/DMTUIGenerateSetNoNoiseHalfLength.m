function status = DMTUIGenerateSetNoNoiseHalfLength()
%Generate all datasets of size b*300*5, where b is the batch size =
%numData*64
cd raw
files = ls;
files = files(5:end,:);
[numData,~] = size(files);
numData = numData*128; %extract 16 sets each each with 8 elements from the data set
ctr=1;
cd ..
d = zeros(numData,9);
x = zeros(numData,300,6);
for j = 1:numData/128
filename = files(j,:);
[~,l_filename] =size(filename); 
A = DataProcess(['raw/',filename]);
PlotData(A)
p = ginput(16);
p = floor(p(:,1));
pos = zeros(1,128);
for i = 1:16
    pos(i:8:128-1+i) = p-1+i;
end
for i = 1:128
    try
    x(ctr,:,:) = A(pos(i)-299:pos(i),:);
    d(ctr,str2num(filename(l_filename-4)))=1;
    ctr=ctr+1;
    catch
    end
end
cla
end
x = reshape(x,[],1);
%x = reshape(x,[],300,6);
csvwrite('input.csv',x);
d = reshape(d,[],1);
%d = reshape(d,[],9);
csvwrite('desired.csv',d);
status = 'Done';