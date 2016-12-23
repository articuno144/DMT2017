function status = DMTUIGenerateSetNoNoise()
%Generate all datasets of size b*600*5, where b is the batch size =
%numData*64
cd DMTdata
files = ls;
files = files(3:end,:);
[numData,~] = size(files);
numData = numData*64; %extract 8 sets each each with 8 elements from the data set
ctr=1;
cd ..
for j = 1:numData
filename = files(j,:);
[~,l_filename] =size(filename); 
d = zeros(numData,5);
x = zeros(numData,600,5);
A = DMTDataProcess(['DMTdata/',filename]);
plot(1:length(A(:,4)'),A(:,4)',1:length(A(:,5)'),A(:,5)')
p = ginput(8);
p = floor(p(:,1));
pos = zeros(1,64);
for i = 1:8
    pos(i:8:64-1+i) = p-1+i;
end
for i = 1:64
    x(ctr,:,:) = A(:,pos(i)-600:pos(i));
    d(ctr,str2num(filename(l_filename-4)))=0;
end
end
status = 'Done';