%selects the useful data
cd raw
files = ls;
files = files(5:end,:);

[numData,~] = size(files);
for i = 1:numData
filename = files(i,:);
try
A = DataProcess(filename);
catch
    delete(filename);
end
PlotData(A)
disp(filename);
want = input('is this looking ok?');
if ~want
    delete(filename);
end
hold off
end
cd ..