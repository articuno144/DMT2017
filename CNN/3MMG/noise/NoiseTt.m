%%function status = DMTGenerateNoiseHalfLength()
ctr = 1;
x = zeros(18000,300,6);
cd test
files = ls;
files = files(3:end,:);
[num_noise,~] = size(files);
cd ..
for j = 1:num_noise
filename = files(j,:);
A = DataProcess(['test/',filename]);
[m,~] = size(A);
    for i =1:m-301
        x(ctr,:,:) = A(i:i+299,:);
        ctr=ctr+1;
        if ctr>=18001
            break
        end
    end
    if ctr>=18001
        break
    end
disp(filename)
end
x = reshape(x,[],1);
%x = reshape(x,[],300,6);
csvwrite('noiset.csv',x);
