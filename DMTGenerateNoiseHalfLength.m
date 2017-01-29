%%function status = DMTGenerateNoiseHalfLength()
ctr = 1;
x = zeros(37000,300,5);
for filename = 1:6

A = DMTDataProcessQuarter(['NO',num2str(8-filename),'.csv']);
[m,~] = size(A);
    for i =1:m-301
        x(ctr,:,:) = A(i:i+299,:);
        ctr=ctr+1;
        if ctr>=37000
            break
        end
    end
    if ctr>=37000
        break
    end
disp(filename)
end
x = reshape(x,[],1);
%x = reshape(x,[],300,5);
csvwrite('noisemyt2.csv',x);