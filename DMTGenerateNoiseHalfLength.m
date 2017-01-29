%%function status = DMTGenerateNoiseHalfLength()
ctr = 1;
x = zeros(37000,300,5);
<<<<<<< HEAD
for filename = 1:6

A = DMTDataProcessQuarter(['NO',num2str(8-filename),'.csv']);
=======
for filename = 1:7
disp(filename)
A = DMTDataProcessQuarter(['NO',num2str(filename),'.csv']);
>>>>>>> 9ac3e67eabdea5dd73a9fe2443e792d20addc07b
[m,~] = size(A);
    for i =1:m-301
        x(ctr,:,:) = A(i:i+299,:);
        ctr=ctr+1;
<<<<<<< HEAD
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
=======
        if ctr>=37001
            break
        end
    end
    if ctr>=37001
        break
    end
end
x = reshape(x,[],1);
%x = reshape(x,[],300,5);
csvwrite('noise.csv',x);
>>>>>>> 9ac3e67eabdea5dd73a9fe2443e792d20addc07b
