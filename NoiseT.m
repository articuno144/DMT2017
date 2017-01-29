%%function status = DMTGenerateNoiseHalfLength()

x = zeros(5500,300,5);
<<<<<<< HEAD
A = DMTDataProcessQuarter('NO3.csv');
=======
A = DMTDataProcessQuarter('NOT.csv');
>>>>>>> 9ac3e67eabdea5dd73a9fe2443e792d20addc07b
[m,~] = size(A);
    for i =1:5500
        x(i,:,:) = A(i:i+299,:);
    end
x = reshape(x,[],1);
%x = reshape(x,[],300,5);
<<<<<<< HEAD
csvwrite('noiset.csv',x);   
=======
csvwrite('noiset.csv',x);
>>>>>>> 9ac3e67eabdea5dd73a9fe2443e792d20addc07b
