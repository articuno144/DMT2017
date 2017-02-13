%%function status = DMTGenerateNoiseHalfLength()

x = zeros(5500,300,5);
A = DMTDataProcessQuarter('myt51.csv');
[m,~] = size(A);
    for i =1:5500
        x(i,:,:) = A(i:i+299,:);
    end
x = reshape(x,[],1);
%x = reshape(x,[],300,5);
csvwrite('noiset.csv',x);   
