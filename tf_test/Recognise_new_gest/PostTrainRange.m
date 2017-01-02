for i = 1:5
    x = zeros(5);
    A = DMTDataProcessQuarter(['sample9gesture',num2str(i),'.csv']);
    plot(1:length(A(:,4)'),A(:,4)',1:length(A(:,5)'),A(:,5)')
    p = ginput(1);
    p = floor(p(1,1));
    x = A(1:p,:);
    csvwrite(['s9g',num2str(i),'.csv'],x);
end