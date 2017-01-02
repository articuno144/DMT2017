A = csvread('s9g2.csv');
B = csvread('auto_preds.csv');
C = zeros(1,1002);
idx =[];
for i = 1:700
    C(i+300) = B(i,2)>=0.99;
    if B(i,2)>=0.99
        idx = [idx,i+300];
    end
end
hold on
plot(A(:,4));
plot(A(:,5));
plot(C*200);
[k,D] = kmeans(idx',4);
xlabel('time')
ylabel('signal strength')
legend('MMG1','MMG2','gesture recognised')