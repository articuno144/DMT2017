for n = 5
k = csvread(['s9g',num2str(n),'.csv']);
hold off
hold on
plot(k(:,4));
plot(k(:,5));
[x,y] = ginput();
disp(x,y)
end