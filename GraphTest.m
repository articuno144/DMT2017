
%X = DataProcess('data\YMtest.csv');
X = DataProcess('data2\OP.csv');
l = length(X);
c = zeros(1,l);
for i = 61:l
x = X(i-59:i);
%plot(x)
xpb = zeros(n+1,1);
xpb(2:n+1,1) = x;
xpb(1,:) = 0.5;
xpb = xpb/max(max(xpb));
L2 = sigmoid(w1*xpb);
L2pb = zeros(h+1,1);
L2pb(2:h+1,:) = L2;
L2pb(1,:) = 0.5;
L3 = sigmoid(w2*L2pb);
c(i) = L3;
%pause(0.01)
%clear plot
end
plot(1:l,200*c,1:l,X(1:l))