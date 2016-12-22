k = 25000; %number of iterations

% initialise
a = 0.00001;
inputMatrix = csvread('data\Merged.csv');
x = inputMatrix(1:end-1,:);
y = inputMatrix(end,:);
[n,p] = size(x); % n = input size, p = batch size
h = 30; % hidden size
w1 = rand(h,n+1)-0.5;
w2 = rand(h,h+1)-0.5;
w3 = rand(h,h+1)-0.5;
w4 = rand(1,h+1)-0.5;
E = zeros(1,k);
xpb = zeros(n+1,p);
xpb(2:n+1,:) = x;
xpb(1,:) = 0.5;
L2pb = zeros(h+1,p);
L3pb = zeros(h+1,p);
L4pb = zeros(h+1,p);

%pre-processing
xpb = xpb/max(max(xpb));

disp(['number of datasets: ',int2str(p)]);
disp(['number of parameters: ',int2str(h*(n+1)+(2h+1)*(h+1))]);

for i = 1:k
    % forward
    L2 = sigmoid(w1*xpb);
    L2pb(2:h+1,:) = L2;
    L2pb(1,:) = 0.5;   
    L3 = sigmoid(w2*L2pb);
    L3pb(2:h+1,:) = L3;
    L3pb(1,:) = 0.5;
    L4 = sigmoid(w3*L3pb);
    L4pb(2:h+1,:) = L4;
    L4pb(1,:) = 0.5;    
    L5 = sigmoid(w4*L4pb);

    % backward
    d5 = L5 - y;
    d4 = w4'*d5.*((1-L4pb).*L4pb);
    d3 = w3'*d4(2:h+1,:).*((1-L3pb).*L3pb);
    d2 = w2'*d3(2:h+1,:).*((1-L2pb).*L2pb);

    % updating weight
    w1 = w1- a*d2(2:h+1,:)*xpb';
    w2 = w2- a*d3(2:h+1,:)*L3pb';
    w3 = w3- a*d4(2:h+1,:)*L4pb';
    w4 = w4- a*d5*L4pb';
    E(i) = -sum(y.*(log(L5+exp(-100))+(1-y).*(log(1-L5+exp(-100)))));
end

plot(E)