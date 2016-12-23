function C = DMTDataProcess(filename)
%generate full data, 5 columns: A_x, A_y, A_z, MMG_top, MMG_btm
A = csvread(filename);
[r,~] = size(A);
B = zeros(r,5);
B(:,1:3) = A(:,8:2:12);
B(:,4) = A(:,20)*255 + A(:,21);
B(:,5) = A(:,34)*255 + A(:,35);
C=B;