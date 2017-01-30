function C = DMTDataProcessQuarter(filename)
%generate halved data, 5 columns: A_x, A_y, A_z, MMG_top, MMG_btm
A = csvread(filename);
[r,~] = size(A);
B = zeros(ceil(r/4),5);
B(:,1:3) = A(1:4:end,8:2:12);
B(:,4) = A(1:4:end,20)*255 + A(1:4:end,21);
B(:,5) = A(1:4:end,22)*255 + A(1:4:end,23);
C=B;