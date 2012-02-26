function [H, M] = OKID(y,u,r)
% OKID based on 1991 NASA TM-104069 by Juang, Phan, Horta and Longman
% inputs
%     y: sampled output with dimensions [numOutputs x numSamples] 
%     u: sampled input with dimensions [numInputs x numSamples]
%     r: effective system order, the number of Markov params estimated
% outputs
%     H: Markov parameters
%     M: Observer gain, currently not implemented

% double uppercase UU, YY indicate bold-faced quantities in paper
% single uppercase U, Y indicate script quantities in paper

% Steve Brunton, November 2010.
% Last edited by Brandt Belson, Feb. 2012 to work for MIMO.

% Step 0, check shapes of y,u
yshape = size(y);
q = yshape(1);  % q is the number of outputs 
l = yshape(2);  % L is the number of output samples
ushape = size(u);
m = ushape(1);  % m is the number of inputs
lu = ushape(2); % Lu i the number of input samples 
assert(l==lu);  % L and Lu need to be the same length


% Step 1, p is the number of estimated markov params
p = r;

% Step 2, form data matrices y and V as shown in Eq. (7),
% solve for observer Markov parameters, Ybar
V = zeros(m + (m+q)*p,l);
for i=1:l
    V(1:m,i) = u(1:m,i);
end
for i=2:p+1
    for j=1:l+1-i
        vtemp = [u(:,j);y(:,j)];
        V(m+(i-2)*(m+q)+1:m+(i-1)*(m+q),i+j-1) = vtemp;
%         V((i-1)*(m+q):i*(m+q)-1,i+j-1) = vtemp;
    end
end



Ybar = y*pinv(V, 1e-5);

% Step 3, isolate system Markov parameters H, and observer gain M
D = Ybar(:,1:m);  % feed-through term (or D matrix) is the first term
YbarNoD = Ybar(:,m+1:end);


Ybar1 = zeros(q,m,length(YbarNoD));
Ybar2 = zeros(q,q,length(YbarNoD));
for i=1:p
    Ybar1(:,:,i) = YbarNoD(:,1+(m+q)*(i-1):(m+q)*(i-1)+m);
    Ybar2(:,:,i) = YbarNoD(:,(m+q)*(i-1)+m+1:(m+q)*(i-1)+m+q);
end

Y(:,:,1) = Ybar1(:,:,1) + Ybar2(:,:,1)*D;
for k=2:p
    Y(:,:,k) = Ybar1(:,:,k) + Ybar2(:,:,k)*D;
    for i=1:k-1
        Y(:,:,k) = Y(:,:,k) + Ybar2(:,:,i)*Y(:,:,k-i);
    end
end


size(Y)

H = D;
for k=1:p
    H = [H Y(:,:,k)];
end

% H = Ybar;
M = 0; % not computed yet!