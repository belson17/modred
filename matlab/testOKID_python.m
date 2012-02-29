% Test OKID
close all
clear all
clc

saveFiles = 0;
saveDir = '../tests/files_okid/MIMO/';

mkdir(saveDir);

numInputs = 2;
numOutputs = 3;
numStates = 8;
sys = drss(numStates, numOutputs, numInputs);
% A = eye(numStates)*.5;
% B = ones(numStates,numInputs);
% C = ones(numOutputs, numStates);
% D = ones(numOutputs, numInputs)*2;
% sys = ss(A,B,C,D,1);

nt = 50;
t = (0:nt-1)';
r = 20;

% save_mat_text(sys.a, sprintf('%sA.txt', saveDir), ' ');
% save_mat_text(sys.b, sprintf('%sB.txt', saveDir),' ');
% save_mat_text(sys.c, sprintf('%sC.txt', saveDir),' ');
% save_mat_text(sys.d, sprintf('%sD.txt', saveDir),' ');

% A=load_mat_text(sprintf('%sA.txt', saveDir));
% B=load_mat_text(sprintf('%sB.txt', saveDir)));
% C=load_mat_text(sprintf('%sC.txt', saveDir)));
% D=load_mat_text(sprintf('%sD.txt', saveDir)));
% sys=ss(A,B,C,D,1);

u = [randn(numInputs, 2*nt/4) zeros(numInputs, 2*nt/4)];
y = lsim(sys, u, t)';

[H,M] = OKID(y, u, r);
MarkovsTrue = impulse(sys, t);
MarkovsTrue = permute(MarkovsTrue, [2, 3, 1]);

% Reshape H into a 3D array, [output, input, time]
Markovs = zeros(numOutputs,numInputs,r);
for ir = 1:r
    Markovs(:,:,ir) = H(:,1+(ir-1)*numInputs:ir*numInputs);
end

figure('Position',[100 100 1400 900])
for iIn = 1:numInputs
    for iOut = 1:numOutputs
        subplot(numOutputs, numInputs, iIn+(iOut-1)*numInputs)
        plot(t, squeeze(MarkovsTrue(iOut, iIn, :)),'k')
        hold on
        %plot(H(iOut,iIn:numInputs:end),'b')
        plot(0:r-1, squeeze(Markovs(iOut,iIn,:)),'b--')
        if iIn==1 && iOut == 1
            legend('True','OKID')
        end
        title(sprintf('Input %d to output %d', iIn, iOut))
    end
end

if saveFiles == 1
    save_mat_text(squeeze(u), sprintf('%sinputs.txt', saveDir), ' ')
    save_mat_text(squeeze(y), sprintf('%soutputs.txt', saveDir), ' ');
    
    for iOut=1:numOutputs
        save_mat_text(squeeze(Markovs(iOut,:,:)), ...
            sprintf('%sMarkovs_Matlab_output%d.txt', saveDir, iOut),' ');
        save_mat_text(squeeze(MarkovsTrue(iOut,:,:)), ...
            sprintf('%sMarkovs_true_output%d.txt', saveDir, iOut),' ');
    end
end

fprintf('Max diff between truth and OKID estimate is %g\n',amax(Markovs-MarkovsTrue(:,:,1:length(Markovs))))
