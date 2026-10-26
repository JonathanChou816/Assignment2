% Uses the saved experiment to reproduce the reported results in MATLAB.
% Alternatively, uncomment the hw2q2 call below to generate new random data.
clear; close all;
root = fileparts(mfilename('fullpath'));
load(fullfile(root,'results','experiment.mat'),'xTrain','yTrain','xValidate','yValidate');
% rng(2024); [xTrain,yTrain,xValidate,yValidate] = hw2q2(100,1000);

phi = @(x) [ones(size(x,2),1), x(1,:)', x(2,:)', ...
    (x(1,:).^2)', (x(1,:).*x(2,:))', (x(2,:).^2)', ...
    (x(1,:).^3)', (x(1,:).^2.*x(2,:))', ...
    (x(1,:).*x(2,:).^2)', (x(2,:).^3)'];
A = phi(xTrain); B = phi(xValidate);
y = yTrain(:); yv = yValidate(:);
sigma2 = 1; % Working assumption; not supplied by the GMM generator.
wML = A\y;
mseML = mean((yv-B*wML).^2);
gammas = logspace(-8,8,321);
mseMAP = zeros(size(gammas));
weights = zeros(10,length(gammas));
for k = 1:length(gammas)
    lambda = sigma2/gammas(k);
    weights(:,k) = [A; sqrt(lambda)*eye(10)]\[y; zeros(10,1)];
    mseMAP(k) = mean((yv-B*weights(:,k)).^2);
end
[bestMSE,idx] = min(mseMAP);
fprintf('ML validation MSE: %.8f\n',mseML);
fprintf('Best MAP validation MSE: %.8f\n',bestMSE);
fprintf('Best tested gamma: %.9g; lambda: %.9g\n',gammas(idx),sigma2/gammas(idx));
figure;
semilogx(gammas,mseMAP,'LineWidth',1.5); hold on;
semilogx(gammas,mseML*ones(size(gammas)),'--','LineWidth',1.5);
semilogx(gammas(idx),bestMSE,'k*','MarkerSize',10);
xlabel('gamma (working sigma squared = 1)'); ylabel('Validation MSE');
legend('MAP','ML','Best tested MAP'); grid on;
title('Cubic regression: ML versus MAP');
