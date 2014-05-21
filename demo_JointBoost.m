nCls = 3;
nDataPerCls = [100; 100; 100; 100; 100];
mus = [0 0; 0 1; 1 0; -1 2; 1 2];
sigs = ones(nCls, 1)*0.01;
train_cols = {'rx' 'gx' 'bx' 'kx' 'yx'};
test_cols = {'rs' 'gs' 'bs' 'ks' 'ys'};
trainRatio = 0.5;
libDir = 'libs';
%% init

dataDim = 2;
featDim = dataDim;
labels = [];
data = [];
for c=1:nCls
    labels = [labels; ones(nDataPerCls(c), 1)*c];
%     data = [data; max(repmat([0, 0], nDataPerCls(c), 1), min(repmat([1 1], nDataPerCls(c), 1), mvnrnd(mus(c, :), eye(dataDim)*sigs(c), nDataPerCls(c))))];
    data = [data; mvnrnd(mus(c, :), eye(dataDim)*sigs(c), nDataPerCls(c))];
end

nData = size(data, 1);

rpInd = randperm(nData);
trInd = rpInd(1:round(nData*trainRatio));
teInd = rpInd(round(nData*trainRatio)+1:end);

addpath(genpath(libDir));

%% train
disp('- train...');
jointBoostParams = [];
jointBoostParams.nWeakLearner = 100;
jointBoostParams.featDim = featDim;
jointBoostParams.rndFeatSelRatio = 1;
jointBoostParams.featValRange = -5:0.1:5;
jointBoostParams.nData = numel(trInd);
jointBoostParams.nCls = nCls;

mdls = trainJointBoost_mult( ...
    jointBoostParams, ...
    @(funcParams, featInd) data(trInd, featInd), ...
    [], ...
    labels(trInd), []);


%% test
disp('- test...');
jointBoostParams.nData = numel(teInd);
[estCls, vals] = testJointBoost_mult(...
    jointBoostParams, ...
    mdls, ...
    @(funcParams, dataInd, featInd) data(teInd(dataInd), featInd), ...
    []);


%% show
figure(1001); clf;

% train data
for c=1:nCls
    ind = intersect(trInd, find(labels == c));
    plot(data(ind, 1), data(ind, 2), train_cols{c}); hold on;
end

% % train data
% ind = intersect(trInd, find(labels == 1));
% plot(data(ind, 1), data(ind, 2), 'rx'); hold on;
% 
% ind = intersect(trInd, find(labels == 2));
% plot(data(ind, 1), data(ind, 2), 'gx'); hold on;
% 
% ind = intersect(trInd, find(labels == 3));
% plot(data(ind, 1), data(ind, 2), 'bx'); hold on;


% test data
for c=1:nCls
    ind = intersect(teInd, teInd(estCls == c));
    plot(data(ind, 1), data(ind, 2), test_cols{c}); hold on;
end


% % test data
% ind = intersect(teInd, teInd(estCls == 1));
% plot(data(ind, 1), data(ind, 2), 'rs'); hold on;
% 
% ind = intersect(teInd, teInd(estCls == 2));
% plot(data(ind, 1), data(ind, 2), 'gs'); hold on;
% 
% ind = intersect(teInd, teInd(estCls == 3));
% plot(data(ind, 1), data(ind, 2), 'bs'); hold on;

% legend('train data of class 1', 'train data of class 2', 'train data of class 3', ...
%     'test data of class 1', 'test data of class 2', 'test data of class 3');
