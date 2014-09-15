%% data settings
nCls = 2;
featDim = 2;
nDataPerCls = [100; 100]; %100*ones(nCls, 1);
mus = [0 0; 0 1; 1 0; -1 2; 1 2];
sigs = ones(nCls, 1)*0.1;
train_cols = {'rx' 'gx' 'bx' 'kx' 'yx'};
test_cols = {'rs' 'gs' 'bs' 'ks' 'ys'};
trainRatio = 0.5;

%% init
labels = cell(nCls, 1);
labels_tr = cell(nCls, 1);
labels_te = cell(nCls, 1);

data = cell(nCls, 1);
data_tr = cell(nCls, 1);
data_te = cell(nCls, 1);
for c=1:nCls
    labels{c} = ones(nDataPerCls(c), 1)*c;
    data{c} = mvnrnd(mus(c, :), eye(featDim)*sigs(c), nDataPerCls(c));
    
    rpInd = randperm(size(data{c}, 1));
    nTr = round(size(data{c}, 1)*trainRatio);
    trInd = rpInd(1:nTr);
    teInd = rpInd(nTr+1:end);
    
    data_tr{c} = data{c}(trInd, :);
    data_te{c} = data{c}(teInd, :);
    
    labels_tr{c} = labels{c}(trInd, :);
    labels_te{c} = labels{c}(teInd, :);
end
data_tr = cell2mat(data_tr);
data_te = cell2mat(data_te);

labels_tr = cell2mat(labels_tr);
labels_te = cell2mat(labels_te);

%% train
disp('* train...');
JBParams = struct(...
    'nWeakLearner', 100, ...
    'nCls', nCls, ...
    'nData', numel(labels_tr), ...
    'featDim', featDim, ...
    'featSelRatio', 1, ...
    'featValRange', -5:0.1:5, ...
    'verbosity', 1);

mdls = TrainJointBoost( ...
    data_tr, ...
    labels_tr, JBParams);

%% test
disp('* test...');
JBParams.nData = numel(teInd);
[estCls, vals] = PredJointBoost(...
    data_te, ...
    mdls, ...
    JBParams);


%% show
figure(1001); clf;

% train data
for i=1:size(data_tr, 1)
    hold on;
    plot(data_tr(i, 1), data_tr(i, 2), train_cols{labels_tr(i)});
    hold off;
end

% test data
for i=1:size(data_te, 1)
    hold on;
    plot(data_te(i, 1), data_te(i, 2), test_cols{estCls(i)});
    hold off;
end

