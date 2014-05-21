function [ o_mdl ] = trainJointBoost_mult( i_params, i_featFunc, i_featFuncParams, i_labels, i_mdl )
%LEARNGENTLEBOOST learn gentleboost using axis-parallel linear functions
%   Detailed explanation goes here

nWeakLearner = i_params.nWeakLearner;
nData = i_params.nData;
nCls = i_params.nCls;
rndFeatSelRatio = i_params.rndFeatSelRatio;

labels = ones(nData, nCls)*(-1);
Hs = zeros(nData, nCls);
ws = ones(nData, nCls);

mdls = struct('a', 0, 'b', 0, 'f', 0, 'theta', 0, 'kc', zeros(1, nCls), 'S', false(1, nCls));
mdls = repmat(mdls, [nWeakLearner, 1]);

%% init labels
for dInd=1:size(labels, 1)
    if isnan(i_labels(dInd))
        continue;
    end
    labels(dInd, i_labels(dInd)) = 1;
end

%% consider previous weak classifiers
for m=1:numel(i_mdl)
    hs = geths(i_params, i_featFunc, i_featFuncParams, i_mdl(m));
    
    Hs = bsxfun(@plus, Hs, hs);
    ws = bsxfun(@times, ws, exp(-bsxfun(@times, labels, hs))); 
end


%% train weak classifiers
for m=1:nWeakLearner
    fprintf('* boosting iter: %d/%d...\n', int32(m), int32(nWeakLearner));
    
    if rndFeatSelRatio == 1 %% what happens on the Matlab Coder! cannot beautify my codes
        [mdls(m), hs] = fitStump(i_params, i_featFunc, i_featFuncParams, labels, ws, 1:i_params.featDim);
    else
        rndFeatInd = randperm(i_params.featDim, max(1, round(i_params.featDim*rndFeatSelRatio)));
        [mdls(m), hs] = fitStump_rnd(i_params, i_featFuncParams, labels, ws, rndFeatInd);
    end
    
    Hs = bsxfun(@plus, Hs, hs);
    ws = bsxfun(@times, ws, exp(-bsxfun(@times, labels, hs))); 
    
end

%% return
if isempty(i_mdl)
    o_mdl = mdls;
else
    o_mdl = [i_mdl; mdls];
end
end

function [o_hs] = geths(i_params, i_featFunc, i_featFuncParams, i_mdl)
nData = i_params.nData;
nCls = i_params.nCls;

%% reconstruct hs
delta_pos = i_featFunc(i_featFuncParams, 1:nData, i_mdl.f)>i_mdl.theta;
o_hs = zeros(nData, nCls);
o_hs(:, i_mdl.S) = repmat(i_mdl.a*delta_pos + i_mdl.b*(~delta_pos), [1, sum(i_mdl.S)]);
o_hs(:, ~i_mdl.S) = repmat(i_mdl.kc(~i_mdl.S), [nData, 1]);

end


function [o_mdl, o_hs] = fitStump(i_params, i_featFunc, i_featFuncParams, i_labels, i_ws, i_featInd)
featValRange = i_params.featValRange;
nData = i_params.nData;
nCls = i_params.nCls;
nMaxThread = 32;

% pre-allocate
mdl_init = struct('a', 0, 'b', 0, 'f', 0, 'theta', 0, 'kc', zeros(1, nCls), 'S', false(1, nCls));
bestCost_S = inf;
bestMdl_S = mdl_init;
besths_S = zeros(nData, nCls);

% pre-calculate features
feats_precalc = zeros(nData, numel(i_featInd));
parfor (fIndInd=1:numel(i_featInd), nMaxThread)
    feats_precalc(:, fIndInd) = i_featFunc(i_featFuncParams, i_featInd(fIndInd));
end

% greadly finding S(n)
S = false(1, nCls);
a_leaf = zeros(1, nCls, numel(i_featInd), numel(featValRange));
b_leaf = zeros(1, nCls, numel(i_featInd), numel(featValRange));
for n=1:nCls    
    bestCost_ft = inf;
    bestMdl_ft = mdl_init;
    besths_ft = zeros(nData, nCls);
    
    for c=1:nCls
        curS = S;
        if curS(c)
            continue;
        else
            curS(c) = true;
        end

        % precalculate for efficiency
        ws_inS = i_ws(:, curS);
        ws_ninS = i_ws(:, ~curS);
        labels_inS = i_labels(:, curS);
        labels_ninS = i_labels(:, ~curS);
        
        % estimate k (out side of the following loops for efficiency)
        ks = zeros(1, nCls)*nan;
        ks(~curS) = sum(ws_ninS.*labels_ninS, 1)./(sum(ws_ninS, 1)+eps);
        
        % precalculate for efficiency
        cost_ninS = sum(sum(ws_ninS.*(bsxfun(@plus, labels_ninS, -ks(~curS))).^2));
        
        % estimate a, b, f, theta
        bestCosts_ft = zeros(numel(i_featInd), 1);
        bestMdls_ft = repmat(mdl_init, [numel(i_featInd), 1]);
        bestths_ft = zeros(numel(i_featInd), 1);
        
        parfor (fIndInd=1:numel(i_featInd), nMaxThread)

            fInd = i_featInd(fIndInd);
            ithfeats = feats_precalc(:, fIndInd);
    
            bestCost_tmp = inf;
            bestMdl_tmp = mdl_init;
            bestth_tmp = inf;
            a_leaf_tmp = a_leaf(1, curS, fIndInd, :);
            b_leaf_tmp = b_leaf(1, curS, fIndInd, :);
            for tInd=1:numel(featValRange)
%                 delta_pos = ithfeats>featValRange(tInd); 
                w_pos = sum(bsxfun(@times, ws_inS, (ithfeats>featValRange(tInd))), 1);
                w_neg = sum(bsxfun(@times, ws_inS, ~(ithfeats>featValRange(tInd))), 1);

                % estimate a and b
                if sum(curS) == 1 % leaf
                    a = sum(sum(ws_inS.*labels_inS, 2).*(ithfeats>featValRange(tInd)))/(sum(sum(ws_inS, 2).*(ithfeats>featValRange(tInd)))+eps);
                    b = sum(sum(ws_inS.*labels_inS, 2).*(~(ithfeats>featValRange(tInd))))/(sum(sum(ws_inS, 2).*(~(ithfeats>featValRange(tInd))))+eps);
                    
                    a_leaf_tmp(1, 1, 1, tInd) = a;
                    b_leaf_tmp(1, 1, 1, tInd) = b;
                else % internal nodes
                    a = sum(a_leaf_tmp(1, :, 1, tInd).*w_pos)/(sum(w_pos)+eps);
                    b = sum(b_leaf_tmp(1, :, 1, tInd).*w_neg)/(sum(w_neg)+eps);
                end

                % calc cost 
                cost_t = (1-a^2)*sum(w_pos) + (1-b^2)*sum(w_neg) + cost_ninS;
                
                % keep the best
                if cost_t < bestCost_tmp
                    bestCost_tmp = cost_t;
                    bestMdl_tmp.a = a;
                    bestMdl_tmp.b = b;
                    bestMdl_tmp.f = fInd;
                    bestMdl_tmp.theta = featValRange(tInd);
                    bestMdl_tmp.kc = ks;
                    bestMdl_tmp.S = curS;
                    
                    bestth_tmp = tInd;
                end
            end
            
            % save
            bestCosts_ft(fIndInd) = bestCost_tmp;
            bestMdls_ft(fIndInd) = bestMdl_tmp;
            bestths_ft(fIndInd) = bestth_tmp;
            
            a_leaf(1, curS, fIndInd, :) = a_leaf_tmp;
            b_leaf(1, curS, fIndInd, :) = b_leaf_tmp;
        end
        
        % find best
        [bestCost_ft, minInd] = min(bestCosts_ft);
        minFeatInd = i_featInd(minInd);
        bestMdl_ft = bestMdls_ft(minInd);
        bestth = bestths_ft(minInd);

        % recalc best hs
        delta_pos = i_featFunc(i_featFuncParams, minFeatInd)>featValRange(bestth);
        delta_neg = ~delta_pos;
        
        hs = zeros(nData, nCls);
        hs(:, curS) = repmat(bestMdl_ft.a*delta_pos + bestMdl_ft.b*delta_neg, [1, sum(curS)]);
        hs(:, ~curS) = repmat(bestMdl_ft.kc(~curS), [nData, 1]);
        besths_ft = hs;
    end
    
    if bestCost_S > bestCost_ft
        bestCost_S = bestCost_ft;
        bestMdl_S = bestMdl_ft;
        besths_S = besths_ft;
        
        S = bestMdl_S.S;
    end
end

%% return
o_mdl = bestMdl_S;
o_hs = besths_S;

end


function [o_mdl, o_hs] = fitStump_rnd(i_params, i_featFuncParams, i_labels, i_ws, i_featInd)

nThread = 32;

% featDim = i_params.featDim;
featValRange = i_params.featValRange;
nData = i_params.nData;
nCls = i_params.nCls;
% rndFeatSelRatio = i_params.rndFeatSelRatio;

mdl_init = struct('a', 0, 'b', 0, 'f', 0, 'theta', 0, 'kc', zeros(1, nCls), 'S', false(1, nCls));
bestCost_S = inf;
bestMdl_S = mdl_init;
besths_S = zeros(nData, nCls);


%% precalc features
% randFeatInd = randperm(featDim, max(1, round(featDim*rndFeatSelRatio)));
redFeatDim = numel(i_featInd);
ithfeats_precomp = zeros(nData, redFeatDim, 'single');
parfor (i=1:redFeatDim, nThread)    
    ithfeats_precomp(:, i) = getithFeat_JB( i_featFuncParams, i_featInd(i) ); % cannot beautify due to the Matlab Coder. It's more efficient
end


% greadly finding S(n)
S = false(1, nCls);
a_leaf = zeros(1, nCls, numel(i_featInd), numel(featValRange));
b_leaf = zeros(1, nCls, numel(i_featInd), numel(featValRange));
for n=1:nCls    
    bestCost_ft = inf;
    bestMdl_ft = mdl_init;
    besths_ft = zeros(nData, nCls);
    for c=1:nCls
        curS = S;
        if curS(c)
            continue;
        else
            curS(c) = true;
        end

        % precalculate for efficiency
        ws_inS = i_ws(:, curS);
        ws_ninS = i_ws(:, ~curS);
        labels_inS = i_labels(:, curS);
        labels_ninS = i_labels(:, ~curS);
        
        % estimate k (out side of the following loops for efficiency)
        ks = zeros(1, nCls)*nan;
        ks(~curS) = sum(ws_ninS.*labels_ninS, 1)./(sum(ws_ninS, 1)+eps);
        
        % precalculate for efficiency
        cost_ninS = sum(sum(ws_ninS.*(bsxfun(@plus, labels_ninS, -ks(~curS))).^2));
        
        % estimate a, b, f, theta                
        bestCosts_ft = zeros(redFeatDim, 1); % parfor friendly code
        bestMdls_ft = repmat(mdl_init, [redFeatDim, 1]); % parfor friendly code
        bestths_ft = zeros(redFeatDim, 1); % parfor friendly code

        parfor (rfInd=1:redFeatDim, nThread)
            fInd = i_featInd(rfInd);
            ithfeats = ithfeats_precomp(:, rfInd);

            bestCost_tmp = inf;
            bestMdl_tmp = mdl_init;
            bestth_tmp = inf;
            a_leaf_tmp = a_leaf(1, curS, rfInd, :);
            b_leaf_tmp = b_leaf(1, curS, rfInd, :);
            for tInd=1:numel(featValRange)
%                 delta_pos = (ithfeats>featValRange(tInd)); % for efficiency

                
                % estimate a and b
                w_pos = sum(bsxfun(@times, ws_inS, (ithfeats>featValRange(tInd))), 1);
                w_neg = sum(bsxfun(@times, ws_inS, ~(ithfeats>featValRange(tInd))), 1);
                if sum(curS) == 1 % leaf
                    a = sum(sum(ws_inS.*labels_inS, 2).*(ithfeats>featValRange(tInd)))/(sum(sum(ws_inS, 2).*(ithfeats>featValRange(tInd)))+eps);
                    b = sum(sum(ws_inS.*labels_inS, 2).*(~(ithfeats>featValRange(tInd))))/(sum(sum(ws_inS, 2).*(~(ithfeats>featValRange(tInd))))+eps);
                    
                    a_leaf_tmp(1, 1, 1, tInd) = a;
                    b_leaf_tmp(1, 1, 1, tInd) = b;
                else
                    a = sum(a_leaf_tmp(1, :, 1, tInd).*w_pos)/(sum(w_pos)+eps);
                    b = sum(b_leaf_tmp(1, :, 1, tInd).*w_neg)/(sum(w_neg)+eps);
                end

                % calc cost 
                cost_t = (1-a^2)*sum(w_pos) + (1-b^2)*sum(w_neg) + cost_ninS;


%                 % estimate a and b
%                 a = sum(sum(ws_inS.*labels_inS, 2).*(ithfeats>featValRange(tInd)))/(sum(sum(ws_inS, 2).*(ithfeats>featValRange(tInd)))+eps);
%                 b = sum(sum(ws_inS.*labels_inS, 2).*(~(ithfeats>featValRange(tInd))))/(sum(sum(ws_inS, 2).*(~(ithfeats>featValRange(tInd))))+eps);
% 
%                 % calc cost 
%                 cost_t = (1-a^2)*sum(sum(bsxfun(@times, ws_inS, (ithfeats>featValRange(tInd))), 1)) + (1-b^2)*sum(sum(bsxfun(@times, ws_inS, ~(ithfeats>featValRange(tInd))), 1)) + cost_ninS;
             
                % keep the best
                if cost_t < bestCost_tmp
                    bestCost_tmp = cost_t;
                    bestMdl_tmp.a = a;
                    bestMdl_tmp.b = b;
                    bestMdl_tmp.f = fInd;
                    bestMdl_tmp.theta = featValRange(tInd);
                    bestMdl_tmp.kc = ks;
                    bestMdl_tmp.S = curS;

                    bestth_tmp = tInd;
                end
            end

            % save
            bestCosts_ft(rfInd) = bestCost_tmp;
            bestMdls_ft(rfInd) = bestMdl_tmp;
            bestths_ft(rfInd) = bestth_tmp;
            
            a_leaf(1, curS, rfInd, :) = a_leaf_tmp;
            b_leaf(1, curS, rfInd, :) = b_leaf_tmp;
        end
        % find best
        [minVal, minInd] = min(bestCosts_ft);
        if minVal < bestCost_ft
            bestCost_ft = minVal;
            bestMdl_ft = bestMdls_ft(minInd);
            bestth = bestths_ft(minInd);
            
            minfInd = i_featInd(minInd);
            delta_pos = getithFeat_JB(i_featFuncParams, minfInd)>featValRange(bestth);
%             delta_pos = ithfeats_precomp(:, minInd) >
%             featValRange(bestth); % Matlab Coder crashes
            hs = zeros(nData, nCls);
            hs(:, curS) = repmat(bestMdl_ft.a*(delta_pos) + bestMdl_ft.b*(~(delta_pos)), [1, sum(curS)]);
            hs(:, ~curS) = repmat(bestMdl_ft.kc(~curS), [nData, 1]);
            besths_ft = hs;
        end
    end
    
    if bestCost_S > bestCost_ft
        bestCost_S = bestCost_ft;
        bestMdl_S = bestMdl_ft;
        besths_S = besths_ft;
        
        S = bestMdl_S.S;
    end
end

%% return
o_mdl = bestMdl_S;
o_hs = besths_S;

end

