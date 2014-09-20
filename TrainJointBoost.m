function [ o_mdl ] = TrainJointBoost( i_xs, i_ys, i_params, i_x_meta )
% 
%   train JointBoost
%   
% ----------
%   Input: 
% 
%       i_xs        a data or a function handle. n x f matrix, where n is the number of data (i_params.nData) and 
%                   f is the number of a feature (i_params.featDim)
% 
%       i_ys        labels of each data. Non-negative integer number. 0 is the bg.
% 
%       i_params    parameters of the JointBoost algorithm
%           i_params.nWeakLearner       the number of a weak learner
%           i_params.nCls               the number of a class except for the bg
%           i_params.nData
%           i_params.featDim            the number of a feature
%           i_params.featSelRatio       feature sampling ratio. [0, 1].
%           i_params.featValRange       the range of each feature. Usually
%                                       0:0.1:1
%           i_params.verbosity          verbosity level. 0: silent
% 
%       i_x_meta    (optional) meta information of i_xs when i_xs is a function
%                   handle. Need this due to Matlab Coder.
% 
% ----------
%   Output:
% 
%       o_mdl:          i_params.nWeakLearner number of weak learners    
% 
% ----------
%   DEPENDENCY:
%   
%
% ----------
% Written by Sangdon Park (sangdonp@cis.upenn.edu), 2014.
% All rights reserved.
%

%% init
assert(all(i_ys>=0));
nWeakLearner = i_params.nWeakLearner;
nData = i_params.nData;
nCls = i_params.nCls;
verbosity = i_params.verbosity;

if nargin == 3
    i_x_meta = [];
end
    

zs = ones(nData, nCls)*(-1);
% Hs = zeros(nData, nCls);
ws = ones(nData, nCls);

mdls = struct('a', 0, 'b', 0, 'f', 0, 'theta', 0, 'kc', zeros(1, nCls), 'S', false(1, nCls));
mdls = repmat(mdls, [nWeakLearner, 1]);

% init labels
for dInd=1:nData
    if i_ys(dInd) == 0 % bg
        continue;
    end
    zs(dInd, i_ys(dInd)) = 1;
end

%% train weak classifiers
for m=1:nWeakLearner
    if verbosity >= 1
        fprintf('* boosting iter: %d/%d...', m, nWeakLearner);
    end
            
    [mdls(m), hs] = FitStumpForAllS(i_xs, zs, ws, i_params, i_x_meta);
%     Hs = Hs + hs; % don't need to calc.
    ws = ws.*exp(-zs.*hs);
    
    if verbosity >= 1
        fprintf('J_wse = % 12.06f\n', Jwse(ws, zs, hs));
    end
end
    
%% return
o_mdl = mdls;
end

function [o_cost] = Jwse(i_ws, i_zs, i_hs)
o_cost = sum(sum(i_ws.*(i_zs - i_hs).^2));
end

function [o_hs] = geths(i_nData, i_nCls, i_delta_pos, i_mdl) %%FIXME: should be consistent with one in the EvalJointBoost.m

%% reconstruct hs
o_hs = bsxfun(@times, ones(i_nData, i_nCls), i_mdl.kc);
o_hs(:, i_mdl.S) = repmat(i_mdl.a*i_delta_pos + i_mdl.b*(~i_delta_pos), [1, sum(i_mdl.S)]);

end


function [o_mdl, o_hs] = FitStumpForAllS(i_xs, i_zs, i_ws, i_params, i_x_meta)

%% init
featDim = i_params.featDim;
featValRange = i_params.featValRange;
nData = i_params.nData;
nCls = i_params.nCls;
featSelRatio = i_params.featSelRatio;

mdl_init = struct('a', 0, 'b', 0, 'f', 0, 'theta', 0, 'kc', zeros(1, nCls), 'S', false(1, nCls));

%% n* = argmin_n Jwse(n)
Jwse_best = inf;
mdl_best = mdl_init;
hs_best = [];

% greedly select S(n) 
S = false(1, nCls);
for totSize=1:nCls
    for candInd=find(~S)
        % choose a candidate S
        curS = S;
        curS(candInd) = true;
        
        % estimate k, which is independent on f, and theta
        % fit a stump. Find a weak learner given a S
        Jwse_S_best = inf;
        mdl_S_best = mdl_init;
        hs_S_best = [];
        for fInd=1:featDim
            if rand(1) > featSelRatio
                continue;
            end
            wz_S = i_ws(:, curS).*i_zs(:, curS);
            wz_nS = i_ws(:, ~curS).*i_zs(:, ~curS);
            kc = ones(1, nCls)*nan;
            kc(~curS) = sum(wz_nS, 1)./sum(i_ws(:, ~curS), 1); 
            for tInd=1:numel(featValRange)
                curTheta = featValRange(tInd);
                
                % estimate a and b
                if isa(i_xs, 'function_handle')
                    delta_pos = i_xs(1:nData, fInd, i_x_meta) > curTheta;
                else
                    delta_pos = i_xs(:, fInd) > curTheta;
                end
                
                a = sum(sum(bsxfun(@times, wz_S, delta_pos), 1))/sum(sum(bsxfun(@times, i_ws(:, curS), delta_pos), 1));
                b = sum(sum(bsxfun(@times, wz_S, ~delta_pos), 1))/sum(sum(bsxfun(@times, i_ws(:, curS), ~delta_pos), 1));
                
                % mdl
                mdl = struct('a', a, 'b', b, 'f', fInd, 'theta', curTheta, 'kc', kc, 'S', curS);
                
                % calc cost 
                hs_S_f_t = geths(nData, nCls, delta_pos, mdl);
                Jwse_S_f_t = Jwse(i_ws, i_zs, hs_S_f_t);
                
                % keep the best
                if Jwse_S_best > Jwse_S_f_t
                    Jwse_S_best = Jwse_S_f_t;
                    mdl_S_best = mdl; 
                    hs_S_best = hs_S_f_t;
                end
            end
        end
        
        % keep the best
        if Jwse_best > Jwse_S_best
            Jwse_best = Jwse_S_best;
            mdl_best = mdl_S_best;
            S = curS;
            hs_best = hs_S_best;
        end
    end
end

%% return
o_mdl = mdl_best;
o_hs = hs_best;
end

