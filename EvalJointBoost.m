function [o_cls, o_vals] = EvalJointBoost( i_x, i_mdls, i_params )


%% init
M = numel(i_mdls);
assert(M == i_params.nWeakLearner);
nCls = i_params.nCls;

featDim = i_params.featDim;
assert(featDim == size(i_x, 2));
nData = size(i_x, 1);

%% eval
Hs = zeros(nData, nCls);
for m=1:M
    mdl = i_mdls(m);
    hs = geths(nData, nCls, i_x(:, mdl.f) > mdl.theta, mdl);
    Hs = Hs + hs;
end

%% return
[~, o_cls] = max(Hs, [], 2);
o_vals = Hs;
end

function [o_hs] = geths(i_nData, i_nCls, i_delta_pos, i_mdl) %%FIXME: should be consistent with one in the TrainJointBoost.m

%% reconstruct hs
o_hs = bsxfun(@times, ones(i_nData, i_nCls), i_mdl.kc);
o_hs(:, i_mdl.S) = repmat(i_mdl.a*i_delta_pos + i_mdl.b*(~i_delta_pos), [1, sum(i_mdl.S)]);

end

