function [ o_estCls, o_vals ] = testJointBoost_mult( i_params, i_mdls, i_featFunc, i_featFuncParams )
%TESTGENTLEBOOST_APL Summary of this function goes here
%   Detailed explanation goes here

nCls = i_params.nCls;
nData = i_params.nData;

o_estCls = zeros(nData, 1)*nan;
o_vals = zeros(nData, nCls);
% parfor (dInd=1:nData, 24)
for dInd=1:nData
    [o_estCls(dInd), o_vals(dInd, :)] = classifyJB( i_params, i_mdls, i_featFunc, i_featFuncParams, dInd );
end

end

function [o_cls, o_vals] = classifyJB( i_params, i_mdls, i_featFunc, i_featFuncParams, i_dataInd )
M = numel(i_mdls);
nCls = i_params.nCls;

val_max = -inf;
o_cls = nan;
o_vals = zeros(nCls, 1);

for c=1:nCls
    val = 0;
    for m=1:M
        mdl = i_mdls(m);

        delta_pos = i_featFunc(i_featFuncParams, i_dataInd, mdl.f)>mdl.theta;
        delta_neg = ~delta_pos;

        if mdl.S(c)
            val = val + mdl.a*delta_pos + mdl.b*delta_neg;
        else
            val = val + mdl.kc(c);
        end
    end
    o_vals(c) = val;
    
    if val_max < val
        val_max = val;
        o_cls = c;
    end
end
end