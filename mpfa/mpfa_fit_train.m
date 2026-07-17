function fit = mpfa_fit_train(data,varargin)
% MPFA_FIT_TRAIN  Fit the dynamic train model to per-stimulus Pr / N / PPR data.
%
%   fit = mpfa_fit_train(data) adjusts the free parameters of MPFA_TRAIN_CORE
%   so the model's per-stimulus release probability (Pr), ready-pool site count
%   (N) and paired-pulse amplitude ratio (PPR = A_k/A_1) best match the target
%   vectors. It uses only base MATLAB (fminsearch), no toolboxes.
%
%   data fields (supply any subset of pr / N / ppr; NaNs mark missing points):
%     .pr      1xK target release probability P_k
%     .N       1xK target apparent site count N_k
%     .ppr     1xK target amplitude ratio A_k / A_1
%     .freqHz  stimulation frequency in Hz (default 20)
%     .Q       quantal 72A-fast iGluSnFR signal in DeltaF/F0 (default 0.5)
%     .label   optional text label carried into the result
%
%   Name/value options:
%     'Weights'   [wPr wN wPPR] relative series weights (default [1 1 1])
%     'Restarts'  random restarts for the local search (default 12)
%     'Verbose'   print progress (default false)
%     'InitialN'/'InitialP' shared first-pulse values to pin (optional)
%     'MinNmaxAbs'/'MinPmaxAbs' lower bounds on absolute asymptotes
%     'RefillMode' 'constant', 'dynamic', or AIC-selected 'auto' (default)
%
%   Returns a struct with fields:
%     .pars       struct(N,pbar,Q)
%     .trainPars  struct with latent N, Pr and evolving-refill controls
%     .model      MPFA_TRAIN_CORE output at the optimum
%     .rmse       struct(pr,N,ppr) per-series root-mean-square error
%     .r2         struct(pr,N,ppr) per-series coefficient of determination
%     .cost       final weighted objective value
%     .consistency fixed-Q PPR implied directly by the target N and Pr
%     .preset     struct ready for mpfa_npq_simulator(preset)
%     .label      echoed label
%
%   Note: apparent N and PPR need not be jointly consistent with a single
%   mechanistic model (see model_info.md), so treat the fit as a weighted
%   best-effort and inspect the per-series R^2.
%
%   See also MPFA_TRAIN_CORE, MPFA_NPQ_SIMULATOR, MPFA_FIT_DEMO.

    opt = struct('Weights',[1 1 1],'Restarts',12,'Verbose',false, ...
        'InitialN',[],'InitialP',[],'MinNmaxAbs',[],'MinPmaxAbs',[], ...
        'RefillMode','auto','RrEvolutionPenalty',1e-3);
    for i = 1:2:numel(varargin)
        opt.(varargin{i}) = varargin{i+1};
    end

    if ~isfield(data,'freqHz') || isempty(data.freqHz), data.freqHz = 20; end
    if ~isfield(data,'Q') || isempty(data.Q), data.Q = 0.5; end
    if ~isfield(data,'label'), data.label = ''; end

    prT  = getfielddef(data,'pr');
    nT   = getfielddef(data,'N');
    pprT = getfielddef(data,'ppr');
    K = max([numel(prT) numel(nT) numel(pprT)]);
    assert(K >= 2,'Need at least two stimuli of target data.');
    prT = padto(prT,K); nT = padto(nT,K); pprT = padto(pprT,K);

    w = opt.Weights;
    % Per-series scale so Pr (~0-1), N (~1-7) and PPR (~1-2) contribute comparably.
    sPr  = nanscale(prT);
    sN   = nanscale(nT);
    sPPR = nanscale(pprT);

    % Parameters: [N1 P1 nMaxFactor tauN refill1 refillInf tauRefill
    %              pMaxFactor tauP] with bounds.
    %   pMaxFactor = Pmax/P1 (release-prob facilitation asymptote), tauP its
    %   rise time constant. Latent Nmax/tauN plus evolving refill drive the
    %   apparent competent-pool N trajectory.
    lb = [1   0.02 1.00 0.25  0.0  0.0 0.25 0.20  0.25];
    ub = [20  0.95 8.00 12.0 15.0 15.0 12.0 3.00 12.00];
    rrCeiling = max(1,1.5 * safemax(nT));
    if isfinite(rrCeiling)
        ub(5:6) = min(ub(5:6),rrCeiling);
    end
    n1Fit = pick(nT,1,4);
    p1Fit = pick(prT,1,0.30);
    if ~isempty(opt.InitialN), n1Fit = opt.InitialN; end
    if ~isempty(opt.InitialP), p1Fit = opt.InitialP; end
    n1Fit = clampv(n1Fit,lb(1),ub(1));
    p1Fit = clampv(p1Fit,lb(2),ub(2));
    if ~isempty(opt.MinNmaxAbs)
        lb(3) = max(lb(3),opt.MinNmaxAbs / max(n1Fit,eps));
    end
    if ~isempty(opt.MinPmaxAbs)
        lb(8) = max(lb(8),opt.MinPmaxAbs / max(p1Fit,eps));
    end
    assert(lb(3) < ub(3) && lb(8) < ub(8), ...
        'Cross-condition asymptote constraint exceeds the fitter bounds.');
    prPlateau = nanmedian(prT(max(1,K-2):K));
    nPlateau = nanmedian(nT(max(1,K-2):K));
    refillLate0 = clampv(nPlateau * prPlateau,0,15);
    x0 = [ n1Fit, p1Fit, ...
           clampv(safemax(nT)/n1Fit,max(1.01,lb(3)),ub(3)), 2.0, 1.0, refillLate0, 1.5, ...
           clampv(prPlateau/p1Fit,max(0.3,lb(8)),ub(8)), 1.5 ];
    if isnan(x0(3)), x0(3) = 1.5; end
    if isnan(x0(6)), x0(6) = 1.0; end
    if isnan(x0(8)), x0(8) = 1.0; end
    x0 = clampvec(x0,lb,ub);

    % N1 and P1 are the directly measured first-pulse values, so pin them to
    % the data (unless FixN1/FixP1 are turned off) and let only the dynamic
    % knobs (Nmax, tauN, evolving refill, Pmax, tauP) be fitted. This decouples the
    % first-point match from the joint tradeoff and keeps the fit robust.
    fixedVal = nan(1,9);
    freeMask = true(1,9);
    if getfielddef(opt,'FixN1',true) && ~isnan(n1Fit)
        fixedVal(1) = n1Fit; freeMask(1) = false;
    end
    if getfielddef(opt,'FixP1',true) && ~isnan(p1Fit)
        fixedVal(2) = p1Fit; freeMask(2) = false;
    end
    x0(~freeMask) = fixedVal(~freeMask);
    mode = lower(char(opt.RefillMode));
    assert(any(strcmp(mode,{'auto','constant','dynamic'})), ...
        'RefillMode must be auto, constant, or dynamic.');
    if strcmp(mode,'auto')
        [pConst,cConst,dConst,kConst] = optimizeMode('constant',x0,lb,ub,fixedVal,freeMask, ...
            opt,data,prT,nT,pprT,w,sPr,sN,sPPR,K);
        [pDyn,cDyn,dDyn,kDyn] = optimizeMode('dynamic',x0,lb,ub,fixedVal,freeMask, ...
            opt,data,prT,nT,pprT,w,sPr,sN,sPPR,K);
        nEff = effectiveObservations(w,prT,nT,pprT);
        aicConst = nEff * log(max(dConst,eps)) + 2*kConst;
        aicDyn = nEff * log(max(dDyn,eps)) + 2*kDyn;
        if aicDyn + 2 < aicConst
            p = pDyn; bestC = cDyn; selectedMode = 'dynamic';
        else
            p = pConst; bestC = cConst; selectedMode = 'constant';
        end
    else
        [p,bestC] = optimizeMode(mode,x0,lb,ub,fixedVal,freeMask, ...
            opt,data,prT,nT,pprT,w,sPr,sN,sPPR,K);
        selectedMode = mode;
        aicConst = NaN; aicDyn = NaN;
    end
    pars = struct('N',p(1),'pbar',p(2),'Q',data.Q,'cvQ',0,'cvP',0);
    trainPars = struct('nStim',K,'freqHz',data.freqHz,'refillPerStim',p(5), ...
        'refillSteadyPerStim',p(6),'tauRefillStim',p(7), ...
        'nMaxFactor',p(3),'tauNStim',p(4),'pMaxFactor',p(8),'tauPStim',p(9));
    model = mpfa_train_core(pars,trainPars);

    fit.pars = pars;
    fit.trainPars = trainPars;
    fit.model = model;
    fit.cost = bestC;
    fit.refillModel = struct('selected',selectedMode,'aicConstant',aicConst,'aicDynamic',aicDyn);
    fit.rmse = struct('pr',rmse(model.pr,prT),'N',rmse(model.readyBefore,nT), ...
        'ppr',rmse(model.releasePpr,pprT));
    fit.r2 = struct('pr',r2(model.pr,prT),'N',r2(model.readyBefore,nT), ...
        'ppr',r2(model.releasePpr,pprT));
    pprFromTargets = fixedQPpr(prT,nT);
    fit.consistency = struct('pprFromTargets',pprFromTargets, ...
        'rmse',rmse(pprFromTargets,pprT));
    fit.preset = struct('N',round(p(1)*10)/10,'pbar',round(p(2)*100)/100,'Q',data.Q, ...
        'train',struct('nStim',K,'freqHz',data.freqHz, ...
            'refillPerStim',round(p(5)*10)/10,'nMaxFactor',round(p(3)*100)/100, ...
            'refillSteadyPerStim',round(p(6)*10)/10,'tauRefillStim',round(p(7)*100)/100, ...
            'tauNStim',round(p(4)*100)/100,'pMaxFactor',round(p(8)*100)/100, ...
            'tauPStim',round(p(9)*100)/100, ...
            'tauDecayMs',2.5), ...
        'target',struct('stim',1:K,'pr',prT,'N',nT,'ppr',pprT,'label',data.label));
    fit.label = data.label;

    if opt.Verbose
        fprintf('%s: N1=%.2f P1=%.2f nMax=%.2fxA1 tauN=%.2f RR=%.2f->%.2f tauRR=%.2f Pmax=%.2fxP1 tauP=%.2f | R2 pr/N/ppr = %.2f/%.2f/%.2f\n', ...
            data.label,p(1),p(2),p(3),p(4),p(5),p(6),p(7),p(8),p(9),fit.r2.pr,fit.r2.N,fit.r2.ppr);
    end
end

% ---------------------------------------------------------------------------
function [bestP,bestC,dataC,kFree] = optimizeMode(mode,x0,lb,ub,fixedVal,freeMask, ...
        opt,data,prT,nT,pprT,w,sPr,sN,sPPR,K)
    modeFixed = fixedVal;
    modeFree = freeMask;
    if strcmp(mode,'constant')
        % RRinf is tied to RR1; tau_RR has no meaning in this nested model.
        modeFixed(6) = x0(6); modeFree(6) = false;
        modeFixed(7) = 1.5; modeFree(7) = false;
    end
    expand = @(zf) applyRefillMode(mergeParams( ...
        unpack(zf,lb(modeFree),ub(modeFree)),modeFixed,modeFree),mode);
    objfun = @(zf) trainCost(expand(zf),data,prT,nT,pprT,w,sPr,sN,sPPR,K,opt);
    bestZ = pack(x0(modeFree),lb(modeFree),ub(modeFree));
    bestC = objfun(bestZ);
    sopt = optimset('Display','off','MaxFunEvals',6000,'MaxIter',6000, ...
        'TolX',1e-6,'TolFun',1e-8);
    rng(7 + double(strcmp(mode,'dynamic')),'twister');
    for r = 1:opt.Restarts
        if r == 1
            z0 = pack(x0(modeFree),lb(modeFree),ub(modeFree));
        else
            xr = clampvec(x0 .* (0.6 + 0.8*rand(1,9)) + ...
                (rand(1,9)-0.5).*(ub-lb)*0.15,lb,ub);
            z0 = pack(xr(modeFree),lb(modeFree),ub(modeFree));
        end
        [z,c] = fminsearch(objfun,z0,sopt);
        if c < bestC, bestC = c; bestZ = z; end
        if opt.Verbose
            fprintf('  %s restart %2d: cost=%.5f (best=%.5f)\n',mode,r,c,bestC);
        end
    end
    bestP = expand(bestZ);
    dataC = trainDataCost(bestP,data,prT,nT,pprT,w,sPr,sN,sPPR,K);
    kFree = sum(modeFree);
end

function p = applyRefillMode(p,mode)
    if strcmp(mode,'constant')
        p(6) = p(5);
        p(7) = 1.5;
    end
end

function c = trainCost(p,data,prT,nT,pprT,w,sPr,sN,sPPR,K,opt)
    c = trainDataCost(p,data,prT,nT,pprT,w,sPr,sN,sPPR,K);
    rrScale = max(sN,1);
    % Magnitude prior resolves the flat direction above the latent-N ceiling;
    % the difference prior asks dynamic RR to justify its two extra parameters.
    c = c + 1e-4 * ((p(5)/rrScale)^2 + (p(6)/rrScale)^2) + ...
        opt.RrEvolutionPenalty * ((p(6)-p(5))/rrScale)^2;
end

function c = trainDataCost(p,data,prT,nT,pprT,w,sPr,sN,sPPR,K)
    pars = struct('N',p(1),'pbar',p(2),'Q',data.Q,'cvQ',0,'cvP',0);
    tp = struct('nStim',K,'freqHz',data.freqHz,'refillPerStim',p(5), ...
        'refillSteadyPerStim',p(6),'tauRefillStim',p(7), ...
        'nMaxFactor',p(3),'tauNStim',p(4),'pMaxFactor',p(8),'tauPStim',p(9));
    m = mpfa_train_core(pars,tp);
    c = w(1)*nmse(m.pr,prT,sPr) + w(2)*nmse(m.readyBefore,nT,sN) + w(3)*nmse(m.ppr,pprT,sPPR);
end

function n = effectiveObservations(w,prT,nT,pprT)
    n = 0;
    if w(1) > 0, n = n + sum(~isnan(prT)); end
    if w(2) > 0, n = n + sum(~isnan(nT)); end
    if w(3) > 0, n = n + sum(~isnan(pprT)); end
    n = max(n,1);
end

function e = nmse(model,target,scale)
    v = ~isnan(target);
    if ~any(v), e = 0; return; end
    e = mean(((model(v)-target(v))/scale).^2);
end

function v = rmse(model,target)
    m = ~isnan(target);
    if ~any(m), v = NaN; return; end
    v = sqrt(mean((model(m)-target(m)).^2));
end

function v = r2(model,target)
    m = ~isnan(target);
    if sum(m) < 2, v = NaN; return; end
    t = target(m); f = model(m);
    ss = sum((t-mean(t)).^2);
    if ss < eps, v = NaN; return; end
    v = 1 - sum((t-f).^2)/ss;
end

function y = fixedQPpr(prT,nT)
    y = nan(size(prT));
    valid = ~isnan(prT) & ~isnan(nT);
    if isempty(prT) || isempty(nT) || isnan(prT(1)) || isnan(nT(1))
        return;
    end
    y(valid) = (prT(valid) ./ max(prT(1),eps)) .* ...
        (nT(valid) ./ max(nT(1),eps));
end

function p = mergeParams(freeVals,fixedVal,freeMask)
    p = fixedVal;
    p(freeMask) = freeVals;
end

% bounded transform: z (real) <-> param in [lb,ub] via logistic
function p = unpack(z,lb,ub)
    p = lb + (ub-lb)./(1+exp(-z));
end
function z = pack(p,lb,ub)
    p = clampvec(p,lb+1e-6*(ub-lb),ub-1e-6*(ub-lb));
    z = log((p-lb)./(ub-p));
end

function y = getfielddef(s,f,def)
    if nargin < 3, def = []; end
    if isfield(s,f) && ~isempty(s.(f))
        y = s.(f);
        if isnumeric(y) && ~isscalar(y), y = y(:)'; end
    else
        y = def;
    end
end
function y = padto(x,K)
    y = nan(1,K); if ~isempty(x), y(1:numel(x)) = x; end
end
function s = nanscale(x)
    m = mean(x(~isnan(x))); if isempty(m) || ~isfinite(m) || m==0, s = 1; else, s = abs(m); end
end
function v = pick(x,i,def)
    if numel(x) >= i && ~isnan(x(i)), v = x(i); else, v = def; end
end
function v = safemax(x)
    x = x(~isnan(x)); if isempty(x), v = NaN; else, v = max(x); end
end
function v = nanmedian(x)
    x = x(~isnan(x)); if isempty(x), v = NaN; else, v = median(x); end
end
function v = clampv(x,lo,hi)
    v = min(max(x,lo),hi);
end
function v = clampvec(x,lo,hi)
    v = min(max(x,lo),hi);
end
