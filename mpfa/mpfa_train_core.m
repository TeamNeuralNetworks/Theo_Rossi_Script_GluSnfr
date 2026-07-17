function out = mpfa_train_core(pars,tp)
% MPFA_TRAIN_CORE  Pure (graphics-free) dynamic release model for a stimulus train.
%
%   out = mpfa_train_core(pars,tp) evaluates the mean-field depletion/refill
%   model used by MPFA_NPQ_SIMULATOR's train mode. It is shared by the GUI and
%   by MPFA_FIT_TRAIN so the two can never drift apart.
%
%   pars fields:  N, pbar, Q            (cvQ, cvP optional, default 0)
%   tp   fields:  nStim, freqHz, refillPerStim, nMaxFactor, tauNStim,
%                 pMaxFactor, tauPStim
%   optional:     refillSteadyPerStim, tauRefillStim
%
%   Release probability follows a saturating facilitation/depression law
%     P_k = Pmax - (Pmax - P1) exp(-(k-1)/tauP),   Pmax = clip(P1 * pMaxFactor)
%   which reduces to flat P when pMaxFactor = 1 (the default if the field is
%   absent). pMaxFactor > 1 gives facilitation, < 1 depression.
%
%   Returned struct fields (all 1 x nStim unless noted):
%     stim            stimulus index 1..nStim
%     latentN         latent recruitment "command" N_k (the tau_N/nMax target)
%     nMax            scalar asymptote N * nMaxFactor
%     readyBefore     ready pool before each pulse (apparent N_k)
%     readyAfter      ready pool after release
%     refillPerInterval replenishment command after each pulse
%     pr              release probability P_k
%     released        released quanta per pulse
%     meanResp        mean amplitude mu_k = released * Q
%     varResp         release variance sigma^2_k
%     normCumRelease  cumulative release normalized to pulse 1
%     relativeX       mu_k / mu_1
%     relativeY       (sigma^2_k/mu_k) / (sigma^2_1/mu_1)
%     releasePpr      released-glutamate amplitude ratio mu_k / mu_1
%     ppr             alias of releasePpr for display compatibility
%     isiMs           inter-stimulus interval (ms)

    if ~isfield(pars,'cvQ'), pars.cvQ = 0; end
    if ~isfield(pars,'cvP'), pars.cvP = 0; end

    n = tp.nStim;
    latentN = zeros(1,n);
    readyBefore = zeros(1,n);
    readyAfter = zeros(1,n);
    pr = zeros(1,n);
    meanResp = zeros(1,n);
    varResp = zeros(1,n);
    released = zeros(1,n);
    refillPerInterval = zeros(1,n);

    qEff = pars.Q * (1 + pars.cvQ^2);
    curvature = 1 + pars.cvP^2;
    nMax = pars.N * tp.nMaxFactor;
    tauN = max(tp.tauNStim,0.05);
    readyState = pars.N;

    % Saturating release-probability facilitation/depression law.
    %   P_k = Pmax - (Pmax - P1) exp(-(k-1)/tauP)
    pMaxFactor = getfielddef(tp,'pMaxFactor',1);   % default: flat P
    pMax = min(max(pars.pbar * pMaxFactor,0.01),0.98);
    tauP = max(getfielddef(tp,'tauPStim',1.5),0.05);

    % Replenishment may itself facilitate or depress across the train. The
    % legacy scalar behavior is recovered when refillSteadyPerStim is absent.
    refillFirst = max(tp.refillPerStim,0);
    refillSteady = max(getfielddef(tp,'refillSteadyPerStim',refillFirst),0);
    tauRefill = max(getfielddef(tp,'tauRefillStim',1.5),0.05);

    for kk = 1:n
        latentN(kk) = pars.N + (nMax - pars.N) * (1 - exp(-(kk - 1) / tauN));
        readyState = min(readyState,latentN(kk));
        readyBefore(kk) = readyState;
        pr(kk) = min(max(pMax - (pMax - pars.pbar) * exp(-(kk - 1) / tauP),0.01),0.95);
        released(kk) = readyBefore(kk) * pr(kk);
        meanResp(kk) = released(kk) * pars.Q;

        if readyBefore(kk) > 0 && meanResp(kk) > 0
            varResp(kk) = max(0,qEff * meanResp(kk) - curvature * meanResp(kk)^2 / readyBefore(kk));
        else
            varResp(kk) = 0;
        end

        readyAfter(kk) = max(readyBefore(kk) - released(kk),0);
        refillPerInterval(kk) = refillSteady - (refillSteady - refillFirst) * ...
            exp(-(kk - 1) / tauRefill);
        if kk < n
            nextLatentN = pars.N + (nMax - pars.N) * (1 - exp(-kk / tauN));
            recruited = max(nextLatentN - latentN(kk),0);
            readyState = min(nextLatentN,readyAfter(kk) + refillPerInterval(kk) + recruited);
        end
    end

    relVarRef = varResp(1) / max(meanResp(1),eps);
    relY = zeros(1,n);
    valid = meanResp > 0;
    relY(valid) = (varResp(valid) ./ meanResp(valid)) / max(relVarRef,eps);

    releasePpr = meanResp / max(meanResp(1),eps);

    out = struct( ...
        'stim',1:n, ...
        'latentN',latentN, ...
        'nMax',nMax, ...
        'readyBefore',readyBefore, ...
        'readyAfter',readyAfter, ...
        'refillPerInterval',refillPerInterval, ...
        'pr',pr, ...
        'released',released, ...
        'meanResp',meanResp, ...
        'varResp',varResp, ...
        'normCumRelease',cumsum(released) / max(released(1),eps), ...
        'relativeX',releasePpr, ...
        'relativeY',relY, ...
        'releasePpr',releasePpr, ...
        'ppr',releasePpr, ...
        'isiMs',1000 / max(tp.freqHz,eps));
end

function v = getfielddef(s,f,def)
    if isfield(s,f) && ~isempty(s.(f)), v = s.(f); else, v = def; end
end
