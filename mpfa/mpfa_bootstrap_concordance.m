function res = mpfa_bootstrap_concordance(obs,varargin)
% MPFA_BOOTSTRAP_CONCORDANCE  Quantify agreement between an MLE release-probability
%                             estimate and the variance-to-mean geometry.
%
%   res = mpfa_bootstrap_concordance(obs) turns the qualitative "the MLE point
%   falls on the predicted locus" statement into a number with a confidence
%   interval. It fits the reference release probability P0 implied by an observed
%   normalized variance-to-mean trajectory and compares it with an independent
%   MLE estimate of P0.
%
%   obs fields:
%     .x      1xK mean ratio  mu_k/mu_1            (relativeX)
%     .y      1xK normalized  (sigma^2/mu) ratio   (relativeY)
%     .ySD    1xK standard error of y  (optional; see CI fields)
%     .yLo/.yHi  1xK 95% CI of y (used to derive ySD = (hi-lo)/(2*1.96))
%     .P0mle  scalar independent MLE estimate of the reference P (optional)
%     .trials 1xK cell of raw per-pulse y replicates (optional; enables a
%             nonparametric bootstrap instead of the parametric one)
%
%   opts (name/value or struct):
%     'B'      bootstrap replicates (default 2000)
%     'Seed'   RNG seed (default 11)
%
%   Returned res fields:
%     .P0geom          P0 fitted from the variance-to-mean trajectory
%     .P0geomCI        [lo hi] 95% bootstrap interval for P0geom
%     .P0mle           echoed independent MLE estimate
%     .zConcordance    standardized |P0geom - P0mle| / bootstrapSD (NaN if no MLE)
%     .agree           true if P0mle lies within P0geomCI
%     .perPulse        struct(rhoN,rhoP,rhoNsd,rhoPsd) fixed-Q decomposition
%     .bootSamples     1xB fitted P0geom replicates
%
%   See also MPFA_CROSSMETHOD, MPFA_FIT_HETEROGENEITY.

    opts = optsFrom(varargin);
    B    = getdef(opts,'B',2000);
    seed = getdef(opts,'Seed',11);

    x = obs.x(:)'; y = obs.y(:)';
    K = numel(x);
    ySD = resolveSD(obs,K);
    P0mle = getdef(obs,'P0mle',NaN);

    % Point estimate: P0 that best matches the pure-P locus y=(1-P0 x)/(1-P0).
    P0geom = fitP0(x,y,ySD);

    % Bootstrap the trajectory to get a CI on P0geom.
    rng(seed,'twister');
    boot = nan(1,B);
    haveTrials = isfield(obs,'trials') && ~isempty(obs.trials);
    for b = 1:B
        if haveTrials
            yb = nonparamResample(obs.trials);
        else
            yb = y + ySD.*randn(1,K);      % parametric resample from the CIs
        end
        boot(b) = fitP0(x,yb,ySD);
    end
    boot = boot(isfinite(boot));
    ci = quantile95(boot);
    sdBoot = std(boot);

    if isfinite(P0mle) && sdBoot > 0
        zC = abs(P0geom - P0mle)/sdBoot;
        agree = P0mle >= ci(1) && P0mle <= ci(2);
    else
        zC = NaN; agree = NaN;
    end

    % Fixed-Q per-pulse decomposition using the MLE P0 when available.
    P0dec = P0mle; if ~isfinite(P0dec), P0dec = P0geom; end
    rhoP = (1 - y.*(1-P0dec))/P0dec;
    rhoN = x./rhoP;
    % Propagate y uncertainty to rhoN, rhoP (delta method).
    drhoP = abs(-(1-P0dec)/P0dec).*ySD;
    drhoN = abs(x./(rhoP.^2)).*drhoP;

    res = struct('P0geom',P0geom,'P0geomCI',ci,'P0mle',P0mle, ...
        'zConcordance',zC,'agree',agree,'bootSamples',boot, ...
        'perPulse',struct('rhoN',rhoN,'rhoP',rhoP,'rhoNsd',drhoN,'rhoPsd',drhoP));
end

% -------------------------------------------------------------------------
function P0 = fitP0(x,y,ySD)
    w = 1./max(ySD,eps).^2;
    obj = @(p) sum(w.*(y - (1-p.*x)./(1-p)).^2);
    P0 = fminbnd(obj,1e-3,0.98);
end

function yb = nonparamResample(trials)
    K = numel(trials);
    yb = nan(1,K);
    for k = 1:K
        v = trials{k}; v = v(~isnan(v));
        if isempty(v), yb(k) = NaN; else
            yb(k) = mean(v(randi(numel(v),1,numel(v))));
        end
    end
end

function ySD = resolveSD(obs,K)
    if isfield(obs,'ySD') && ~isempty(obs.ySD)
        ySD = obs.ySD(:)';
    elseif isfield(obs,'yLo') && isfield(obs,'yHi')
        ySD = (obs.yHi(:)' - obs.yLo(:)')/(2*1.959963985);
    else
        ySD = ones(1,K);
    end
    ySD(~isfinite(ySD) | ySD<=0) = median(ySD(isfinite(ySD) & ySD>0),'omitnan');
    if all(~isfinite(ySD)), ySD = ones(1,K); end
end

function ci = quantile95(v)
    if isempty(v), ci = [NaN NaN]; return; end
    ci = quantile(v,[0.025 0.975]);
end

function s = optsFrom(args)
    if numel(args)==1 && isstruct(args{1}), s = args{1}; return; end
    s = struct;
    for i = 1:2:numel(args), s.(args{i}) = args{i+1}; end
end

function v = getdef(s,f,def)
    if isfield(s,f) && ~isempty(s.(f)), v = s.(f); else, v = def; end
end
