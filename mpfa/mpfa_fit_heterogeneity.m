function het = mpfa_fit_heterogeneity(amps,varargin)
% MPFA_FIT_HETEROGENEITY  Estimate N, P, Q and quantal scatter CV_Q from an event
%                         amplitude distribution, with AIC model selection.
%
%   het = mpfa_fit_heterogeneity(amps) fits the binomial-Gaussian mixture of
%   Eq. (17) to a vector of measured event amplitudes:
%       p(a) = sum_{j=0}^N Binom(N,j,P) * Normal(a; j*Q, sqrt(sigma0^2 + j*(CV_Q*Q)^2))
%   by maximum likelihood (base MATLAB fminsearch, integer N searched on a grid).
%   It fits a homogeneous model (CV_Q = 0) and a heterogeneous model (CV_Q free)
%   and selects between them by AIC, so the quantal-scatter correction applied to
%   a dataset is fitted rather than assumed.
%
%   opts (name/value or struct):
%     'Nrange'  candidate site counts (default 1:12)
%     'Mode'    'auto' (AIC, default) | 'homogeneous' | 'heterogeneous'
%     'Sigma0'  fixed recording-noise SD (optional; else fitted)
%     'Restarts' random restarts per N (default 6)
%
%   Returned het fields:
%     .N,.P,.Q,.sigma0,.cvQ   selected-model parameters
%     .logL,.aic,.k           selected-model fit quality
%     .homogeneous,.heterogeneous  full sub-model structs
%     .selected               'homogeneous' | 'heterogeneous'
%     .dAIC                   aic(homogeneous) - aic(heterogeneous)
%
%   CV_P (site-to-site release-probability heterogeneity) is not identifiable
%   from a single amplitude histogram; use PARABOLA_CVP (local helper, see
%   mpfa_fit_heterogeneity('help')) on a multi-condition variance-mean fit with an
%   independent N. This routine returns the identifiable CV_Q.
%
%   See also MPFA_CROSSMETHOD, MPFA_BOOTSTRAP_CONCORDANCE.

    if ischar(amps) && strcmpi(amps,'help')
        het = @parabolaCVp; return;   % expose the parabola CV_P helper
    end
    opts = optsFrom(varargin);
    Nrange   = getdef(opts,'Nrange',1:12);
    mode     = lower(getdef(opts,'Mode','auto'));
    sigma0Fx = getdef(opts,'Sigma0',[]);
    restarts = getdef(opts,'Restarts',6);

    a = amps(:); a = a(~isnan(a));
    assert(numel(a) >= 10,'Need at least 10 amplitudes to fit.');

    hom = fitOne(a,Nrange,true ,sigma0Fx,restarts);
    het2 = fitOne(a,Nrange,false,sigma0Fx,restarts);

    switch mode
        case 'homogeneous',   sel = 'homogeneous';   S = hom;
        case 'heterogeneous', sel = 'heterogeneous'; S = het2;
        otherwise
            if het2.aic + 2 < hom.aic, sel = 'heterogeneous'; S = het2;
            else,                       sel = 'homogeneous';   S = hom;  end
    end

    het = S;
    het.homogeneous = hom;
    het.heterogeneous = het2;
    het.selected = sel;
    het.dAIC = hom.aic - het2.aic;
end

% -------------------------------------------------------------------------
function best = fitOne(a,Nrange,homogeneous,sigma0Fx,restarts)
    best = struct('N',NaN,'P',NaN,'Q',NaN,'sigma0',NaN,'cvQ',0, ...
        'logL',-Inf,'aic',Inf,'k',NaN);
    a95 = quantile(a,0.95);
    for N = Nrange
        % Parameters optimized in transformed space: [P Q sigma0 (cvQ)].
        Q0 = max(a95/max(N,1),1e-3);
        s0 = max(std(a)/3,1e-3);
        base = [0.3, Q0, s0];
        if ~homogeneous, base = [base, 0.3]; end
        kFit = numel(base) + 1;                 % +1 for integer N
        if ~isempty(sigma0Fx), kFit = kFit - 1; end
        rng(101+N,'twister');
        for r = 1:restarts
            if r == 1, z = base; else
                z = base .* (0.5 + rand(size(base)));
            end
            obj = @(t) -mixLogL(a,N,unpackPar(t,homogeneous,sigma0Fx));
            t0 = packPar(z,homogeneous,sigma0Fx);
            t = fminsearch(obj,t0,optimset('Display','off', ...
                'MaxFunEvals',4000,'MaxIter',4000,'TolX',1e-6,'TolFun',1e-8));
            par = unpackPar(t,homogeneous,sigma0Fx);
            LL = mixLogL(a,N,par);
            aic = 2*kFit - 2*LL;
            if aic < best.aic
                best = struct('N',N,'P',par.P,'Q',par.Q,'sigma0',par.sigma0, ...
                    'cvQ',par.cvQ,'logL',LL,'aic',aic,'k',kFit);
            end
        end
    end
end

function LL = mixLogL(a,N,par)
    P = par.P; Q = par.Q; s0 = par.sigma0; sQ = par.cvQ*Q;
    if P<=0 || P>=1 || Q<=0 || s0<=0, LL = -1e12; return; end
    j = (0:N);
    logw = gammaln(N+1)-gammaln(j+1)-gammaln(N-j+1) + j*log(P) + (N-j)*log(1-P);
    sdj = sqrt(s0^2 + j*sQ^2);
    % log Normal(a; jQ, sdj) for every amplitude x every component.
    A = a(:);                                   % M x 1
    mu = j*Q;                                    % 1 x (N+1)
    comp = -0.5*log(2*pi) - log(sdj) - 0.5*((A-mu)./sdj).^2;  % M x (N+1)
    comp = comp + logw;                          % add mixture log-weights
    LL = sum(logsumexp_rows(comp));
    if ~isfinite(LL), LL = -1e12; end
end

function s = logsumexp_rows(X)
    m = max(X,[],2);
    s = m + log(sum(exp(X-m),2));
end

function par = unpackPar(t,homogeneous,sigma0Fx)
    P  = 1/(1+exp(-t(1)));
    Q  = exp(t(2));
    if isempty(sigma0Fx), s0 = exp(t(3)); idx = 4; else, s0 = sigma0Fx; idx = 3; end
    if homogeneous, cvQ = 0; else, cvQ = exp(t(idx)); end
    par = struct('P',P,'Q',Q,'sigma0',s0,'cvQ',cvQ);
end

function t = packPar(z,homogeneous,sigma0Fx)
    P = min(max(z(1),1e-3),1-1e-3);
    t = [log(P/(1-P)), log(max(z(2),1e-6))];
    if isempty(sigma0Fx), t = [t, log(max(z(3),1e-6))]; base = 3;
    else, base = 2; end
    if ~homogeneous, t = [t, log(max(z(base+1),1e-6))]; end
end

% CV_P from a variance-mean parabola given an independent N (Eq. 7/8).
function cvP = parabolaCVp(muVec,varVec,Nindep,Qapp)
    % Fit sigma^2 = Qapp*mu - b*mu^2 for b, then N_app = 1/b and
    % (1+CV_P^2) = N_indep / N_app  =>  CV_P = sqrt(N_indep/N_app - 1).
    mu = muVec(:); v = varVec(:);
    b = (Qapp*mu - v) \ (mu.^2);        % least squares for curvature term
    Napp = 1/max(b,eps);
    cvP = sqrt(max(Nindep/Napp - 1,0));
end

function s = optsFrom(args)
    if numel(args)==1 && isstruct(args{1}), s = args{1}; return; end
    s = struct;
    for i = 1:2:numel(args), s.(args{i}) = args{i+1}; end
end

function v = getdef(s,f,def)
    if isfield(s,f) && ~isempty(s.(f)), v = s.(f); else, v = def; end
end
