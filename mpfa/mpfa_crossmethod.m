function pred = mpfa_crossmethod(source,in,ref)
% MPFA_CROSSMETHOD  Round-trip prediction across the quantal analysis views.
%
%   pred = mpfa_crossmethod(source,in,ref) takes an estimate obtained by one
%   analysis view and predicts the observables of the other views, using the
%   fact that the variance-mean parabola, the two-state relative-change plot and
%   the event-amplitude fit are all interfaces onto the same binomial model
%       mu = N*P*Q,   sigma^2 = N*P*(1-P)*Q^2.
%
%   ref is the reference state struct('N0',N0,'P0',P0,'Q0',Q0).
%
%   source selects the input view:
%     'npq'      in = struct('N',N1,'P',P1,'Q',Q1) for a single test state.
%     'relative' in = struct('x',x,'y',y): observed normalized coordinates.
%                Inverted under the fixed-Q constraint (Eq. 6 of the manuscript).
%     'mpfa'     in = struct('N',N,'Q',Q,'P',Pvec[,'refIndex',j]): an MPFA fit
%                and a probability series; predicts every condition's location.
%
%   Returned pred fields (scalar unless a series is supplied):
%     x, y            normalized relative-change coordinates
%     mu, sigma2      variance-mean operating point(s)
%     releasePpr      release-linear amplitude ratio mu/mu0 (= x)
%     rhoN,rhoP,rhoQ  parameter ratios vs the reference (NaN where unidentified)
%     N1,P1,Q1        implied test-state parameters (NaN where unidentified)
%     assumption      text note on any constraint used
%
%   The relative-change inversion is unique only under fixed Q, because the two
%   observables (x,y) cannot determine three ratios; 'assumption' records this.
%
%   See also MPFA_TRAIN_CORE, MPFA_FIT_HETEROGENEITY, MPFA_BOOTSTRAP_CONCORDANCE.

    ref = fillRef(ref);
    N0 = ref.N0; P0 = ref.P0; Q0 = ref.Q0;
    mu0 = N0*P0*Q0;

    switch lower(source)
        case 'npq'
            N1 = in.N; P1 = in.P; Q1 = getdef(in,'Q',Q0);
            mu  = N1.*P1.*Q1;
            sig = N1.*P1.*(1-P1).*Q1.^2;
            x = mu./mu0;
            rhoP = P1./P0; rhoQ = Q1./Q0; rhoN = N1./N0;
            y = rhoQ .* (1 - P0.*rhoP) ./ (1 - P0);
            pred = packpred(x,y,mu,sig,rhoN,rhoP,rhoQ,N1,P1,Q1, ...
                'forward binomial map (no constraint)');

        case 'relative'
            x = in.x; y = in.y;
            % Fixed-Q inversion: P1 = 1 - y(1-P0), rhoP = P1/P0, rhoN = x/rhoP.
            P1 = 1 - y.*(1-P0);
            rhoP = P1./P0;
            rhoN = x./rhoP;
            rhoQ = ones(size(x));
            N1 = rhoN.*N0; Q1 = Q0.*ones(size(x));
            mu = x.*mu0;
            sig = N1.*P1.*(1-P1).*Q1.^2;
            pred = packpred(x,y,mu,sig,rhoN,rhoP,rhoQ,N1,P1,Q1, ...
                'fixed-Q constraint (rhoQ = 1)');

        case 'mpfa'
            N = in.N; Q = in.Q; P = in.P(:)';
            j = getdef(in,'refIndex',1);
            mu  = N.*P.*Q;
            sig = N.*P.*(1-P).*Q.^2;
            muRef = mu(j); vmrRef = sig(j)./mu(j);
            x = mu./muRef;
            y = (sig./mu)./vmrRef;
            rhoP = P./P(j); rhoQ = ones(size(P)); rhoN = ones(size(P));
            pred = packpred(x,y,mu,sig,rhoN,rhoP,rhoQ, ...
                N.*ones(size(P)),P,Q.*ones(size(P)), ...
                'MPFA series: N,Q fixed, P varied');
            pred.refIndex = j;

        otherwise
            error('mpfa_crossmethod:source', ...
                'Unknown source "%s" (use npq, relative, or mpfa).',source);
    end

    % Loci predicted for the current reference, for overlay convenience.
    pred.loci = relativeChangeLoci(P0);
    pred.ref = ref;
end

% -------------------------------------------------------------------------
function loci = relativeChangeLoci(P0)
% The three pure-change lines through (1,1) in the normalized plane.
    xg = linspace(0,3,121);
    loci = struct('x',xg, ...
        'yN',ones(size(xg)), ...              % pure N change: y = 1
        'yQ',xg, ...                          % pure Q change: y = x
        'yP',(1 - P0.*xg)./(1 - P0), ...      % pure P change: y = (1-P0 x)/(1-P0)
        'slopeP',-P0/(1-P0));
end

function pred = packpred(x,y,mu,sig,rhoN,rhoP,rhoQ,N1,P1,Q1,note)
    pred = struct('x',x,'y',y,'mu',mu,'sigma2',sig,'releasePpr',x, ...
        'rhoN',rhoN,'rhoP',rhoP,'rhoQ',rhoQ,'N1',N1,'P1',P1,'Q1',Q1, ...
        'assumption',note);
end

function ref = fillRef(ref)
    if ~isfield(ref,'N0') || isempty(ref.N0), ref.N0 = 8;   end
    if ~isfield(ref,'P0') || isempty(ref.P0), ref.P0 = 0.25; end
    if ~isfield(ref,'Q0') || isempty(ref.Q0), ref.Q0 = 0.5;  end
end

function v = getdef(s,f,def)
    if isfield(s,f) && ~isempty(s.(f)), v = s.(f); else, v = def; end
end
