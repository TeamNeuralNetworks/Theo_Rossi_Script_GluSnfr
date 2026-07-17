function results = mpfa_fit_demo(launchCond,doLaunch,weights)
% MPFA_FIT_DEMO  Fit the dynamic train model to an example dataset and launch
%                the MPFA simulator preloaded with the best-fit configuration.
%
%   results = mpfa_fit_demo() fits every Ca/frequency condition in the embedded
%   dataset, prints a summary table, and opens MPFA_NPQ_SIMULATOR with the
%   '2.5mM_20Hz' preset (train mode auto-enabled).
%
%   results = mpfa_fit_demo(launchCond) launches a different condition, one of:
%       '1.5mM_20Hz' '1.5mM_50Hz' '2.5mM_20Hz' '2.5mM_50Hz' '4mM_20Hz' '4mM_50Hz'
%
%   results = mpfa_fit_demo(launchCond,false) fits and reports without opening
%   the GUI (useful for scripting / headless runs).
%
%   results = mpfa_fit_demo(launchCond,doLaunch,[wPr wN wPPR]) sets the fit
%   weights. The default [1 1 0] fits the two directly measured curves (Pr and
%   apparent N) and leaves PPR as the fixed-Q consistency prediction.
%
%   The dataset gives, per condition, the apparent per-stimulus release
%   probability (Pr), apparent site count (N) and an amplitude ratio (PPR)
%   across a 10-pulse train. Q is not constrained by these ratios, so it
%   is fixed at 0.5 DeltaF/F0 per quantum; the fit pins N1/P1, then adjusts latent Nmax, tau_N,
%   evolving refill (RR1, RRinf, tau_RR), and the Pr-facilitation asymptote
%   Pmax/P1 with its rise time tau_P. Because apparent
%   N need not be mechanistically consistent with Pr and PPR, inspect the
%   per-series RMSE in the printed table.
%
%   PPR target convention. The fitted Pr and N are population-level (ensemble)
%   quantities, so the amplitude ratio they predict, (N_k/N_1)(P_k/P_1), is the
%   ratio of population means. The held-out PPR target is therefore taken from
%   Mean_percent_A1 in FigS16_var_imean_profiles_6conditions.csv (the ensemble
%   ratio), NOT from the PPR columns of the profiles matrix: those PPR columns
%   are a per-bouton mean-of-ratios statistic (each bouton's own AMP_k/AMP_1,
%   averaged across boutons) and describe the typical bouton rather than the
%   ensemble. Mixing the two differs by ~2x at these synapses.
%
%   This is a WORKAROUND, not the fix. The real problem is upstream: Pr, N and
%   PPR are each aggregated across boutons at a different point in the
%   calculation, and those operations do not commute. See
%   TODO_per_bouton_aggregation.md. Once the pipeline estimates per bouton and
%   aggregates last over one MLE-valid bouton list, revert this to read the PPR
%   target from the profiles matrix again.
%   The demo is fitted hierarchically: both frequencies share N1/P1 at each
%   calcium level, absolute latent Nmax/Pmax are monotonic with calcium and
%   frequency, and evolving RR is retained only when it beats constant RR by
%   AIC. These constraints prevent ceiling-hidden and boundary solutions.
%
%   See also MPFA_FIT_TRAIN, MPFA_TRAIN_CORE, MPFA_NPQ_SIMULATOR.

    if nargin < 1 || isempty(launchCond), launchCond = '2.5mM_20Hz'; end
    if nargin < 2 || isempty(doLaunch), doLaunch = true; end
    if nargin < 3 || isempty(weights), weights = [1 1 0]; end

    % Conditions in this order (must match the CSV column order below):
    labels = {'1.5mM_20Hz','1.5mM_50Hz','2.5mM_20Hz','2.5mM_50Hz','4mM_20Hz','4mM_50Hz'};
    freqs  = [20 50 20 50 20 50];

    % Per-stimulus apparent Pr and apparent N come directly from the real
    % dataset, read from the shipped CSV rather than a hardcoded copy. The file
    % has one header row and columns Stim | Pr(6) | N(6) | PPR(6) in the same
    % condition order as `labels`, so column 1+c is Pr and 7+c is N for
    % condition c (see FigS16_mpfa_demo_profiles_matrix.csv).
    here = fileparts(mfilename('fullpath'));
    csvPath = fullfile(here,'FigS16_mpfa_demo_profiles_matrix.csv');
    assert(isfile(csvPath),'Demo data CSV not found: %s',csvPath);
    M = readmatrix(csvPath);
    assert(size(M,2) == 19, ...
        'Expected 19 columns (Stim + Pr/N/PPR x6) in %s, found %d.', ...
        csvPath,size(M,2));

    % Held-out PPR target = ensemble amplitude ratio (ratio of population
    % means), matching the population-level Pr/N above. See the header note.
    vmPath = fullfile(here,'FigS16_var_imean_profiles_6conditions.csv');
    assert(isfile(vmPath),'Variance-to-mean CSV not found: %s',vmPath);
    pprEnsemble = ensemblePprMatrix(readtable(vmPath, ...
        'VariableNamingRule','preserve'),labels,size(M,1));

    nCond = numel(labels);
    results = struct('label',{},'fit',{});

    % The two frequencies at each calcium concentration start from one shared
    % A1 state. This removes a source of non-biological freedom and makes A1
    % independent of the train that follows it.
    sharedN1 = [mean(M(1,8:9)) mean(M(1,10:11)) mean(M(1,12:13))];
    sharedP1 = [mean(M(1,2:3)) mean(M(1,4:5)) mean(M(1,6:7))];

    % Fit 20 Hz in increasing Ca, then 50 Hz in increasing Ca. Absolute latent
    % Nmax and Pmax are constrained not to decrease with either Ca or frequency.
    fitOrder = [1 3 5 2 4 6];
    nFloorByFreq = [0 0]; pFloorByFreq = [0 0];
    nMax20 = zeros(1,3); pMax20 = zeros(1,3);
    for ii = 1:nCond
        c = fitOrder(ii);
        caIdx = ceil(c/2);
        freqIdx = 1 + (freqs(c) == 50);
        minNmax = nFloorByFreq(freqIdx);
        minPmax = pFloorByFreq(freqIdx);
        if freqIdx == 2
            minNmax = max(minNmax,nMax20(caIdx));
            minPmax = max(minPmax,pMax20(caIdx));
        end
        data = struct('pr',M(:,1+c)','N',M(:,7+c)','ppr',pprEnsemble(:,c)', ...
            'freqHz',freqs(c),'Q',0.5,'label',labels{c});
        fit = mpfa_fit_train(data,'Weights',weights,'Restarts',14, ...
            'InitialN',sharedN1(caIdx),'InitialP',sharedP1(caIdx), ...
            'MinNmaxAbs',minNmax,'MinPmaxAbs',minPmax,'RefillMode','auto');
        results(c).label = labels{c};
        results(c).fit = fit;
        absNmax = fit.pars.N * fit.trainPars.nMaxFactor;
        absPmax = min(fit.pars.pbar * fit.trainPars.pMaxFactor,0.98);
        nFloorByFreq(freqIdx) = absNmax;
        pFloorByFreq(freqIdx) = absPmax;
        if freqIdx == 1
            nMax20(caIdx) = absNmax;
            pMax20(caIdx) = absPmax;
        end
    end

    fprintf('\n%-12s %5s %5s %6s %6s %6s %6s %6s %6s %6s %-5s | %6s %6s %6s\n', ...
        'condition','N1','P1','Nmax','tauN','RR1','RRinf','tauRR','Pmax','tauP','RRmod','rmsePr','rmseN','rmPPR');
    fprintf('%s\n',repmat('-',1,109));
    for c = 1:nCond
        fit = results(c).fit;
        p = fit.pars; t = fit.trainPars;
        fprintf('%-12s %5.2f %5.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %-5s | %6.3f %6.2f %6.3f\n', ...
            labels{c},p.N,p.pbar,p.N*t.nMaxFactor,t.tauNStim,t.refillPerStim, ...
            t.refillSteadyPerStim,t.tauRefillStim,min(p.pbar*t.pMaxFactor,0.98),t.tauPStim, ...
            fit.refillModel.selected, ...
            fit.rmse.pr,fit.rmse.N,fit.rmse.ppr);
    end
    fprintf('%s\n',repmat('-',1,109));
    fprintf(['RMSE is in native units (Pr, sites, ratio). Nmax and Pmax are absolute\n', ...
             'asymptotes; tau_N, tau_RR and tau_P act per stimulus. RRmod is selected\n', ...
             'by AIC (constant unless evolving RR earns its extra parameters). The two\n', ...
             'frequencies share N1/P1 within each Ca, and absolute Nmax/Pmax cannot\n', ...
             'decrease with increasing Ca or frequency. Pmax factor >1 = facilitation, <1 =\n', ...
             'depression. The held-out PPR target is the ensemble amplitude ratio\n', ...
             '(Mean_percent_A1), which is the quantity the population-level N and Pr\n', ...
             'predict; the per-bouton PPR columns are a different statistic.\n\n']);

    idx = find(strcmp(labels,launchCond),1);
    if isempty(idx)
        warning('Unknown condition "%s"; not launching. Valid: %s',launchCond,strjoin(labels,', '));
        return;
    end
    fprintf('Preset for %s:\n',launchCond);
    disp(results(idx).fit.preset);
    disp(results(idx).fit.preset.train);

    if doLaunch
        fprintf('Launching mpfa_npq_simulator with the %s preset...\n',launchCond);
        mpfa_npq_simulator(results(idx).fit.preset);
    end
end

% -------------------------------------------------------------------------
function P = ensemblePprMatrix(T,labels,nStim)
% Ensemble amplitude ratio per condition: Mean_percent_A1/100 from the
% variance-to-mean CSV, ordered by stimulus and aligned to `labels`.
    P = nan(nStim,numel(labels));
    freqCol = T.Frequency_Hz;
    caCol = string(T.Calcium);
    for c = 1:numel(labels)
        tok = regexp(labels{c},'([\d.]+)mM_(\d+)Hz','tokens','once');
        ca = str2double(tok{1}); fr = str2double(tok{2});
        sel = freqCol == fr & caCol == sprintf('%g mM',ca);
        assert(any(sel),'No variance-to-mean rows found for condition %s.',labels{c});
        ev = T.Event(sel);
        val = T.Mean_percent_A1(sel) / 100;
        keep = ev >= 1 & ev <= nStim;
        P(ev(keep),c) = val(keep);
    end
end
