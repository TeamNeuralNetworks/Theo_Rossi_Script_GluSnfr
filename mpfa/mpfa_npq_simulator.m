function mpfa_npq_simulator(preset)
% MPFA_NPQ_SIMULATOR
% Interactive dashboard for multiple-probability fluctuation analysis (MPFA)
% parabola intuition, quantal-heterogeneity bias, and the Humeau-Poulain
% variance-to-mean relative-change loci for N, P, and Q.
%
% mpfa_npq_simulator(preset) starts from a supplied configuration, e.g. the
% output of MPFA_FIT_TRAIN. preset may contain the main-window fields N, pbar,
% Q and a preset.train struct (nStim, freqHz, refillPerStim,
% refillSteadyPerStim, tauRefillStim, nMaxFactor, tauNStim, ...); when
% preset.train is given, train mode opens
% automatically at those values. The relative-change reference stays at the
% canonical N0=8, p0=0.25, Q0=0.5 DeltaF/F0 regardless of the preset.
%
% References:
%   Clements JD, Silver RA (2000) Unveiling synaptic plasticity: a new
%       graphical and analytical approach. Trends Neurosci 23(3):105-113.
%   Silver RA (2003) Estimation of nonuniform quantal parameters with
%       multiple-probability fluctuation analysis: theory, application and
%       limitations. J Neurosci Methods 130(2):127-141.
%   Humeau Y, Doussau F, Popoff MR, Benfenati F, Poulain B (2007) Fast
%       changes in the functional status of release sites during short-term
%       plasticity: involvement of a frequency-dependent bypass of Rac at
%       Aplysia synapses. J Physiol 583(Pt 3):983-1004.  (variance-to-mean
%       relative-change analysis used to separate N- from P-driven changes.)
%   Classic CV / variance-to-mean lineage: Malinow & Tsien 1990;
%       Bekkers & Stevens 1990; Faber & Korn 1991; Lupica et al. 1992.

    if nargin < 1 || isempty(preset)
        preset = struct();
    end

    theme = makeTheme();
    defaults = struct( ...
        'N',8, ...
        'pbar',0.25, ...
        'Q',0.50, ...
        'showHet',false, ...
        'cvQ',0.00, ...
        'cvP',0.00);

    % Reference (comparison) state stays canonical even when a preset is loaded.
    ref = defaults;

    % Preset overrides only the current operating state, not the reference.
    if isfield(preset,'N'),    defaults.N = preset.N; end
    if isfield(preset,'pbar'), defaults.pbar = preset.pbar; end
    if isfield(preset,'Q'),    defaults.Q = preset.Q; end

    % Keep the relative-change view focused on the biologically relevant
    % neighborhood. It expands only when the current state/train requires it.
    relAxisMax = 2.5;
    % Reference-state quantities are constant for the whole session, so
    % precompute them once instead of on every slider refresh.
    refCache = buildReferenceCache(ref);
    H = struct();
    Train = struct( ...
        'enabled',false, ...
        'fig',[], ...
        'defaults',struct( ...
            'nStim',10, ...
            'freqHz',20, ...
            'refillPerStim',1.0, ...
            'refillSteadyPerStim',1.0, ... % late-train RR asymptote (ves/stim)
            'tauRefillStim',1.50, ...      % RR evolution time constant (stim)
            'nMaxFactor',1.00, ...   % Nmax / A1 (site recruitment asymptote)
            'tauNStim',0.40, ...     % site recruitment time constant (stim)
            'pMaxFactor',1.00, ...   % Pmax / P1 (release-prob facilitation asymptote)
            'tauPStim',1.50, ...     % release-prob facilitation time constant (stim)
            'tauDecayMs',2.5, ...  % 72A-fast sensor decay time constant (ms)
            'tauRiseMs',0.5, ...   % 72A-fast sensor rise time constant (ms)
            'baselineMs',20), ...   % flat pre-stimulus baseline (ms)
        'handles',struct(), ...
        'target',[], ...            % measured target (pr/N/ppr) shown behind model
        'data',[]);

    % Preset train configuration (e.g. from MPFA_FIT_TRAIN) opens train mode.
    startTrainMode = false;
    if isfield(preset,'train') && ~isempty(preset.train)
        tf = fieldnames(preset.train);
        for ii = 1:numel(tf)
            if isfield(Train.defaults,tf{ii})
                Train.defaults.(tf{ii}) = preset.train.(tf{ii});
            end
        end
        startTrainMode = true;
    end
    if isfield(preset,'target') && ~isempty(preset.target)
        Train.target = preset.target;   % faint background trace vs the model
    end

    Rep = struct( ...
        'enabled',false, ...
        'fig',[], ...
        'defaults',struct( ...
            'nRep',60, ...          % overlaid repetitions of the protocol
            'noiseSd',0.15, ...     % fluorescence baseline noise SD (DeltaF/F0)
            'eventIndex',1), ...    % which stimulus feeds the amplitude histogram
        'handles',struct(), ...
        'data',[]);

    screenSize = get(groot,'ScreenSize');
    figWidth = min(1360,screenSize(3) - 120);
    figHeight = min(860,screenSize(4) - 120);
    figLeft = max(40,round((screenSize(3) - figWidth) / 2));
    figBottom = max(40,round((screenSize(4) - figHeight) / 2));

    fig = figure( ...
        'Name','MPFA and relative-change explorer', ...
        'NumberTitle','off', ...
        'Color',theme.bg, ...
        'Units','pixels', ...
        'Position',[figLeft figBottom figWidth figHeight], ...
        'MenuBar','none', ...
        'ToolBar','none', ...
        'CloseRequestFcn',@closeMainFigure);

    uicontrol(fig,'Style','text','Units','normalized', ...
        'Position',[0.05 0.948 0.52 0.030], ...
        'String','MPFA parabola and Humeau-Poulain variance-to-mean relative-change explorer', ...
        'BackgroundColor',theme.bg, ...
        'ForegroundColor',theme.text, ...
        'FontSize',15, ...
        'FontWeight','bold', ...
        'HorizontalAlignment','left');

    uicontrol(fig,'Style','text','Units','normalized', ...
        'Position',[0.05 0.920 0.52 0.022], ...
        'String','Aligned MPFA display plus normalized N, P, Q diagnostics.', ...
        'BackgroundColor',theme.bg, ...
        'ForegroundColor',theme.muted, ...
        'FontSize',9.2, ...
        'HorizontalAlignment','left');

    H.relMeta = uicontrol(fig,'Style','text','Units','normalized', ...
        'Position',[0.59 0.920 0.36 0.022], ...
        'String',sprintf('Reference: N0=%d, p0=%.2f, Q0=%.2f ΔF/F₀',ref.N,ref.pbar,ref.Q), ...
        'BackgroundColor',theme.bg, ...
        'ForegroundColor',theme.muted, ...
        'FontSize',9.5, ...
        'HorizontalAlignment','right');

    H.relSummary = uicontrol(fig,'Style','text','Units','normalized', ...
        'Position',[0.59 0.895 0.36 0.018], ...
        'String','', ...
        'BackgroundColor',theme.bg, ...
        'ForegroundColor',theme.muted, ...
        'FontSize',8.0, ...
        'HorizontalAlignment','right');

    panelDist = uipanel(fig,'Units','normalized', ...
        'Position',[0.05 0.635 0.27 0.255], ...
        'BackgroundColor',theme.card, ...
        'BorderType','none');

    panelParab = uipanel(fig,'Units','normalized', ...
        'Position',[0.365 0.635 0.29 0.255], ...
        'BackgroundColor',theme.card, ...
        'BorderType','none');

    panelRel = uipanel(fig,'Units','normalized', ...
        'Position',[0.70 0.635 0.25 0.255], ...
        'BackgroundColor',theme.card, ...
        'BorderType','none');

    H.axDist = axes('Parent',panelDist,'Units','normalized', ...
        'Position',[0.13 0.17 0.80 0.74]);
    H.axParab = axes('Parent',panelParab,'Units','normalized', ...
        'Position',[0.17 0.17 0.78 0.74]);
    H.axRel = axes('Parent',panelRel,'Units','normalized', ...
        'Position',[0.17 0.17 0.76 0.74]);

    styleAxes(H.axDist,'iGluSnFR ΔF/F₀','P(k)','Quantal fluorescence distribution',theme);
    styleAxes(H.axParab,'Mean ΔF/F₀','Variance (ΔF/F₀)^2','Variance-mean parabola',theme);
    styleAxes(H.axRel,'x = mu / mu0','y = (Var/mu)/(Var0/mu0)', ...
        'Relative N-P-Q change',theme);

    panelLegend = uipanel(fig,'Units','normalized', ...
        'Position',[0.05 0.555 0.90 0.055], ...
        'BackgroundColor',theme.card, ...
        'BorderType','none');

    H.axLegend = axes('Parent',panelLegend,'Units','normalized', ...
        'Position',[0.02 0.10 0.96 0.80], ...
        'Color',theme.card, ...
        'XColor',theme.card, ...
        'YColor',theme.card);
    hold(H.axLegend,'on');
    H.legendHandles = [ ...
        plot(H.axLegend,NaN,NaN,'Color',theme.blue,'LineWidth',2.2), ...
        plot(H.axLegend,NaN,NaN,'Color',theme.green,'LineWidth',2.2), ...
        plot(H.axLegend,NaN,NaN,'Color',theme.slope,'LineStyle','--','LineWidth',1.6), ...
        plot(H.axLegend,NaN,NaN,'Color',theme.orange,'LineStyle','--','LineWidth',1.6), ...
        plot(H.axLegend,NaN,NaN,'o','Color',theme.orange,'MarkerFaceColor',theme.orange,'LineStyle','none','MarkerSize',7), ...
        plot(H.axLegend,NaN,NaN,'o','Color',theme.green,'MarkerFaceColor',theme.green,'LineStyle','none','MarkerSize',7), ...
        plot(H.axLegend,NaN,NaN,'Color',theme.blue,'LineStyle',':','LineWidth',1.7), ...
        plot(H.axLegend,NaN,NaN,'Color',theme.green,'LineStyle',':','LineWidth',1.7), ...
        plot(H.axLegend,NaN,NaN,'s','Color',theme.fail,'MarkerFaceColor',theme.fail,'LineStyle','none','MarkerSize',7), ...
        plot(H.axLegend,NaN,NaN,'Color',theme.envelope,'LineWidth',1.8)];

    H.mainLegend = legend(H.axLegend,H.legendHandles,{ ...
        'homogeneous', ...
        'heterogeneous', ...
        'slope = Q', ...
        'apparent slope = Qapp', ...
        'op. point', ...
        'apparent op. point', ...
        'Imax', ...
        'Imax,app', ...
        'failures', ...
        'Q variability envelope'}, ...
        'Location','north', ...
        'Orientation','horizontal', ...
        'Box','off', ...
        'FontSize',8);
    try
        set(H.mainLegend,'NumColumns',5);
    catch
    end
    axis(H.axLegend,'off');

    cardX = [0.05 0.28 0.51 0.74];
    cardW = 0.21;
    cardH = 0.095;
    H.cards.failure = makeCard([cardX(1) 0.430 cardW cardH], ...
        'P(failure)',theme.fail,'failure',theme);
    H.cards.mean = makeCard([cardX(2) 0.430 cardW cardH], ...
        'mu = N * pbar * Q',theme.blue,'mean',theme);
    H.cards.variance = makeCard([cardX(3) 0.430 cardW cardH], ...
        'sigma^2 (true)',theme.green,'variance',theme);
    H.cards.cv = makeCard([cardX(4) 0.430 cardW cardH], ...
        'CV = sigma / mu',theme.slope,'cv',theme);

    H.biasTitle = uicontrol(fig,'Style','text','Units','normalized', ...
        'Position',[0.05 0.393 0.55 0.024], ...
        'String','Naive parabola fit bias - Clements and Silver 2000; Silver 2003', ...
        'BackgroundColor',theme.bg, ...
        'ForegroundColor',theme.text, ...
        'FontSize',10.5, ...
        'FontWeight','bold', ...
        'HorizontalAlignment','left');

    H.cards.qapp = makeCard([cardX(1) 0.292 cardW cardH], ...
        'Qapp = Q(1+CVQ^2)',theme.orange,'qapp',theme);
    H.cards.napp = makeCard([cardX(2) 0.292 cardW cardH], ...
        'Napp = N/(1+CVp^2)',theme.green,'napp',theme);
    H.cards.papp = makeCard([cardX(3) 0.292 cardW cardH], ...
        'papp estimate',theme.purple,'papp',theme);
    H.cards.imax = makeCard([cardX(4) 0.292 cardW cardH], ...
        'Imax,app vs Imax',theme.slope,'imax',theme);

    H.ctrlN = makeSliderControl(fig,theme,0.05,0.205,0.27, ...
        'N - apparent competent sites',1,20,defaults.N,[0.1/19 1/19],@sliderChanged);
    H.ctrlP = makeSliderControl(fig,theme,0.37,0.205,0.27, ...
        'pbar - mean release prob.',0.02,0.98,defaults.pbar,[0.02/0.96 0.10/0.96],@sliderChanged);
    H.ctrlQ = makeSliderControl(fig,theme,0.69,0.205,0.23, ...
        'Q - quantal iGluSnFR signal (ΔF/F₀)',0.05,3.0,defaults.Q,[0.05/2.95 0.25/2.95],@sliderChanged);

    H.chkHet = uicontrol(fig,'Style','checkbox','Units','normalized', ...
        'Position',[0.05 0.165 0.28 0.030], ...
        'String','Show parameter heterogeneity', ...
        'Value',double(defaults.showHet), ...
        'BackgroundColor',theme.bg, ...
        'ForegroundColor',theme.text, ...
        'FontWeight','bold', ...
        'Callback',@sliderChanged);

    H.chkTrain = uicontrol(fig,'Style','checkbox','Units','normalized', ...
        'Position',[0.335 0.165 0.20 0.030], ...
        'String','Train mode (new window)', ...
        'Value',0, ...
        'BackgroundColor',theme.bg, ...
        'ForegroundColor',theme.text, ...
        'FontWeight','bold', ...
        'Callback',@trainModeChanged);

    H.chkRep = uicontrol(fig,'Style','checkbox','Units','normalized', ...
        'Position',[0.545 0.165 0.24 0.030], ...
        'String','Repetition mode (new window)', ...
        'Value',0, ...
        'BackgroundColor',theme.bg, ...
        'ForegroundColor',theme.text, ...
        'FontWeight','bold', ...
        'Callback',@repModeChanged);

    H.btnReset = makeFlatButton(fig,theme,[0.80 0.163 0.10 0.028], ...
        'Reset defaults',@resetDashboard,'right');

    H.btnClose = makeFlatButton(fig,theme,[0.91 0.163 0.04 0.028], ...
        'Close',@(src,evt) close(fig),'right');

    H.ctrlCvQ = makeSliderControl(fig,theme,0.05,0.080,0.40, ...
        'CVQ - intrasite quantal variability',0.00,0.80,defaults.cvQ,[0.05/0.80 0.20/0.80],@sliderChanged);
    H.ctrlCvP = makeSliderControl(fig,theme,0.55,0.080,0.40, ...
        'CVp - intersite p variability',0.00,0.80,defaults.cvP,[0.05/0.80 0.20/0.80],@sliderChanged);

    H.infoQ = uicontrol(fig,'Style','text','Units','normalized', ...
        'Position',[0.05 0.012 0.40 0.050], ...
        'String','Vesicle-to-vesicle Q scatter steepens the initial slope to Qapp and broadens the amplitude peaks.', ...
        'BackgroundColor',theme.bg, ...
        'ForegroundColor',theme.muted, ...
        'FontSize',9, ...
        'HorizontalAlignment','left');

    H.infoP = uicontrol(fig,'Style','text','Units','normalized', ...
        'Position',[0.55 0.012 0.40 0.050], ...
        'String','Site-to-site p scatter sharpens curvature, narrows the apparent arch, and shifts Imax,app leftward.', ...
        'BackgroundColor',theme.bg, ...
        'ForegroundColor',theme.muted, ...
        'FontSize',9, ...
        'HorizontalAlignment','left');

    refreshDashboard();

    if startTrainMode
        set(H.chkTrain,'Value',1);
        trainModeChanged();
    end

    function sliderChanged(~,~)
        refreshDashboard();
    end

    function resetDashboard(~,~)
        setSliderValue(H.ctrlN,defaults.N,false);
        setSliderValue(H.ctrlP,defaults.pbar,false);
        setSliderValue(H.ctrlQ,defaults.Q,false);
        set(H.chkHet,'Value',double(defaults.showHet));
        setSliderValue(H.ctrlCvQ,defaults.cvQ,false);
        setSliderValue(H.ctrlCvP,defaults.cvP,false);
        refreshDashboard();
    end

    function trainModeChanged(~,~)
        Train.enabled = logical(get(H.chkTrain,'Value'));
        if Train.enabled
            ensureTrainWindow();
        else
            closeTrainWindow();
        end
        refreshDashboard();
    end

    function closeTrainWindow(~,~)
        if isgraphics(Train.fig)
            set(Train.fig,'CloseRequestFcn','');
            delete(Train.fig);
        end
        Train.fig = [];
        Train.handles = struct();
        Train.data = [];
        Train.enabled = false;
        if isgraphics(H.chkTrain)
            set(H.chkTrain,'Value',0);
        end
        if isgraphics(fig) && strcmp(get(fig,'BeingDeleted'),'off')
            refreshDashboard();
        end
    end

    function repModeChanged(~,~)
        Rep.enabled = logical(get(H.chkRep,'Value'));
        if Rep.enabled
            ensureRepWindow();
        else
            closeRepWindow();
        end
        refreshDashboard();
    end

    function closeRepWindow(~,~)
        if isgraphics(Rep.fig)
            set(Rep.fig,'CloseRequestFcn','');
            delete(Rep.fig);
        end
        Rep.fig = [];
        Rep.handles = struct();
        Rep.data = [];
        Rep.enabled = false;
        if isgraphics(H.chkRep)
            set(H.chkRep,'Value',0);
        end
        if isgraphics(fig) && strcmp(get(fig,'BeingDeleted'),'off')
            refreshDashboard();
        end
    end

    function closeMainFigure(~,~)
        closeTrainWindow();
        closeRepWindow();
        delete(fig);
    end

    function ensureTrainWindow
        if isgraphics(Train.fig)
            figure(Train.fig);
            return;
        end

        screenSizeLocal = get(groot,'ScreenSize');
        trainWidth = min(1160,screenSizeLocal(3) - 160);
        trainHeight = min(840,screenSizeLocal(4) - 140);
        trainLeft = max(60,round((screenSizeLocal(3) - trainWidth) / 2) + 40);
        trainBottom = max(40,round((screenSizeLocal(4) - trainHeight) / 2) - 20);

        Train.fig = figure( ...
            'Name','Train mode - dynamic release model', ...
            'NumberTitle','off', ...
            'Color',theme.bg, ...
            'Units','pixels', ...
            'Position',[trainLeft trainBottom trainWidth trainHeight], ...
            'MenuBar','none', ...
            'ToolBar','none', ...
            'CloseRequestFcn',@closeTrainWindow);

        uicontrol(Train.fig,'Style','text','Units','normalized', ...
            'Position',[0.05 0.94 0.44 0.035], ...
            'String','Train mode - dynamic release extension', ...
            'BackgroundColor',theme.bg, ...
            'ForegroundColor',theme.text, ...
            'FontSize',14, ...
            'FontWeight','bold', ...
            'HorizontalAlignment','left');

        Train.handles.txtBase = uicontrol(Train.fig,'Style','text','Units','normalized', ...
            'Position',[0.05 0.908 0.70 0.022], ...
            'String','', ...
            'BackgroundColor',theme.bg, ...
            'ForegroundColor',theme.muted, ...
            'FontSize',9.0, ...
            'HorizontalAlignment','left');

        Train.handles.txtRule = uicontrol(Train.fig,'Style','text','Units','normalized', ...
            'Position',[0.05 0.882 0.70 0.020], ...
            'String','Ready-sites plot: dotted orange = latent recruitment command (set by tau_N, Nmax); solid blue = actual pool after depletion/refill.', ...
            'BackgroundColor',theme.bg, ...
            'ForegroundColor',theme.muted, ...
            'FontSize',8.4, ...
            'HorizontalAlignment','left');

        Train.handles.btnClose = makeFlatButton(Train.fig,theme,[0.90 0.944 0.05 0.025], ...
            'Close',@closeTrainWindow,'right');

        Train.handles.axPr = axes('Parent',Train.fig,'Units','normalized', ...
            'Position',[0.055 0.680 0.185 0.150]);
        Train.handles.axReady = axes('Parent',Train.fig,'Units','normalized', ...
            'Position',[0.295 0.680 0.185 0.150]);
        Train.handles.axPPR = axes('Parent',Train.fig,'Units','normalized', ...
            'Position',[0.535 0.680 0.185 0.150]);
        Train.handles.axCum = axes('Parent',Train.fig,'Units','normalized', ...
            'Position',[0.775 0.680 0.185 0.150]);

        styleAxes(Train.handles.axPr,'Stimulus','Pr','Release probability',theme);
        styleAxes(Train.handles.axReady,'Stimulus','N','Release-competent sites',theme);
        styleAxes(Train.handles.axPPR,'Stimulus','A_k / A_1','Paired-pulse ratio (PPR)',theme);
        styleAxes(Train.handles.axCum,'Stimulus','Norm cumulative','Cumulative output',theme);

        % Full-width reconstructed 72A-fast iGluSnFR fluorescence train.
        Train.handles.axSensor = axes('Parent',Train.fig,'Units','normalized', ...
            'Position',[0.06 0.475 0.88 0.155]);
        styleAxes(Train.handles.axSensor,'Time (ms)','iGluSnFR ΔF/F₀', ...
            'Predicted 72A-fast iGluSnFR fluorescence train',theme);

        Train.handles.txtSummary = uicontrol(Train.fig,'Style','text','Units','normalized', ...
            'Position',[0.06 0.408 0.88 0.030], ...
            'String','', ...
            'BackgroundColor',theme.bg, ...
            'ForegroundColor',theme.text, ...
            'FontSize',9.2, ...
            'HorizontalAlignment','left');

        Train.handles.txtSecondary = uicontrol(Train.fig,'Style','text','Units','normalized', ...
            'Position',[0.06 0.380 0.88 0.024], ...
            'String','', ...
            'BackgroundColor',theme.bg, ...
            'ForegroundColor',theme.muted, ...
            'FontSize',8.6, ...
            'HorizontalAlignment','left');

        Train.handles.ctrlStim = makeSliderControl(Train.fig,theme,0.06,0.295,0.40, ...
            'Stimuli in train',2,20,Train.defaults.nStim,[1/18 2/18],@trainParameterChanged);
        Train.handles.ctrlFreq = makeSliderControl(Train.fig,theme,0.54,0.295,0.40, ...
            'Frequency (Hz)',5,100,Train.defaults.freqHz,[1/95 10/95],@trainParameterChanged);
        Train.handles.ctrlNmax = makeSliderControl(Train.fig,theme,0.06,0.230,0.40, ...
            'Latent Nmax / N1',1.0,8.0,Train.defaults.nMaxFactor,[0.05/7.0 0.25/7.0],@trainParameterChanged);
        Train.handles.ctrlTauN = makeSliderControl(Train.fig,theme,0.54,0.230,0.40, ...
            'tau_N (site rise, stim)',0.25,6.0,Train.defaults.tauNStim,[0.05/5.75 0.25/5.75],@trainParameterChanged);
        Train.handles.ctrlPmax = makeSliderControl(Train.fig,theme,0.06,0.165,0.40, ...
            'Pmax / P1 (Pr facilitation)',0.2,3.0,Train.defaults.pMaxFactor,[0.05/2.8 0.2/2.8],@trainParameterChanged);
        Train.handles.ctrlTauP = makeSliderControl(Train.fig,theme,0.54,0.165,0.40, ...
            'tau_P (Pr rise, stim)',0.25,8.0,Train.defaults.tauPStim,[0.05/7.75 0.25/7.75],@trainParameterChanged);
        Train.handles.ctrlRefill = makeSliderControl(Train.fig,theme,0.06,0.100,0.40, ...
            'RR1 (refill after pulse 1)',0.0,15.0,Train.defaults.refillPerStim,[0.1/15 0.5/15],@trainParameterChanged);
        Train.handles.ctrlRefillSteady = makeSliderControl(Train.fig,theme,0.54,0.100,0.40, ...
            'RR steady state (ves/stim)',0.0,15.0,Train.defaults.refillSteadyPerStim,[0.1/15 0.5/15],@trainParameterChanged);
        Train.handles.ctrlTauRefill = makeSliderControl(Train.fig,theme,0.06,0.035,0.40, ...
            'tau_RR (refill evolution, stim)',0.25,12.0,Train.defaults.tauRefillStim,[0.05/11.75 0.25/11.75],@trainParameterChanged);
    end

    function trainParameterChanged(~,~)
        refreshDashboard();
    end

    function trainPars = currentTrainParameters
        trainPars = Train.defaults;
        if isempty(fieldnames(Train.handles)) || ~isgraphics(Train.fig)
            return;
        end

        trainPars.nStim = round(getSliderValue(Train.handles.ctrlStim));
        setSliderValue(Train.handles.ctrlStim,trainPars.nStim,false);

        trainPars.freqHz = round(getSliderValue(Train.handles.ctrlFreq));
        setSliderValue(Train.handles.ctrlFreq,trainPars.freqHz,false);

        trainPars.refillPerStim = round(getSliderValue(Train.handles.ctrlRefill) * 10) / 10;
        setSliderValue(Train.handles.ctrlRefill,trainPars.refillPerStim,false);

        trainPars.refillSteadyPerStim = round(getSliderValue(Train.handles.ctrlRefillSteady) * 10) / 10;
        setSliderValue(Train.handles.ctrlRefillSteady,trainPars.refillSteadyPerStim,false);

        trainPars.tauRefillStim = round(getSliderValue(Train.handles.ctrlTauRefill) * 100) / 100;
        setSliderValue(Train.handles.ctrlTauRefill,trainPars.tauRefillStim,false);

        trainPars.nMaxFactor = round(getSliderValue(Train.handles.ctrlNmax) * 100) / 100;
        setSliderValue(Train.handles.ctrlNmax,trainPars.nMaxFactor,false);

        trainPars.tauNStim = round(getSliderValue(Train.handles.ctrlTauN) * 100) / 100;
        setSliderValue(Train.handles.ctrlTauN,trainPars.tauNStim,false);

        trainPars.pMaxFactor = round(getSliderValue(Train.handles.ctrlPmax) * 100) / 100;
        setSliderValue(Train.handles.ctrlPmax,trainPars.pMaxFactor,false);

        trainPars.tauPStim = round(getSliderValue(Train.handles.ctrlTauP) * 100) / 100;
        setSliderValue(Train.handles.ctrlTauP,trainPars.tauPStim,false);
    end

    function trainData = simulateTrainModel(pars,cvQTrain,cvPTrain,trainPars)
        % Mechanistic recurrence comes from the shared graphics-free core so
        % the GUI and MPFA_FIT_TRAIN evaluate exactly the same model.
        corePars = pars;
        corePars.cvQ = cvQTrain;
        corePars.cvP = cvPTrain;
        core = mpfa_train_core(corePars,trainPars);

        linearMeanResp = core.meanResp;
        meanResp = core.meanResp;
        isiMs = core.isiMs;

        % ---- Reconstructed 72A-fast iGluSnFR fluorescence train ----
        % Each stimulus injects a unit-peak dual-exponential sensor transient
        % scaled by the release-linear fluorescence response.
        tauRise = max(trainPars.tauRiseMs,0.02);
        tauDecay = max(trainPars.tauDecayMs,1.5 * tauRise);
        baselineMs = max(trainPars.baselineMs,0);
        onsetTimes = baselineMs + (0:trainPars.nStim - 1) * isiMs;
        tEnd = onsetTimes(end) + 6 * tauDecay + 2;
        sensorTime = linspace(0,tEnd,1400);
        tPeak = (tauDecay * tauRise / (tauDecay - tauRise)) * log(tauDecay / tauRise);
        kernelNorm = exp(-tPeak / tauDecay) - exp(-tPeak / tauRise);
        sensorTrace = zeros(size(sensorTime));
        for kk = 1:trainPars.nStim
            dt = sensorTime - onsetTimes(kk);
            active = dt >= 0;
            sensorTrace(active) = sensorTrace(active) + meanResp(kk) * ...
                (exp(-dt(active) / tauDecay) - exp(-dt(active) / tauRise)) / kernelNorm;
        end

        trainData = struct( ...
            'stim',core.stim, ...
            'latentN',core.latentN, ...
            'nMax',core.nMax, ...
            'pr',core.pr, ...
            'readyBefore',core.readyBefore, ...
            'readyAfter',core.readyAfter, ...
            'refillPerInterval',core.refillPerInterval, ...
            'releasedQuanta',core.released, ...
            'linearMeanResp',linearMeanResp, ...
            'meanResp',meanResp, ...
            'varResp',core.varResp, ...
            'normCumRelease',core.normCumRelease, ...
            'relativeX',core.relativeX, ...
            'releasePpr',core.releasePpr, ...
            'relativeY',core.relativeY, ...
            'isiMs',isiMs, ...
            'refillPerSecond',core.refillPerInterval * trainPars.freqHz, ...
            'sensorTime',sensorTime, ...
            'sensorTrace',sensorTrace, ...
            'sensorOnsets',onsetTimes, ...
            'sensorPeakTimes',onsetTimes + tPeak, ...
            'sensorTauRise',tauRise, ...
            'sensorTauDecay',tauDecay);
    end

    function refreshTrainWindow(pars,trainPars,trainData)
        if ~Train.enabled || ~isgraphics(Train.fig)
            return;
        end

        set(Train.handles.txtBase,'String',sprintf( ...
            'A1 inherited: apparent N=%.2f, P1=%.2f, Q=%.2f ΔF/F₀ per quantum', ...
            pars.N,pars.pbar,pars.Q));

        cla(Train.handles.axPr);
        cla(Train.handles.axReady);
        cla(Train.handles.axPPR);
        cla(Train.handles.axCum);
        cla(Train.handles.axSensor);
        hold(Train.handles.axPr,'on');
        hold(Train.handles.axReady,'on');
        hold(Train.handles.axPPR,'on');
        hold(Train.handles.axCum,'on');
        hold(Train.handles.axSensor,'on');

        % Measured target (if a fit/preset supplied one) as a faint shaded
        % band behind the model, so model vs data is directly comparable.
        hMeasuredPpr = gobjects(0);
        if ~isempty(Train.target)
            plotTargetBand(Train.handles.axPr,Train.target,'pr',theme.fail);
            plotTargetBand(Train.handles.axReady,Train.target,'N',theme.blue);
            hMeasuredPpr = plotTargetBand(Train.handles.axPPR,Train.target,'ppr',theme.purple);
        end

        plot(Train.handles.axPr,trainData.stim,trainData.pr,'-o', ...
            'Color',theme.fail,'MarkerFaceColor',theme.fail,'LineWidth',1.8,'MarkerSize',5.5);
        % Latent recruitment "command" set by tau_N and Nmax (dotted orange),
        % against the actual depletion/refill-limited ready pool (solid blue).
        plot(Train.handles.axReady,trainData.stim,trainData.latentN,':', ...
            'Color',theme.orange,'LineWidth',1.6,'Marker','none');
        plot(Train.handles.axReady,trainData.stim,trainData.nMax * ones(size(trainData.stim)),'--', ...
            'Color',theme.slope,'LineWidth',1.1);
        plot(Train.handles.axReady,trainData.stim,trainData.readyBefore,'-o', ...
            'Color',theme.blue,'MarkerFaceColor',theme.blue,'LineWidth',1.8,'MarkerSize',5.5);
        hPredictedPpr = plot(Train.handles.axPPR,trainData.stim,trainData.releasePpr,'-o', ...
            'Color',theme.purple,'MarkerFaceColor',theme.purple,'LineWidth',1.8,'MarkerSize',5.5);
        plot(Train.handles.axPPR,[1 trainPars.nStim],[1 1],':', ...
            'Color',theme.slope,'LineWidth',1.0,'HandleVisibility','off');
        if ~isempty(hMeasuredPpr) && isgraphics(hMeasuredPpr)
            legend(Train.handles.axPPR,[hMeasuredPpr hPredictedPpr], ...
                {'measured','predicted'}, ...
                'Location','southeast','Box','off','FontSize',7.2);
        else
            legend(Train.handles.axPPR,hPredictedPpr,'predicted', ...
                'Location','southeast','Box','off','FontSize',7.2);
        end
        plot(Train.handles.axCum,trainData.stim,trainData.normCumRelease,'-o', ...
            'Color',theme.green,'MarkerFaceColor',theme.green,'LineWidth',1.8,'MarkerSize',5.5);

        % Predicted 72A-fast iGluSnFR train and isolated fluorescence peaks.
        plot(Train.handles.axSensor,trainData.sensorTime,trainData.sensorTrace,'-', ...
            'Color',theme.train,'LineWidth',1.6);
        stem(Train.handles.axSensor,trainData.sensorPeakTimes,trainData.meanResp,':', ...
            'Color',theme.orange,'Marker','o','MarkerFaceColor',theme.orange, ...
            'MarkerSize',4.5,'LineWidth',1.0,'BaseValue',0);

        tgtNmax = targetMax(Train.target,'N');
        tgtPPRmax = targetMax(Train.target,'ppr');
        set(Train.handles.axPr,'XLim',[1 trainPars.nStim],'YLim',[0 1]);
        set(Train.handles.axReady,'XLim',[1 trainPars.nStim], ...
            'YLim',[0 max([1.05 * trainData.nMax, 1.05 * tgtNmax, 1])]);
        set(Train.handles.axPPR,'XLim',[1 trainPars.nStim], ...
            'YLim',[0 max([1.15 * max(trainData.releasePpr), 1.15 * tgtPPRmax, 1.3])]);
        set(Train.handles.axCum,'XLim',[1 trainPars.nStim], ...
            'YLim',[0 max(1.15 * max(trainData.normCumRelease),1.2)]);
        sensorYMax = 1.15 * max([trainData.sensorTrace trainData.meanResp 1]);
        set(Train.handles.axSensor,'XLim',[0 max(trainData.sensorTime(end),1)], ...
            'YLim',[0 sensorYMax]);

        ppr = trainData.meanResp(min(2,numel(trainData.meanResp))) / max(trainData.meanResp(1),eps);
        set(Train.handles.txtSummary,'String',sprintf([ ...
            'A1 = %.2f ΔF/F₀ | last pulse = %.2f ΔF/F₀ | PPR(2/1) = %.2f | ' ...
            'total released quanta = %.2f'], ...
            trainData.meanResp(1),trainData.meanResp(end),ppr,sum(trainData.releasedQuanta)));
        set(Train.handles.txtSecondary,'String',sprintf([ ...
            '%d stimuli at %d Hz (ISI %.1f ms) | RR %.1f -> %.1f ves/stim (tau_RR %.2f) | latent Nmax = %.2fx N1 (tau_N %.2f) | Pmax = %.2fx P1 (tau_P %.2f) | sensor tau_r/d = %.1f/%.1f ms'], ...
            trainPars.nStim,trainPars.freqHz,trainData.isiMs, ...
            trainPars.refillPerStim,trainPars.refillSteadyPerStim,trainPars.tauRefillStim, ...
            trainPars.nMaxFactor,trainPars.tauNStim, ...
            trainPars.pMaxFactor,trainPars.tauPStim,trainData.sensorTauRise,trainData.sensorTauDecay));
    end

    function ensureRepWindow
        if isgraphics(Rep.fig)
            figure(Rep.fig);
            return;
        end

        screenSizeLocal = get(groot,'ScreenSize');
        repWidth = min(1080,screenSizeLocal(3) - 200);
        repHeight = min(660,screenSizeLocal(4) - 160);
        repLeft = max(40,round((screenSizeLocal(3) - repWidth) / 2) - 40);
        repBottom = max(40,round((screenSizeLocal(4) - repHeight) / 2) - 60);

        Rep.fig = figure( ...
            'Name','Repetition mode - trial-to-trial variability', ...
            'NumberTitle','off', ...
            'Color',theme.bg, ...
            'Units','pixels', ...
            'Position',[repLeft repBottom repWidth repHeight], ...
            'MenuBar','none', ...
            'ToolBar','none', ...
            'CloseRequestFcn',@closeRepWindow);

        uicontrol(Rep.fig,'Style','text','Units','normalized', ...
            'Position',[0.05 0.93 0.60 0.045], ...
            'String','Repetition mode - trial-to-trial variability', ...
            'BackgroundColor',theme.bg,'ForegroundColor',theme.text, ...
            'FontSize',14,'FontWeight','bold','HorizontalAlignment','left');

        Rep.handles.txtBase = uicontrol(Rep.fig,'Style','text','Units','normalized', ...
            'Position',[0.05 0.895 0.90 0.026], ...
            'String','','BackgroundColor',theme.bg,'ForegroundColor',theme.muted, ...
            'FontSize',9.0,'HorizontalAlignment','left');

        Rep.handles.btnClose = makeFlatButton(Rep.fig,theme,[0.90 0.945 0.05 0.028], ...
            'Close',@closeRepWindow,'right');

        Rep.handles.axOverlay = axes('Parent',Rep.fig,'Units','normalized', ...
            'Position',[0.07 0.48 0.52 0.36]);
        Rep.handles.axDistr = axes('Parent',Rep.fig,'Units','normalized', ...
            'Position',[0.68 0.48 0.28 0.36]);
        styleAxes(Rep.handles.axOverlay,'Time (ms)','iGluSnFR ΔF/F₀', ...
            'Overlaid 72A-fast fluorescence repetitions',theme);
        styleAxes(Rep.handles.axDistr,'Event ΔF/F₀','Probability density', ...
            'Predicted fluorescence distribution + noise',theme);

        Rep.handles.txtSummary = uicontrol(Rep.fig,'Style','text','Units','normalized', ...
            'Position',[0.07 0.365 0.89 0.028], ...
            'String','','BackgroundColor',theme.bg,'ForegroundColor',theme.text, ...
            'FontSize',9.2,'HorizontalAlignment','left');

        Rep.handles.ctrlNrep = makeSliderControl(Rep.fig,theme,0.07,0.245,0.40, ...
            'Repetitions',5,300,Rep.defaults.nRep,[5/295 25/295],@repParameterChanged);
        Rep.handles.ctrlNoise = makeSliderControl(Rep.fig,theme,0.55,0.245,0.40, ...
            'Fluorescence noise SD (ΔF/F₀)',0.0,1.0,Rep.defaults.noiseSd,[0.025 0.10],@repParameterChanged);
        Rep.handles.ctrlEvent = makeSliderControl(Rep.fig,theme,0.07,0.130,0.40, ...
            'Histogram event (stimulus #)',1,20,Rep.defaults.eventIndex,[1/19 2/19],@repParameterChanged);

        Rep.handles.note = uicontrol(Rep.fig,'Style','text','Units','normalized', ...
            'Position',[0.07 0.020 0.89 0.085], ...
            'String',['Each repetition draws a binomial number of released quanta ' ...
            '(k ~ Bin(N, p)), sums k quantal fluorescence signals with Q scatter, and adds ' ...
            'Gaussian recording noise. The right panel overlays the analytic mixture ' ...
            'sum_k Bin(N,k,p) x Normal(kQ, sqrt(k (CVq Q)^2 + noise^2)): the peak at 0 ' ...
            'is release failures, the higher peaks are 1, 2, ... released quanta. ' ...
            'With Train mode on, the protocol is the full train and the histogram uses ' ...
            'the selected stimulus number.'], ...
            'BackgroundColor',theme.bg,'ForegroundColor',theme.muted, ...
            'FontSize',8.4,'HorizontalAlignment','left');
    end

    function repParameterChanged(~,~)
        refreshDashboard();
    end

    function repPars = currentRepParameters
        repPars = Rep.defaults;
        if isempty(fieldnames(Rep.handles)) || ~isgraphics(Rep.fig)
            return;
        end
        repPars.nRep = round(getSliderValue(Rep.handles.ctrlNrep));
        setSliderValue(Rep.handles.ctrlNrep,repPars.nRep,false);
        repPars.noiseSd = round(getSliderValue(Rep.handles.ctrlNoise) * 1000) / 1000;
        setSliderValue(Rep.handles.ctrlNoise,repPars.noiseSd,false);
        repPars.eventIndex = round(getSliderValue(Rep.handles.ctrlEvent));
        setSliderValue(Rep.handles.ctrlEvent,repPars.eventIndex,false);
    end

    function repData = simulateRepetitions(pars,cvQ,repPars,trainData)
        % Deterministic seed so the overlay is stable when unrelated sliders move.
        rng(20240517,'twister');

        nRep = max(round(repPars.nRep),1);
        noiseSd = max(repPars.noiseSd,0);
        maxShow = min(nRep,40);           % cap drawn traces for speed
        tauRise = max(Train.defaults.tauRiseMs,0.02);
        tauDecay = max(Train.defaults.tauDecayMs,1.5 * tauRise);
        tPeak = (tauDecay * tauRise / (tauDecay - tauRise)) * log(tauDecay / tauRise);
        kernelNorm = exp(-tPeak / tauDecay) - exp(-tPeak / tauRise);

        if ~isempty(trainData)
            onsetTimes = trainData.sensorOnsets;
            timeAxis = trainData.sensorTime;
            nSitesPerStim = max(round(trainData.readyBefore),0);
            prPerStim = trainData.pr;
            nStim = numel(onsetTimes);
            evIdx = min(max(repPars.eventIndex,1),nStim);
        else
            nStim = 1;
            baselineMs = max(Train.defaults.baselineMs,0);
            onsetTimes = baselineMs;
            tEnd = baselineMs + 6 * tauDecay + 2;
            timeAxis = linspace(0,tEnd,500);
            nSitesPerStim = max(round(pars.N),1);
            prPerStim = pars.pbar;
            evIdx = 1;
        end

        traces = zeros(maxShow,numel(timeAxis));
        eventAmps = zeros(1,nRep);
        qScatter = cvQ * pars.Q;

        for r = 1:nRep
            pulseAmp = zeros(1,nStim);
            for s = 1:nStim
                nSites = nSitesPerStim(min(s,numel(nSitesPerStim)));
                pRel = prPerStim(min(s,numel(prPerStim)));
                kRel = sum(rand(1,max(nSites,0)) < pRel);   % Bin(nSites, pRel)
                pulseAmp(s) = kRel * pars.Q + sqrt(kRel) * qScatter * randn;
            end
            eventAmps(r) = pulseAmp(evIdx) + noiseSd * randn;
            if r <= maxShow
                trace = noiseSd * randn(1,numel(timeAxis));
                for s = 1:nStim
                    dt = timeAxis - onsetTimes(s);
                    active = dt >= 0;
                    trace(active) = trace(active) + pulseAmp(s) * ...
                        (exp(-dt(active) / tauDecay) - exp(-dt(active) / tauRise)) / kernelNorm;
                end
                traces(r,:) = trace;
            end
        end

        % Analytic amplitude mixture for the histogrammed event.
        nEv = nSitesPerStim(min(evIdx,numel(nSitesPerStim)));
        pEv = prPerStim(min(evIdx,numel(prPerStim)));
        pmfEv = binomialPmfVector(nEv,pEv);
        xMax = (nEv + 3.5) * pars.Q + 3 * noiseSd;
        xMin = -4 * max(noiseSd,0.5 * pars.Q);
        xGrid = linspace(xMin,xMax,400);
        components = zeros(nEv + 1,numel(xGrid));
        for k = 0:nEv
            componentMean = k * pars.Q;
            sdK = sqrt(k * qScatter^2 + noiseSd^2);
            sdK = max(sdK,1e-6);
            components(k + 1,:) = pmfEv(k + 1) * ...
                exp(-0.5 * ((xGrid - componentMean) / sdK).^2) / (sdK * sqrt(2 * pi));
        end
        mixture = sum(components,1);

        repData = struct( ...
            'timeAxis',timeAxis, ...
            'traces',traces, ...
            'meanTrace',mean(traces,1), ...
            'eventAmps',eventAmps, ...
            'xGrid',xGrid, ...
            'mixture',mixture, ...
            'components',components, ...
            'eventIndex',evIdx, ...
            'nEvent',nEv, ...
            'pEvent',pEv, ...
            'noiseSd',noiseSd, ...
            'nRep',nRep, ...
            'shownTraces',maxShow, ...
            'meanAmp',mean(eventAmps), ...
            'cvAmp',std(eventAmps) / max(mean(eventAmps),eps), ...
            'failRate',pmfEv(1));
    end

    function refreshRepWindow(pars,~,repData)
        if ~Rep.enabled || ~isgraphics(Rep.fig)
            return;
        end

        set(Rep.handles.txtBase,'String',sprintf( ...
            'Protocol: %s | apparent N=%.1f, p=%.2f, Q=%.2f ΔF/F₀ | 72A-fast', ...
            ternary(Rep.usesTrain,'train','single event'),pars.N,pars.pbar,pars.Q));

        cla(Rep.handles.axOverlay);
        cla(Rep.handles.axDistr);
        hold(Rep.handles.axOverlay,'on');
        hold(Rep.handles.axDistr,'on');

        for r = 1:repData.shownTraces
            plot(Rep.handles.axOverlay,repData.timeAxis,repData.traces(r,:), ...
                '-','Color',[theme.train 0.18],'LineWidth',0.5);
        end
        plot(Rep.handles.axOverlay,repData.timeAxis,repData.meanTrace, ...
            '-','Color',theme.train,'LineWidth',1.8);
        set(Rep.handles.axOverlay,'XLim',[repData.timeAxis(1) max(repData.timeAxis(end),1)]);

        % Sampled histogram (base MATLAB, pdf-normalized) + analytic mixture.
        histogram(Rep.handles.axDistr,repData.eventAmps,'Normalization','pdf', ...
            'FaceColor',theme.slope,'EdgeColor','none','FaceAlpha',0.35);
        for k = 1:size(repData.components,1)
            plot(Rep.handles.axDistr,repData.xGrid,repData.components(k,:), ...
                '--','Color',[theme.orange 0.55],'LineWidth',0.9);
        end
        plot(Rep.handles.axDistr,repData.xGrid,repData.mixture, ...
            '-','Color',theme.orange,'LineWidth',1.8);
        set(Rep.handles.axDistr,'XLim',[repData.xGrid(1) repData.xGrid(end)], ...
            'YLim',[0 1.15 * max([repData.mixture eps])]);

        set(Rep.handles.txtSummary,'String',sprintf([ ...
            'Event #%d: %d repetitions, noise SD = %.3f ΔF/F₀ | mean signal = %.2f ΔF/F₀ | ' ...
            'CV = %.3f | P(failure) = %.1f %% | showing %d overlaid traces'], ...
            repData.eventIndex,repData.nRep,repData.noiseSd,repData.meanAmp, ...
            repData.cvAmp,100 * repData.failRate,repData.shownTraces));
    end

    function pars = currentParameters
        pars.N = round(getSliderValue(H.ctrlN) * 10) / 10;
        setSliderValue(H.ctrlN,pars.N,false);

        pars.pbar = round(getSliderValue(H.ctrlP) * 100) / 100;
        setSliderValue(H.ctrlP,pars.pbar,false);

        pars.Q = round(getSliderValue(H.ctrlQ) * 100) / 100;
        setSliderValue(H.ctrlQ,pars.Q,false);

        pars.showHet = logical(get(H.chkHet,'Value'));

        pars.cvQ = round(getSliderValue(H.ctrlCvQ) * 100) / 100;
        setSliderValue(H.ctrlCvQ,pars.cvQ,false);

        pars.cvP = round(getSliderValue(H.ctrlCvP) * 100) / 100;
        setSliderValue(H.ctrlCvP,pars.cvP,false);
    end

    function refreshDashboard
        pars = currentParameters();
        hetActive = pars.showHet;
        cvQ = double(hetActive) * pars.cvQ;
        cvP = double(hetActive) * pars.cvP;
        trainPars = [];
        trainData = [];

        if Train.enabled
            ensureTrainWindow();
            trainPars = currentTrainParameters();
            trainData = simulateTrainModel(pars,cvQ,cvP,trainPars);
            Train.data = trainData;
        else
            Train.data = [];
        end

        if Rep.enabled
            ensureRepWindow();
            Rep.usesTrain = ~isempty(trainData);
            % The histogram-event slider only spans the available stimuli.
            nStimForRep = 1;
            if Rep.usesTrain
                nStimForRep = trainPars.nStim;
            end
            setSliderRange(Rep.handles.ctrlEvent,1,max(nStimForRep,1));
            setSliderEnabled(Rep.handles.ctrlEvent,Rep.usesTrain);
            repPars = currentRepParameters();
            repData = simulateRepetitions(pars,cvQ,repPars,trainData);
            Rep.data = repData;
            refreshRepWindow(pars,repPars,repData);
        else
            Rep.data = [];
        end

        trainMeanMax = 0;
        trainVarMax = 0;
        trainRelPeak = 1;
        if ~isempty(trainData)
            trainMeanMax = max(trainData.linearMeanResp);
            trainVarMax = max(trainData.varResp);
            trainRelPeak = max([trainData.relativeX trainData.relativeY 1]);
        end

        set(H.ctrlN.value,'String',sprintf('%.1f',pars.N));
        set(H.ctrlP.value,'String',sprintf('%.2f',pars.pbar));
        set(H.ctrlQ.value,'String',sprintf('%.2f',pars.Q));
        set(H.ctrlCvQ.value,'String',sprintf('%.2f',pars.cvQ));
        set(H.ctrlCvP.value,'String',sprintf('%.2f',pars.cvP));

        setSliderEnabled(H.ctrlCvQ,hetActive);
        setSliderEnabled(H.ctrlCvP,hetActive);
        set(H.ctrlCvQ.label,'ForegroundColor',ternaryColor(hetActive,theme.text,theme.disabled));
        set(H.ctrlCvP.label,'ForegroundColor',ternaryColor(hetActive,theme.text,theme.disabled));
        set(H.ctrlCvQ.value,'ForegroundColor',ternaryColor(hetActive,theme.text,theme.disabled));
        set(H.ctrlCvP.value,'ForegroundColor',ternaryColor(hetActive,theme.text,theme.disabled));
        set(H.infoQ,'Visible',ternary(hetActive,'on','off'));
        set(H.infoP,'Visible',ternary(hetActive,'on','off'));
        set(H.biasTitle,'Visible',ternary(hetActive,'on','off'));
        set(H.cards.qapp.panel,'Visible',ternary(hetActive,'on','off'));
        set(H.cards.napp.panel,'Visible',ternary(hetActive,'on','off'));
        set(H.cards.papp.panel,'Visible',ternary(hetActive,'on','off'));
        set(H.cards.imax.panel,'Visible',ternary(hetActive,'on','off'));

        % MPFA N is a continuous effective estimate; a literal binomial PMF
        % requires an integer site count, so only the discrete illustration
        % uses the nearest integer. Analytic mean/variance retain apparent N.
        nDiscrete = max(round(pars.N),1);
        pmf = binomialPmfVector(nDiscrete,pars.pbar);
        amplitudes = (0:nDiscrete) * pars.Q;
        barWidth = max(0.50,min(0.88,7.5 / (nDiscrete + 1)));
        refPmf = refCache.pmf;
        refAmplitudes = refCache.amplitudes;
        refBarWidth = min(0.96,barWidth + 0.18);

        mu = pars.N * pars.pbar * pars.Q;
        varianceHom = max(0,pars.Q * mu - mu^2 / pars.N);
        qApp = pars.Q * (1 + cvQ^2);
        varianceTrue = max(0,qApp * mu - (1 + cvP^2) * mu^2 / pars.N);
        sigmaTrue = sqrt(varianceTrue);
        overallCv = sigmaTrue / max(mu,eps);

        nApp = pars.N / (1 + cvP^2);
        pAppRaw = pars.pbar * (1 + cvP^2) / max(1 + cvQ^2,eps);
        pApp = min(max(pAppRaw,0),1);
        iMaxTrue = pars.N * pars.Q;
        iMaxApp = pars.N * qApp / max(1 + cvP^2,eps);

        pGrid = refCache.pGrid;
        meanGrid = pars.N * pars.Q * pGrid;
        varGridHom = max(0,pars.Q * meanGrid - meanGrid.^2 / pars.N);
        varGridHet = max(0,qApp * meanGrid - (1 + cvP^2) * meanGrid.^2 / pars.N);
        refMeanGrid = refCache.meanGrid;
        refVarGrid = refCache.varGrid;

        cla(H.axDist);
        hold(H.axDist,'on');
        bar(H.axDist,refAmplitudes,refPmf,refBarWidth, ...
            'FaceColor',theme.refFill, ...
            'EdgeColor',theme.refEdge, ...
            'LineWidth',0.6);
        bars = bar(H.axDist,amplitudes,pmf,barWidth,'FaceColor','flat','EdgeColor','none');
        barColors = repmat(theme.blue,numel(amplitudes),1);
        barColors(1,:) = theme.fail;
        bars.CData = barColors;

        envelopePeak = 0;
        if cvQ > 0.01
            xEnv = linspace(0.10 * pars.Q,(nDiscrete + 1.8) * pars.Q,320);
            envelope = zeros(size(xEnv));
            for k = 1:nDiscrete
                sigmaK = sqrt(k) * cvQ * pars.Q;
                z = (xEnv - k * pars.Q) / sigmaK;
                envelope = envelope + pmf(k + 1) * exp(-0.5 * z.^2) / (sigmaK * sqrt(2 * pi));
            end
            envelope = envelope * pars.Q;
            envelopePeak = max(envelope);
            plot(H.axDist,xEnv,envelope,'-','Color',theme.envelope,'LineWidth',1.8);
        end
        xlim(H.axDist,[-0.25 * min(pars.Q,ref.Q) max((pars.N + 1.8) * pars.Q,(ref.N + 1.8) * ref.Q)]);
        ylim(H.axDist,[0 1.22 * max([max(pmf) max(refPmf) envelopePeak 0.05])]);

        cla(H.axParab);
        hold(H.axParab,'on');
        if isempty(trainData)
            plot(H.axParab,refMeanGrid,refVarGrid,':','Color',theme.refEdge,'LineWidth',1.6);
        end
        hA1Parabola = plot(H.axParab,meanGrid,varGridHom,'-','Color',theme.blue,'LineWidth',2.2);
        if hetActive
            plot(H.axParab,meanGrid,varGridHet,'-','Color',theme.green,'LineWidth',2.2);
        end

        slopeLength = iMaxTrue * 0.38;
        plot(H.axParab,[0 slopeLength],[0 pars.Q * slopeLength],'--', ...
            'Color',theme.slope,'LineWidth',1.6);
        if hetActive
            plot(H.axParab,[0 slopeLength],[0 qApp * slopeLength],'--', ...
                'Color',theme.orange,'LineWidth',1.6);
        end

        yMaxParab = max(pars.N * pars.Q^2 / 4, pars.N * qApp^2 / (4 * max(1 + cvP^2,1)));
        plot(H.axParab,[iMaxTrue iMaxTrue],[0 0.90 * 1.22 * yMaxParab],':', ...
            'Color',theme.blue,'LineWidth',1.7);
        if hetActive && abs(iMaxApp - iMaxTrue) > 0.5
            plot(H.axParab,[iMaxApp iMaxApp],[0 0.90 * 1.22 * yMaxParab],':', ...
                'Color',theme.green,'LineWidth',1.7);
        end

        if isempty(trainData)
            scatter(H.axParab,mu,varianceHom,76,theme.orange,'filled');
            if hetActive
                scatter(H.axParab,mu,varianceTrue,70,theme.green,'filled');
            end
        end

        if ~isempty(trainData)
            maxCompetentN = max(trainData.readyBefore);
            trainEnvelopeMean = maxCompetentN * pars.Q * pGrid;
            trainEnvelopeVar = max(0,pars.Q * trainEnvelopeMean - ...
                trainEnvelopeMean.^2 / max(maxCompetentN,eps));
            hTrainEnvelope = plot(H.axParab,trainEnvelopeMean,trainEnvelopeVar,'--', ...
                'Color',theme.train,'LineWidth',1.6);
            hTrainPoints = plot(H.axParab,trainData.linearMeanResp,trainData.varResp,'-', ...
                'Color',theme.slope,'LineWidth',1.0);
            scatter(H.axParab,trainData.linearMeanResp,trainData.varResp,38, ...
                trainData.stim,'filled','MarkerEdgeColor',theme.card,'LineWidth',0.6);
            colormap(H.axParab,viridisMap(256));
            clim(H.axParab,[1 max(2,trainPars.nStim)]);
            legend(H.axParab,[hA1Parabola hTrainEnvelope hTrainPoints], ...
                {sprintf('A1 parabola (N=%.2f)',pars.N), ...
                 sprintf('max competent-N envelope (N=%.2f)',maxCompetentN), ...
                 'train operating points'}, ...
                'Location','northeast','Box','off','FontSize',7.2);
        end

        if isempty(trainData)
            xMaxParab = 1.10 * max([iMaxTrue iMaxApp pars.Q ref.N * ref.Q ref.Q]);
            yMaxParab = 1.15 * max([max(varGridHom) max(varGridHet) max(refVarGrid) varianceTrue varianceHom eps]);
        else
            xMaxParab = 1.10 * max([iMaxTrue iMaxApp pars.Q trainEnvelopeMean trainMeanMax]);
            yMaxParab = 1.15 * max([max(varGridHom) max(varGridHet) trainEnvelopeVar varianceTrue varianceHom trainVarMax eps]);
        end
        [xMaxParab,xTicksParab] = nicePositiveAxis(xMaxParab,5);
        [yMaxParab,yTicksParab] = nicePositiveAxis(yMaxParab,5);
        set(H.axParab,'XLim',[0 xMaxParab],'YLim',[0 yMaxParab], ...
            'XTick',xTicksParab,'YTick',yTicksParab);
        ytickformat(H.axParab,'%.3g');

        colorbar(H.axRel,'off');
        cla(H.axRel);
        hold(H.axRel,'on');
        mu0 = refCache.mu0;
        var0 = refCache.var0;
        xRel = mu / mu0;
        yRel = (varianceTrue / max(mu,eps)) / (var0 / mu0);
        if isempty(trainData)
            relPeak = max([xRel yRel 1]);
            guidePbar = ref.pbar;
        else
            relPeak = trainRelPeak;
            guidePbar = pars.pbar;
        end
        relRequired = max(relAxisMax,1.08 * relPeak);
        [xRelMax,relTicks] = nicePositiveAxis(relRequired,5);
        xGuide = linspace(0,xRelMax,240);
        xPmax = min(xRelMax,1 / guidePbar);
        xP = linspace(0,xPmax,180);
        yP = (1 - guidePbar * xP) / (1 - guidePbar);
        pColor = theme.purple;
        plot(H.axRel,xGuide,ones(size(xGuide)),'-','Color',theme.green,'LineWidth',1.6);
        plot(H.axRel,xGuide,xGuide,'-','Color',theme.blue,'LineWidth',1.6);
        plot(H.axRel,xP,yP,'-','Color',pColor,'LineWidth',1.6);
        scatter(H.axRel,1,1,42,theme.slope,'filled');
        if isempty(trainData)
            scatter(H.axRel,xRel,yRel,66,theme.orange,'filled');
            xlabel(H.axRel,'x = mu / mu0','Color',theme.text,'FontWeight','bold','FontSize',9.5);
            ylabel(H.axRel,'y = (Var/mu)/(Var0/mu0)','Color',theme.text,'FontWeight','bold','FontSize',8.4);
            set(H.relMeta,'String',sprintf( ...
                'Reference: N0=%d, p0=%.2f, Q0=%.2f ΔF/F₀',ref.N,ref.pbar,ref.Q));
            set(H.relSummary,'String',sprintf([ ...
                'N %.2f | p %.2f | Q %.2f | (x,y)=(%.2f, %.2f)'], ...
                pars.N / ref.N,pars.pbar / ref.pbar,pars.Q / ref.Q,xRel,yRel));
        else
            plot(H.axRel,trainData.relativeX,trainData.relativeY,'-', ...
                'Color',theme.slope,'LineWidth',1.0);
            scatter(H.axRel,trainData.relativeX,trainData.relativeY,38, ...
                trainData.stim,'filled','MarkerEdgeColor',theme.card,'LineWidth',0.6);
            colormap(H.axRel,viridisMap(256));
            clim(H.axRel,[1 max(2,trainPars.nStim)]);
            cb = colorbar(H.axRel,'eastoutside');
            cbTicks = unique(round(linspace(1,trainPars.nStim,min(4,trainPars.nStim))));
            set(cb,'Ticks',cbTicks,'Color',theme.axis,'FontSize',7.5,'Box','off');
            cb.Label.String = 'Stimulus';
            cb.Label.Color = theme.text;
            cb.Label.FontSize = 8;
            xlabel(H.axRel,'x = mu_k / mu_1','Color',theme.text,'FontWeight','bold','FontSize',9.5);
            ylabel(H.axRel,'y = (Var_k/mu_k)/(Var_1/mu_1)', ...
                'Color',theme.text,'FontWeight','bold','FontSize',8.4);
            set(H.relMeta,'String',sprintf( ...
                'Train reference: A1 (N1=%.2f, P1=%.2f, Q=%.2f ΔF/F₀)', ...
                pars.N,pars.pbar,pars.Q));
            set(H.relSummary,'String',sprintf( ...
                'A1 = (1,1) | viridis colors = stimuli 1-%d',trainPars.nStim));
        end
        set(H.axRel,'XLim',[0 xRelMax],'YLim',[0 xRelMax], ...
            'XTick',relTicks,'YTick',relTicks);
        xtickformat(H.axRel,'%.3g');
        ytickformat(H.axRel,'%.3g');

        set(H.cards.failure.value,'String',sprintf('%.1f %%',100 * pmf(1)));
        set(H.cards.failure.subtitle,'String','release failures');
        set(H.cards.mean.value,'String',sprintf('%.2f ΔF/F₀',mu));
        set(H.cards.mean.subtitle,'String','expected mean fluorescence');
        set(H.cards.variance.value,'String',sprintf('%.3f (ΔF/F₀)^2',varianceTrue));
        set(H.cards.variance.subtitle,'String','current-state variance');
        set(H.cards.cv.value,'String',sprintf('%.3f',overallCv));
        set(H.cards.cv.subtitle,'String','sigma / mu');

        set(H.cards.qapp.value,'String',sprintf('%.2f ΔF/F₀',qApp));
        set(H.cards.qapp.subtitle,'String',ternary(cvQ > 0.01, ...
            'overestimate from Q scatter', ...
            'unbiased (CVQ = 0)'));

        set(H.cards.napp.value,'String',sprintf('%.1f sites',nApp));
        set(H.cards.napp.subtitle,'String','underestimate (narrower arch)');

        set(H.cards.papp.value,'String',sprintf('%.3f',pApp));
        if pAppRaw > pars.pbar + 1e-3
            pNote = 'overestimate';
        elseif pAppRaw < pars.pbar - 1e-3
            pNote = 'underestimate';
        else
            pNote = 'unbiased';
        end
        set(H.cards.papp.subtitle,'String',pNote);

        iMaxErr = 100 * (iMaxApp - iMaxTrue) / max(iMaxTrue,eps);
        set(H.cards.imax.value,'String',sprintf('%+.1f %%',iMaxErr));
        if iMaxErr < -0.5
            iNote = 'leftward shift';
        elseif iMaxErr > 0.5
            iNote = 'rightward shift';
        else
            iNote = 'no shift';
        end
        set(H.cards.imax.subtitle,'String',iNote);

        if ~isempty(trainData)
            refreshTrainWindow(pars,trainPars,trainData);
        end

        drawnow;
    end
end

function h = makeFlatButton(fig,theme,position,label,callback,hAlign)
    h = uicontrol(fig,'Style','text','Units','normalized', ...
        'Position',position, ...
        'String',label, ...
        'Enable','inactive', ...
        'ButtonDownFcn',callback, ...
        'BackgroundColor',theme.bg, ...
        'ForegroundColor',theme.link, ...
        'FontSize',10, ...
        'FontWeight','bold', ...
        'HorizontalAlignment',hAlign);
end

function ctl = makeSliderControl(fig,theme,x,y,w,label,vmin,vmax,v0,steps,onChange)
    % Uses MATLAB's native slider so pointer capture is handled by the OS.
    % The previous hand-rolled slider relied on a WindowButtonUpFcn to end the
    % drag; if that release event was dropped (a known regression in recent
    % MATLAB releases) the thumb stayed glued to the cursor.
    ctl.label = uicontrol(fig,'Style','text','Units','normalized', ...
        'Position',[x y + 0.038 0.80 * w 0.026], ...
        'String',label, ...
        'BackgroundColor',theme.bg, ...
        'ForegroundColor',theme.text, ...
        'FontSize',9.4, ...
        'FontWeight','bold', ...
        'HorizontalAlignment','left');

    ctl.value = uicontrol(fig,'Style','text','Units','normalized', ...
        'Position',[x + 0.80 * w y + 0.038 0.20 * w 0.026], ...
        'String','', ...
        'BackgroundColor',theme.bg, ...
        'ForegroundColor',theme.text, ...
        'FontName','FixedWidth', ...
        'FontSize',9.6, ...
        'FontWeight','bold', ...
        'HorizontalAlignment','right');

    sliderStep = steps(:)';
    if isscalar(sliderStep)
        sliderStep = [sliderStep 5 * sliderStep];
    end
    sliderStep = min(max(sliderStep,1e-6),1);

    ctl.slider = uicontrol(fig,'Style','slider','Units','normalized', ...
        'Position',[x y w 0.030], ...
        'Min',vmin,'Max',vmax,'Value',min(max(v0,vmin),vmax), ...
        'SliderStep',sliderStep, ...
        'BackgroundColor',theme.sliderFill);
    ctl.track = ctl.slider;   % keep the .track field name for API compatibility

    step = max(steps(1) * (vmax - vmin),eps);
    state = struct( ...
        'valueHandle',ctl.value, ...
        'valueFmt',sliderValueFormat(step,vmin,vmax), ...
        'step',step, ...
        'enabled',true, ...
        'callback',onChange);
    setappdata(ctl.slider,'MiniSliderState',state);

    set(ctl.slider,'Callback',@(s,~) sliderNativeChanged(s));
    try
        addlistener(ctl.slider,'ContinuousValueChange',@(s,~) sliderNativeChanged(s));
    catch
        % ContinuousValueChange is unavailable in some contexts; the standard
        % Callback (fires on release) still keeps everything in sync.
    end
    updateSliderValueText(ctl.slider);
end

function fmt = sliderValueFormat(step,vmin,vmax)
    if step >= 1 - 1e-9 && abs(vmin - round(vmin)) < 1e-9 && abs(vmax - round(vmax)) < 1e-9
        fmt = '%d';
    elseif step >= 0.1 - 1e-9
        fmt = '%.1f';
    else
        fmt = '%.2f';
    end
end

function card = makeCard(position,titleText,accent,tagBase,theme)
    card.panel = uipanel('Units','normalized', ...
        'Position',position, ...
        'BackgroundColor',theme.card, ...
        'BorderType','line');

    parent = get(card.panel,'Parent');
    set(card.panel,'Parent',parent);

    uipanel(card.panel,'Units','normalized', ...
        'Position',[0 0.96 1 0.04], ...
        'BackgroundColor',accent, ...
        'BorderType','none');

    card.title = uicontrol(card.panel,'Style','text','Units','normalized', ...
        'Position',[0.06 0.68 0.88 0.18], ...
        'String',titleText, ...
        'BackgroundColor',theme.card, ...
        'ForegroundColor',theme.muted, ...
        'FontSize',9.3, ...
        'FontWeight','bold', ...
        'HorizontalAlignment','left');

    card.value = uicontrol(card.panel,'Style','text','Units','normalized', ...
        'Position',[0.06 0.31 0.88 0.28], ...
        'String','', ...
        'Tag',[tagBase 'Value'], ...
        'BackgroundColor',theme.card, ...
        'ForegroundColor',theme.text, ...
        'FontSize',15, ...
        'FontWeight','bold', ...
        'HorizontalAlignment','left');

    card.subtitle = uicontrol(card.panel,'Style','text','Units','normalized', ...
        'Position',[0.06 0.09 0.88 0.18], ...
        'String','', ...
        'Tag',[tagBase 'Subtitle'], ...
        'BackgroundColor',theme.card, ...
        'ForegroundColor',theme.muted, ...
        'FontSize',8.8, ...
        'HorizontalAlignment','left');
end

function m = targetMax(target,field)
    m = 0;
    if ~isempty(target) && isfield(target,field) && ~isempty(target.(field))
        v = target.(field);
        v = v(~isnan(v));
        if ~isempty(v), m = max(v); end
    end
end

function h = plotTargetBand(ax,target,field,color)
    % Faint thick line + small markers behind the model trace, showing the
    % measured target for direct model-vs-data comparison.
    h = gobjects(0);
    if ~isfield(target,field) || isempty(target.(field))
        return;
    end
    y = target.(field);
    if isfield(target,'stim') && numel(target.stim) == numel(y)
        x = target.stim;
    else
        x = 1:numel(y);
    end
    valid = ~isnan(y);
    if ~any(valid)
        return;
    end
    % Light tint of the panel colour (solid RGB: MarkerFaceColor does not
    % accept a 4-element RGBA value, unlike line Color).
    pale = color + (1 - color) * 0.62;
    h = plot(ax,x(valid),y(valid),'-','Color',pale,'LineWidth',6);
    plot(ax,x(valid),y(valid),'o','Color',pale, ...
        'MarkerFaceColor',pale,'MarkerSize',4,'LineStyle','none', ...
        'HandleVisibility','off');
end

function styleAxes(ax,xLabel,yLabel,plotTitle,theme)
    set(ax,'Color',theme.card, ...
        'FontName','Helvetica', ...
        'FontSize',9, ...
        'LineWidth',0.9, ...
        'XColor',theme.axis, ...
        'YColor',theme.axis, ...
        'GridColor',theme.grid, ...
        'Layer','top');
    grid(ax,'on');
    box(ax,'off');
    xlabel(ax,xLabel,'Color',theme.text,'FontWeight','bold','FontSize',9.5);
    ylabel(ax,yLabel,'Color',theme.text,'FontWeight','bold','FontSize',8.4);
    title(ax,plotTitle,'Color',theme.text,'FontWeight','bold','FontSize',10.2);
end

function theme = makeTheme()
    theme.bg = [1.00 1.00 1.00];
    theme.card = [1.00 1.00 1.00];
    theme.text = [0.18 0.20 0.21];
    theme.muted = [0.43 0.46 0.47];
    theme.disabled = [0.66 0.68 0.69];
    theme.axis = [0.30 0.33 0.35];
    theme.grid = [0.93 0.94 0.95];
    theme.blue = [0.22 0.54 0.87];
    theme.green = [0.11 0.62 0.46];
    theme.orange = [0.94 0.62 0.15];
    theme.slope = [0.54 0.53 0.50];
    theme.envelope = [0.11 0.62 0.46];
    theme.fail = [0.85 0.35 0.19];
    theme.purple = [0.50 0.47 0.87];
    theme.train = [0.62 0.30 0.67];
    theme.link = [0.33 0.36 0.40];
    theme.refFill = [0.92 0.95 0.98];
    theme.refEdge = [0.66 0.71 0.78];
    theme.sliderTrack = [0.87 0.89 0.92];
    theme.sliderFill = [0.70 0.75 0.82];
    theme.sliderThumb = [0.48 0.52 0.58];
end

function [upper,ticks] = nicePositiveAxis(rawUpper,targetIntervals)
    % Round an arbitrary positive limit to a compact 1/2/2.5/5 decade
    % step. Explicit ticks keep small fluorescence variances legible.
    rawUpper = max(rawUpper,eps);
    targetStep = rawUpper / max(targetIntervals,1);
    decade = 10 ^ floor(log10(targetStep));
    scaled = targetStep / decade;
    if scaled <= 1
        step = decade;
    elseif scaled <= 2
        step = 2 * decade;
    elseif scaled <= 2.5
        step = 2.5 * decade;
    elseif scaled <= 5
        step = 5 * decade;
    else
        step = 10 * decade;
    end
    upper = ceil(rawUpper / step) * step;
    ticks = 0:step:upper;
end

function map = viridisMap(n)
    % Perceptually uniform viridis anchors (dark violet to yellow),
    % interpolated locally so no toolbox colormap dependency is required.
    anchors = [ ...
        0.267004 0.004874 0.329415
        0.282327 0.094955 0.417331
        0.253935 0.265254 0.529983
        0.206756 0.371758 0.553117
        0.163625 0.471133 0.558148
        0.127568 0.566949 0.550556
        0.134692 0.658636 0.517649
        0.266941 0.748751 0.440573
        0.477504 0.821444 0.318195
        0.741388 0.873449 0.149561
        0.993248 0.906157 0.143936];
    anchorX = linspace(0,1,size(anchors,1));
    map = interp1(anchorX,anchors,linspace(0,1,max(2,n)),'pchip');
    map = min(max(map,0),1);
end

function pmf = binomialPmfVector(n,prob)
    % Full binomial PMF for k = 0..n in a single vectorized pass.
    k = 0:n;
    prob = min(max(prob,eps),1 - eps);
    logPmf = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) + ...
        k * log(prob) + (n - k) * log(1 - prob);
    pmf = exp(logPmf);
end

function cache = buildReferenceCache(ref)
    % Reference state never changes during a session, so cache its PMF,
    % variance-mean grid, and operating-point moments once.
    cache.pGrid = linspace(0,1,161);
    cache.pmf = binomialPmfVector(ref.N,ref.pbar);
    cache.amplitudes = (0:ref.N) * ref.Q;
    cache.meanGrid = ref.N * ref.Q * cache.pGrid;
    cache.varGrid = max(0,ref.Q * cache.meanGrid - cache.meanGrid.^2 / ref.N);
    cache.mu0 = ref.N * ref.pbar * ref.Q;
    cache.var0 = ref.N * ref.pbar * (1 - ref.pbar) * ref.Q^2;
end

function value = ternary(flag,trueValue,falseValue)
    if flag
        value = trueValue;
    else
        value = falseValue;
    end
end

function value = ternaryColor(flag,trueValue,falseValue)
    if flag
        value = trueValue;
    else
        value = falseValue;
    end
end

function value = getSliderValue(ctl)
    value = get(ctl.track,'Value');
end

function setSliderValue(ctl,value,invokeCallback)
    state = getappdata(ctl.track,'MiniSliderState');
    value = min(max(value,get(ctl.track,'Min')),get(ctl.track,'Max'));
    set(ctl.track,'Value',value);
    updateSliderValueText(ctl.track);
    if invokeCallback && ~isempty(state.callback)
        state.callback([],[]);
    end
end

function sliderNativeChanged(sliderHandle)
    state = getappdata(sliderHandle,'MiniSliderState');
    if ~state.enabled
        return;
    end
    updateSliderValueText(sliderHandle);
    if ~isempty(state.callback)
        state.callback([],[]);
    end
end

function updateSliderValueText(sliderHandle)
    state = getappdata(sliderHandle,'MiniSliderState');
    if ~isfield(state,'valueHandle') || ~isgraphics(state.valueHandle)
        return;
    end
    value = get(sliderHandle,'Value');
    if strcmp(state.valueFmt,'%d')
        set(state.valueHandle,'String',sprintf('%d',round(value)));
    else
        set(state.valueHandle,'String',sprintf(state.valueFmt,value));
    end
end

function setSliderRange(ctl,vmin,vmax)
    if get(ctl.track,'Min') == vmin && get(ctl.track,'Max') == vmax
        return;
    end
    value = min(max(get(ctl.track,'Value'),vmin),vmax);
    % Order the assignments so Min never transiently exceeds Max.
    set(ctl.track,'Value',value);
    set(ctl.track,'Min',vmin);
    set(ctl.track,'Max',vmax);
    set(ctl.track,'Value',value);
    updateSliderValueText(ctl.track);
end

function setSliderEnabled(ctl,isEnabled)
    state = getappdata(ctl.track,'MiniSliderState');
    state.enabled = isEnabled;
    setappdata(ctl.track,'MiniSliderState',state);
    if isEnabled
        set(ctl.track,'Enable','on');
    else
        set(ctl.track,'Enable','off');
    end
end
