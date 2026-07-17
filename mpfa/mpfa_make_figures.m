function mpfa_make_figures(outdir)
% MPFA_MAKE_FIGURES  Generate the manuscript figures from the real-data CSVs,
%                    styled to match the mpfa_npq_simulator GUI.
%
%   mpfa_make_figures() fits the six-condition demonstration dataset and writes
%   Figures 1-5 and Fig. S16 (as defined in the Word manuscript) to a figures/
%   subfolder as PNGs. All quantitative panels are driven by the shipped CSVs:
%       FigS16_mpfa_demo_profiles_matrix.csv        (apparent Pr / N / PPR)
%       FigS16_var_imean_profiles_6conditions.csv   (variance-to-mean, with CIs)
%
%   Colours, fonts, grid and line specs come from MPFA_GUI_STYLE so the figures
%   match the interactive tool.
%
%   See also MPFA_FIT_DEMO, MPFA_TRAIN_CORE, MPFA_CROSSMETHOD,
%   MPFA_BOOTSTRAP_CONCORDANCE, MPFA_FIT_HETEROGENEITY, MPFA_GUI_STYLE.

    here = fileparts(mfilename('fullpath'));
    if nargin < 1 || isempty(outdir), outdir = fullfile(here,'figures'); end
    if ~exist(outdir,'dir'), mkdir(outdir); end
    ref = struct('N0',8,'P0',0.25,'Q0',0.5);
    st = mpfa_gui_style();

    fprintf('Fitting six-condition demonstration dataset...\n');
    results = mpfa_fit_demo('2.5mM_20Hz',false);
    T = readtable(fullfile(here,'FigS16_var_imean_profiles_6conditions.csv'), ...
        'VariableNamingRule','preserve');

    fig1_schematic(st,ref,fullfile(outdir,'Fig1_organization.png'));
    fig2_roundtrip(st,ref,fullfile(outdir,'Fig2_roundtrip.png'));
    fig4_latent_apparent(st,fullfile(outdir,'Fig4_latent_apparent.png'));
    fig5_hierarchical(st,results,fullfile(outdir,'Fig5_hierarchical.png'));
    figS16_variance_to_mean(st,T,fullfile(outdir,'FigS16_variance_to_mean.png'));

    fprintf('Figures written to %s\n',outdir);
end

% =========================================================================
function fig1_schematic(st,ref,path)
    th = st.theme;
    f = newfig(st,[960 300]);
    tl = tiledlayout(f,1,4,'Padding','compact','TileSpacing','compact');
    superTitle(tl,th,'Fig. 1  Analytical organization of the framework');

    nexttile; N=8; P=0.35; j=0:N;
    b = exp(gammaln(N+1)-gammaln(j+1)-gammaln(N-j+1)+j*log(P)+(N-j)*log(1-P));
    bar(j,b,'FaceColor',th.blue,'EdgeColor','none');
    st.style(gca,'quanta released','P(k)','A  Binomial release');

    nexttile; mu=linspace(0,N*0.5,100); Q=0.5; s=Q*mu-mu.^2/N;
    plot(mu,s,'-','Color',th.blue,'LineWidth',2.2);
    st.style(gca,'mean','variance','B  MPFA parabola');

    nexttile; L = lociFor(ref.P0); hold on
    plot(L.x,L.yN,'-','Color',th.green,'LineWidth',1.6);
    plot(L.x,L.yQ,'-','Color',th.blue,'LineWidth',1.6);
    plot(L.x,L.yP,'-','Color',th.purple,'LineWidth',1.6);
    scatter(1,1,42,th.slope,'filled'); xlim([0 2.5]); ylim([0 2.5]);
    st.style(gca,'x = \mu_1/\mu_0','y = VMR ratio','C  Relative-change loci');
    legend({'N','Q','P'},'Location','northwest','Box','off','FontSize',7.5);

    nexttile; [~,m] = demoTrain('facil'); hold on
    plot(m.stim,m.latentN,':','Color',th.orange,'LineWidth',1.6);
    plot(m.stim,m.readyBefore,'-o','Color',th.blue,'MarkerFaceColor',th.blue, ...
        'LineWidth',1.8,'MarkerSize',5.5);
    st.style(gca,'stimulus','sites','D  Latent \rightarrow occupancy');
    legend({'latent N_k^*','ready R_k'},'Location','southeast','Box','off','FontSize',7.5);

    saveFig(f,path);
end

% =========================================================================
function fig2_roundtrip(st,ref,path)
    th = st.theme;
    f = newfig(st,[1000 720]);
    tl = tiledlayout(f,2,2,'Padding','compact','TileSpacing','compact');
    superTitle(tl,th,'Fig. 2  Variance-mean and relative-change geometries are mutually predictive');

    Pgrid = linspace(0.05,0.9,8);
    base = mpfa_crossmethod('mpfa',struct('N',ref.N0,'Q',ref.Q0,'P',Pgrid),ref);
    nchg = mpfa_crossmethod('mpfa',struct('N',12,   'Q',ref.Q0,'P',Pgrid),ref);
    qchg = mpfa_crossmethod('mpfa',struct('N',ref.N0,'Q',0.75, 'P',Pgrid),ref);

    % A: parabolas + operating points (GUI: A1 parabola blue, orange points)
    nexttile; hold on
    plot(nchg.mu,nchg.sigma2,'-','Color',th.green,'LineWidth',1.6);
    plot(qchg.mu,qchg.sigma2,'-','Color',th.purple,'LineWidth',1.6);
    plot(base.mu,base.sigma2,'-','Color',th.blue,'LineWidth',2.2);
    scatter(base.mu,base.sigma2,60,th.orange,'filled');
    st.style(gca,'mean \mu','variance \sigma^2','A  Variance-mean parabolas');
    legend({'+N','+Q','reference','points (vary P)'},'Location','southoutside', ...
        'Box','off','NumColumns',2,'FontSize',7.5);

    % B: normalized plane (GUI loci: N green, Q blue, P purple; ref grey)
    nexttile; hold on
    L = base.loci;
    plot(L.x,L.yN,'-','Color',th.green,'LineWidth',1.6);
    plot(L.x,L.yQ,'-','Color',th.blue,'LineWidth',1.6);
    plot(L.x,L.yP,'-','Color',th.purple,'LineWidth',1.6);
    tN = mpfa_crossmethod('npq',struct('N',12,'P',ref.P0,'Q',ref.Q0),ref);
    tQ = mpfa_crossmethod('npq',struct('N',ref.N0,'P',ref.P0,'Q',0.75),ref);
    tP = mpfa_crossmethod('npq',struct('N',ref.N0,'P',0.5,'Q',ref.Q0),ref);
    scatter(tN.x,tN.y,64,th.green,'s','filled');
    scatter(tQ.x,tQ.y,64,th.blue,'^','filled');
    scatter(tP.x,tP.y,64,th.purple,'o','filled');
    scatter(1,1,42,th.slope,'filled');
    xlim([0 2.2]); ylim([0 2.2]);
    st.style(gca,'x = \mu_1/\mu_0','y = VMR_1/VMR_0','B  Normalized relative-change plane');
    legend({'N locus','Q locus','P locus','pure N','pure Q','pure P'}, ...
        'Location','southoutside','Box','off','NumColumns',3,'FontSize',7.2);

    % C: MLE round-trip on synthetic amplitudes
    nexttile; hold on
    trueState = struct('N',8,'P',0.4,'Q',0.5);
    a = simulateAmplitudes(trueState,4000,0.06,0.25);
    est = mpfa_fit_heterogeneity(a,'Nrange',5:11,'Restarts',5);
    truep = mpfa_crossmethod('mpfa',struct('N',trueState.N,'Q',trueState.Q,'P',Pgrid),ref);
    plot(truep.mu,truep.sigma2,'-','Color',th.blue,'LineWidth',2.2);
    muE = est.N*est.P*est.Q; sE = est.N*est.P*(1-est.P)*est.Q^2;
    scatter(muE,sE,110,th.orange,'p','filled');
    st.style(gca,'mean \mu','variance \sigma^2', ...
        sprintf('C  MLE recovers N=%.1f P=%.2f Q=%.2f',est.N,est.P,est.Q));
    legend({'true parabola','MLE estimate'},'Location','southoutside','Box','off','FontSize',7.5);

    % D: residual between MLE-projected and true normalized location
    nexttile; hold on
    plot(L.x,L.yP,'-','Color',th.purple,'LineWidth',1.6);
    tTrue = mpfa_crossmethod('npq',struct('N',trueState.N,'P',0.5,'Q',trueState.Q),ref);
    tEst  = mpfa_crossmethod('npq',struct('N',est.N,'P',0.5,'Q',est.Q),ref);
    scatter(1,1,42,th.slope,'filled');
    scatter(tTrue.x,tTrue.y,64,th.slope,'filled');
    scatter(tEst.x,tEst.y,110,th.orange,'p','filled');
    quiver(tTrue.x,tTrue.y,tEst.x-tTrue.x,tEst.y-tTrue.y,0,'Color',th.fail,'LineWidth',1.2);
    xlim([0 3]); ylim([0 2.2]);
    st.style(gca,'x','y','D  Residual = disagreement between views');
    legend({'P locus','ref','true state','MLE proj.'},'Location','southoutside', ...
        'Box','off','NumColumns',2,'FontSize',7.2);

    saveFig(f,path);
end

% =========================================================================
function fig4_latent_apparent(st,path)
    th = st.theme;
    f = newfig(st,[1000 720]);
    tl = tiledlayout(f,2,3,'Padding','compact','TileSpacing','compact');
    superTitle(tl,th,'Fig. 4  Latent control variables versus apparent event-wise estimates');

    [~,m] = demoTrain('facil');

    nexttile;
    yyaxis left;  plot(m.stim,m.pr,'-o','Color',th.fail,'MarkerFaceColor',th.fail,'LineWidth',1.8,'MarkerSize',5);
    yyaxis right; plot(m.stim,m.latentN,'-s','Color',th.orange,'MarkerFaceColor',th.orange,'LineWidth',1.8,'MarkerSize',5);
    st.style(gca,'stimulus','','A  Latent drive');
    dualY(gca,th,'latent P_k',th.fail,'latent N_k^*',th.orange);

    nexttile;
    plot(m.stim,m.readyBefore,'-o','Color',th.blue,'MarkerFaceColor',th.blue,'LineWidth',1.8,'MarkerSize',5.5);
    st.style(gca,'stimulus','ready pool R_k','B  Occupancy');
    set(gca,'YColor',th.blue);

    nexttile;
    yyaxis left;  plot(m.stim,m.released,'-o','Color',th.green,'MarkerFaceColor',th.green,'LineWidth',1.8,'MarkerSize',5);
    yyaxis right; plot(m.stim,m.normCumRelease,'-s','Color',th.slope,'MarkerFaceColor',th.slope,'LineWidth',1.8,'MarkerSize',5);
    st.style(gca,'stimulus','','C  Output');
    dualY(gca,th,'released quanta',th.green,'cumulative (norm.)',th.slope);

    nexttile;
    yyaxis left;  plot(m.stim,m.readyBefore,'-o','Color',th.blue,'MarkerFaceColor',th.blue,'LineWidth',1.8,'MarkerSize',5);
    yyaxis right; plot(m.stim,m.pr,'-s','Color',th.fail,'MarkerFaceColor',th.fail,'LineWidth',1.8,'MarkerSize',5);
    st.style(gca,'stimulus','','D  Apparent event-wise estimates');
    dualY(gca,th,'apparent N_k',th.blue,'apparent P_k',th.fail);

    nexttile([1 2]); hold on
    [~,mm] = demoTrain('mask');
    plot(mm.stim,mm.pr,'-s','Color',th.fail,'MarkerFaceColor',th.fail,'LineWidth',1.6,'MarkerSize',5);
    plot(mm.stim,mm.meanResp/max(mm.meanResp),'-o','Color',th.slope,'MarkerFaceColor',th.slope,'LineWidth',1.6,'MarkerSize',5);
    plot(mm.stim,mm.readyBefore/max(mm.readyBefore),'-^','Color',th.blue,'MarkerFaceColor',th.blue,'LineWidth',1.4,'MarkerSize',5);
    st.style(gca,'stimulus','normalized','E  Latent facilitation masked by depletion');
    legend({'latent P_k (rising)','mean response (flat)','ready pool R_k (falling)'}, ...
        'Location','east','Box','off','FontSize',7.5);

    saveFig(f,path);
end

% =========================================================================
function fig5_hierarchical(st,results,path)
    th = st.theme;
    f = newfig(st,[1150 760]);
    tl = tiledlayout(f,2,3,'Padding','compact','TileSpacing','compact');
    superTitle(tl,th,'Fig. 5  Hierarchical fits to the six-condition train dataset');

    caList = [1.5 2.5 4]; caCol = [th.blue; th.green; th.orange];
    n = numel(results);

    titles = {'A  Apparent release probability','B  Apparent site count N', ...
              'C  PPR (held out)'};
    ylabs  = {'P_k','sites','A_k/A_1'};
    fieldsMeas = {'pr','N','ppr'}; fieldsMod = {'pr','readyBefore','releasePpr'};
    for pIdx = 1:3
        nexttile; hold on
        for c = 1:n
            [ca,fr] = parseLabel(results(c).label);
            col = caCol(caList==ca,:); ls='-'; if fr==50, ls='--'; end
            tgt = results(c).fit.preset.target; mdl = results(c).fit.model;
            plot(tgt.stim,tgt.(fieldsMeas{pIdx}),'o','Color',col, ...
                'MarkerFaceColor',col,'MarkerSize',4,'LineStyle','none');
            plot(tgt.stim,mdl.(fieldsMod{pIdx}),ls,'Color',col,'LineWidth',1.6);
        end
        st.style(gca,'stimulus',ylabs{pIdx},titles{pIdx});
    end

    nexttile; hold on
    for fr = [20 50]
        Pmax=[]; Nmax=[]; cax=[];
        for c = 1:n
            [ca,f2] = parseLabel(results(c).label); if f2~=fr, continue; end
            p = results(c).fit.pars; t = results(c).fit.trainPars;
            Pmax(end+1)=min(p.pbar*t.pMaxFactor,0.98); Nmax(end+1)=p.N*t.nMaxFactor; cax(end+1)=ca; %#ok<AGROW>
        end
        [cax,ord]=sort(cax); Pmax=Pmax(ord); Nmax=Nmax(ord);
        ls='-'; if fr==50, ls='--'; end
        yyaxis left;  plot(cax,Pmax,['o' ls],'Color',th.fail,'MarkerFaceColor',th.fail,'LineWidth',1.6);
        yyaxis right; plot(cax,Nmax,['s' ls],'Color',th.blue,'MarkerFaceColor',th.blue,'LineWidth',1.6);
    end
    st.style(gca,'extracellular Ca^{2+} (mM)','','D  Latent asymptotes'); xticks(caList);
    dualY(gca,th,'latent P_\infty',th.fail,'latent N_\infty',th.blue);

    nexttile; hold on
    dA = zeros(1,n); lab = cell(1,n);
    for c = 1:n
        rm = results(c).fit.refillModel; dA(c) = rm.aicConstant - rm.aicDynamic; lab{c} = shortLabel(results(c).label);
    end
    bcol = repmat(th.slope,n,1);
    for c=1:n, if strcmp(results(c).fit.refillModel.selected,'dynamic'), bcol(c,:)=th.orange; end; end
    b = bar(1:n,dA,'FaceColor','flat','EdgeColor','none'); b.CData = bcol;
    yline(2,':','Color',th.slope); yline(-2,':','Color',th.slope);
    set(gca,'XTick',1:n,'XTickLabel',lab); xtickangle(35);
    st.style(gca,'','\DeltaAIC (const - dyn)','E  Replenishment model selection');

    nexttile; hold on
    obsAll=[]; prdAll=[];
    for c = 1:n
        tgt = results(c).fit.preset.target; mdl = results(c).fit.model;
        obsAll=[obsAll tgt.ppr]; prdAll=[prdAll mdl.releasePpr]; %#ok<AGROW>
    end
    good=~isnan(obsAll);
    lim=[min([obsAll prdAll],[],'omitnan')*0.95 max([obsAll prdAll],[],'omitnan')*1.05];
    plot(lim,lim,':','Color',th.slope,'LineWidth',1.0);
    scatter(obsAll(good),prdAll(good),26,th.blue,'filled');
    xlim(lim); ylim(lim); axis square
    st.style(gca,'measured PPR','predicted PPR (held out)','F  Held-out PPR prediction');
    legend({'identity','model'},'Location','northwest','Box','off','FontSize',7.2);

    saveFig(f,path);
end

% =========================================================================
function figS16_variance_to_mean(st,T,path)
    th = st.theme;
    f = newfig(st,[1150 760]);
    tl = tiledlayout(f,2,3,'Padding','compact','TileSpacing','compact');
    superTitle(tl,th,'Fig. S16  Variance-to-mean trajectories vs N/P/Q loci (real data, viridis = stimulus)');

    freq = T.Frequency_Hz; ca = string(T.Calcium);
    caVals = ["1.5 mM","2.5 mM","4 mM"]; frVals = [20 50];
    idx = 0;
    for fr = frVals
        for cc = caVals
            idx = idx+1; nexttile; hold on
            sel = freq==fr & ca==cc;
            x  = T.Mean_percent_A1(sel)/100;
            y  = T.Var_over_Imean_percent_A1(sel)/100;
            yLo= T.Var_over_Imean_percent_CI_low(sel)/100;
            yHi= T.Var_over_Imean_percent_CI_high(sel)/100;
            xLo= T.Mean_percent_CI_low(sel)/100;
            xHi= T.Mean_percent_CI_high(sel)/100;
            P0 = T.Median_MLE_P_A1_excluding_cap20(find(sel,1));
            stimN = numel(x); k = (1:stimN)';

            L = lociFor(P0);
            plot(L.x,L.yN,'-','Color',th.green,'LineWidth',1.6);
            plot(L.x,L.yQ,'-','Color',th.blue,'LineWidth',1.6);
            plot(L.x,L.yP,'-','Color',th.purple,'LineWidth',1.6);
            errorbar(x,y,y-yLo,yHi-y,x-xLo,xHi-x,'LineStyle','none', ...
                'Color',[0.75 0.77 0.79],'CapSize',0,'LineWidth',0.6);
            plot(x,y,'-','Color',th.slope,'LineWidth',1.0);
            scatter(x,y,38,k,'filled','MarkerEdgeColor',th.card,'LineWidth',0.6);
            colormap(gca,st.viridis(256)); clim(gca,[1 max(2,stimN)]);
            scatter(1,1,42,th.slope,'filled');

            obs = struct('x',x','y',y','yLo',yLo','yHi',yHi','P0mle',P0);
            res = mpfa_bootstrap_concordance(obs,'B',1000);
            xmax = max(2,max(x)*1.15); xlim([0 xmax]); ylim([0 1.45]);
            st.style(gca,'x = \mu_k/\mu_1','y = VMR_k/VMR_1',sprintf('%s, %d Hz',cc,fr));
            agreeStr = ternary(res.agree,'within CI','outside CI');
            text(0.04*xmax,1.36,sprintf('P0_{MLE}=%.2f  P0_{geom}=%.2f',P0,res.P0geom), ...
                'FontSize',8,'FontName','Helvetica','Color',th.text);
            text(0.04*xmax,1.24,sprintf('z=%.1f (%s)',res.zConcordance,agreeStr), ...
                'FontSize',8,'FontName','Helvetica','Color',th.text);
            if idx==1
                legend({'N','Q','P'},'Location','east','Box','off','FontSize',7.2);
            end
            if idx==3 || idx==6
                cb = colorbar(gca,'eastoutside');
                cb.Label.String='Stimulus'; cb.Label.Color=th.text; cb.Color=th.axis;
                cb.FontSize=7.5; cb.Box='off';
            end
            fprintf('  S16 %-9s %2d Hz: P0_MLE=%.3f  P0_geom=%.3f  z=%.2f  %s\n', ...
                cc,fr,P0,res.P0geom,res.zConcordance,agreeStr);
        end
    end
    saveFig(f,path);
end

% =========================================================================
% Helpers
% =========================================================================
function [pars,m] = demoTrain(kind)
    p = struct('N',8,'pbar',0.28,'Q',0.5,'cvQ',0,'cvP',0);
    switch kind
        case 'facil'
            tp = struct('nStim',10,'freqHz',20,'refillPerStim',1.4,'nMaxFactor',1.0, ...
                'tauNStim',2,'pMaxFactor',2.0,'tauPStim',1.6, ...
                'refillSteadyPerStim',1.4,'tauRefillStim',1.5);
        case 'mask'
            tp = struct('nStim',10,'freqHz',20,'refillPerStim',0.3,'nMaxFactor',1.0, ...
                'tauNStim',2,'pMaxFactor',2.4,'tauPStim',1.4, ...
                'refillSteadyPerStim',0.3,'tauRefillStim',1.5);
    end
    m = mpfa_train_core(p,tp);
    pars = struct('p',p,'tp',tp);
end

function L = lociFor(P0)
    xg = linspace(0,3,161);
    L = struct('x',xg,'yN',ones(size(xg)),'yQ',xg,'yP',(1 - P0.*xg)./(1 - P0));
end

function a = simulateAmplitudes(state,nTrials,cvQ,noiseSD)
    N=state.N; P=state.P; Q=state.Q;
    k = sum(rand(nTrials,N) < P, 2);
    a = zeros(nTrials,1);
    for i=1:nTrials
        if k(i)>0, a(i) = sum(Q + cvQ*Q*randn(k(i),1)); end
    end
    a = a + noiseSD*randn(nTrials,1);
end

function [ca,fr] = parseLabel(lab)
    tok = regexp(lab,'([\d.]+)mM_(\d+)Hz','tokens','once');
    ca = str2double(tok{1}); fr = str2double(tok{2});
end

function s = shortLabel(lab)
    [ca,fr] = parseLabel(lab); s = sprintf('%gmM/%dHz',ca,fr);
end

function dualY(ax,th,leftLbl,leftCol,rightLbl,rightCol)
    yyaxis(ax,'left');  ylabel(ax,leftLbl,'Color',th.text,'FontWeight','bold','FontSize',8.4);
    set(ax,'YColor',leftCol);
    yyaxis(ax,'right'); ylabel(ax,rightLbl,'Color',th.text,'FontWeight','bold','FontSize',8.4);
    set(ax,'YColor',rightCol);
end

function s = ternary(cond,a,b)
    if isequal(cond,true), s = a; else, s = b; end
end

function f = newfig(st,sz)
    f = figure('Color',st.theme.bg,'Position',[70 70 sz],'Visible','off');
end

function superTitle(tl,th,txt)
    title(tl,txt,'FontWeight','bold','Color',th.text,'FontName','Helvetica');
end

function saveFig(f,path)
    try
        exportgraphics(f,path,'Resolution',150);
    catch
        print(f,path,'-dpng','-r150');
    end
    close(f);
    fprintf('  wrote %s\n',path);
end
