function out = mpfa_demo_invert_varmean(doPlot)
% MPFA_DEMO_INVERT_VARMEAN  Recover apparent N_k and P_k from variance-to-mean data.
%
%   This demo runs the analysis in the "inverse" direction: it reads the measured
%   variance-to-mean trajectory (x = mu_k/mu_1, y = VMR_k/VMR_1) for each
%   condition and, under the fixed-Q constraint (Eq. 6 of the manuscript),
%   inverts it into absolute per-pulse apparent release probability P_k and site
%   count N_k:
%       P_k = 1 - y_k (1 - P0),   rho_P = P_k/P0,   N_k = (x_k / rho_P) * N1,
%   where P0 is the pulse-1 MLE release probability and N1 the pulse-1 apparent
%   site count.
%
%   It writes TWO figures (apparent P and apparent N kept separate for
%   legibility), each with one panel per condition overlaying three series:
%     - reality      : the independently measured Pr / N profiles (markers)
%     - train model  : the fitted depletion/refill model (solid line)
%     - var/mean inv.: the fixed-Q inversion of the variance-to-mean data
%                      (dashed line + shaded CI)
%   Each panel is annotated with an R^2 match score of model-vs-reality and
%   inversion-vs-reality; an overall score is printed to the console.
%
%   out = mpfa_demo_invert_varmean(false) returns results without plotting.
%
%   See also MPFA_CROSSMETHOD, MPFA_BOOTSTRAP_CONCORDANCE, MPFA_FIT_DEMO,
%   MPFA_MAKE_FIGURES, MPFA_GUI_STYLE.

    if nargin < 1 || isempty(doPlot), doPlot = true; end
    here = fileparts(mfilename('fullpath'));
    st = mpfa_gui_style(); th = st.theme;

    T = readtable(fullfile(here,'FigS16_var_imean_profiles_6conditions.csv'), ...
        'VariableNamingRule','preserve');
    M = readmatrix(fullfile(here,'FigS16_mpfa_demo_profiles_matrix.csv'));

    labels = {'1.5mM_20Hz','1.5mM_50Hz','2.5mM_20Hz','2.5mM_50Hz','4mM_20Hz','4mM_50Hz'};
    measPr = M(:,2:7); measN = M(:,8:13); measPPR = M(:,14:19);

    results = mpfa_fit_demo('2.5mM_20Hz',false);   % mechanistic model per condition
    freq = T.Frequency_Hz; ca = string(T.Calcium);

    D = struct('label',{},'k',{},'P',{},'N',{},'PPR',{});
    out = struct('label',{},'P',{},'N',{},'PPR',{});

    for c = 1:6
        [caStr,frv] = parseLabel(labels{c});
        sel = freq==frv & ca==sprintf('%g mM',caStr);
        x  = T.Mean_percent_A1(sel)/100;
        y  = T.Var_over_Imean_percent_A1(sel)/100;
        yLo= T.Var_over_Imean_percent_CI_low(sel)/100;
        yHi= T.Var_over_Imean_percent_CI_high(sel)/100;
        P0 = T.Median_MLE_P_A1_excluding_cap20(find(sel,1));
        N1 = measN(1,c);

        Pk = 1 - y.*(1-P0); rhoP = Pk./P0; Nk = (x./rhoP).*N1;
        ySD = (yHi - yLo)/(2*1.959963985);
        Pk_sd = abs(-(1-P0)).*ySD;
        Nk_sd = abs((x.*N1)./(P0.*rhoP.^2)).*Pk_sd;

        Pmeas = measPr(:,c); Nmeas = measN(:,c);
        Pmodel = results(c).fit.model.pr(:); Nmodel = results(c).fit.model.readyBefore(:);

        % 3-point running median over stimulus number. This suppresses the
        % single-pulse spikes, but note they are a re-partitioning between N and
        % P (see PPR below), not an error in the total, so smoothing is
        % cosmetic: it trades an honest CI blow-up for a prettier trace.
        Pk_med = runmed3(Pk); Nk_med = runmed3(Nk);

        % Three PPR estimates with genuinely different provenance:
        %   meas  - the directly measured amplitude ratio AMP_k/AMP_1.
        %   inv   - recovered from the variance-to-mean route only (amplitude
        %           mean + variance; no quantal conversion, no failure
        %           detection, no MLE). Equals rho_N*rho_P of the inversion.
        %   model - the train model's ratio, whose N/P came from the MLE chain
        %           (quantal conversion + failure detection + binomial MLE).
        % Comparing all three tests the two independent chains against reality.
        pprInv   = (Nk(:)./N1) .* (Pk(:)./P0);
        pprModel = results(c).fit.model.releasePpr(:);
        pprMeas  = measPPR(:,c);

        P = struct('meas',Pmeas,'model',Pmodel,'inv',Pk(:),'inv_sd',Pk_sd(:), ...
            'med',Pk_med(:),'r2model',r2_(Pmeas,Pmodel),'r2inv',r2_(Pmeas,Pk(:)), ...
            'r2med',r2_(Pmeas,Pk_med(:)));
        N = struct('meas',Nmeas,'model',Nmodel,'inv',Nk(:),'inv_sd',Nk_sd(:), ...
            'med',Nk_med(:),'r2model',r2_(Nmeas,Nmodel),'r2inv',r2_(Nmeas,Nk(:)), ...
            'r2med',r2_(Nmeas,Nk_med(:)));
        PPR = struct('meas',pprMeas,'inv',pprInv,'model',pprModel, ...
            'r2inv',r2_(pprMeas,pprInv),'r2model',r2_(pprMeas,pprModel), ...
            'r2modelVsInv',r2_(pprInv,pprModel), ...
            'ratioInv',median(pprInv(2:end)./pprMeas(2:end),'omitnan'), ...
            'ratioModel',median(pprModel(2:end)./pprMeas(2:end),'omitnan'));

        D(c) = struct('label',labels{c},'k',(1:numel(x))','P',P,'N',N,'PPR',PPR);
        out(c) = struct('label',labels{c},'P',P,'N',N,'PPR',PPR);
    end

    printScores(D);

    if doPlot
        outdir = fullfile(here,'figures'); if ~exist(outdir,'dir'), mkdir(outdir); end
        renderQuantity(st,D,'P','apparent P',[0 1], ...
            fullfile(outdir,'Fig_invert_varmean_P.png'),th.fail);
        renderQuantity(st,D,'N','apparent N (sites)',[], ...
            fullfile(outdir,'Fig_invert_varmean_N.png'),th.blue);
        renderPpr(st,D,fullfile(outdir,'Fig_invert_varmean_PPR.png'));
    end
end

% -------------------------------------------------------------------------
function renderPpr(st,D,fname)
% Three-way PPR comparison: the directly measured amplitude ratio versus the two
% independent reconstruction chains (variance-to-mean inversion, and the
% MLE-based train model).
    th = st.theme;
    f = figure('Color',th.bg,'Position',[70 70 1150 660],'Visible','off');
    tl = tiledlayout(f,2,3,'Padding','compact','TileSpacing','compact');
    title(tl,'PPR: measured amplitude ratio vs variance-to-mean inversion vs MLE-based model', ...
        'FontWeight','bold','Color',th.text,'FontName','Helvetica');
    for c = 1:numel(D)
        q = D(c).PPR; k = D(c).k;
        nexttile; hold on
        plot(k,q.inv,'-','Color',th.green,'LineWidth',1.8);
        plot(k,q.model,'--','Color',th.purple,'LineWidth',1.6);
        plot(k,q.meas,'o','Color',th.slope,'MarkerFaceColor',th.slope, ...
            'MarkerSize',5,'LineStyle','none');
        st.style(gca,'Stimulus','PPR (A_k/A_1)', ...
            sprintf('%s   inv/meas=%.2fx  model/meas=%.2fx',D(c).label, ...
                q.ratioInv,q.ratioModel));
        if c == 1
            legend({'inversion (var/mean only)','model (MLE N,P)','measured PPR'}, ...
                'Location','northwest','Box','off','FontSize',7.0);
        end
    end
    try
        exportgraphics(f,fname,'Resolution',150);
    catch
        print(f,fname,'-dpng','-r150');
    end
    close(f); fprintf('  wrote %s\n',fname);
end

function y = runmed3(x)
% 3-point running median (base MATLAB, endpoints preserved).
    x = x(:); y = x;
    for i = 2:numel(x)-1
        y(i) = median(x(i-1:i+1));
    end
end

% -------------------------------------------------------------------------
function renderQuantity(st,D,fld,ylab,ylims,fname,col)
    th = st.theme;
    f = figure('Color',th.bg,'Position',[70 70 1150 660],'Visible','off');
    tl = tiledlayout(f,2,3,'Padding','compact','TileSpacing','compact');
    title(tl,sprintf('Inverse fit — %s: measured vs model vs variance-to-mean inversion',ylab), ...
        'FontWeight','bold','Color',th.text,'FontName','Helvetica');
    for c = 1:numel(D)
        q = D(c).(fld); k = D(c).k;
        nexttile; hold on
        fillCI(k,q.inv,q.inv_sd,col);
        plot(k,q.model,'-','Color',col,'LineWidth',1.8);
        plot(k,q.inv,'--','Color',col,'LineWidth',1.4);
        plot(k,q.med,':','Color',th.slope,'LineWidth',1.6);
        plot(k,q.meas,'o','MarkerFaceColor',col,'MarkerEdgeColor',col, ...
            'MarkerSize',5,'LineStyle','none');
        st.style(gca,'Stimulus',ylab, ...
            sprintf('%s   R^2: model %.2f, inv %.2f, med %.2f', ...
                D(c).label,q.r2model,q.r2inv,q.r2med));
        set(gca,'YColor',col);
        if ~isempty(ylims), ylim(ylims); end
        if c == 1
            p1 = plot(nan,nan,'o','Color',th.slope,'MarkerFaceColor',th.slope,'LineStyle','none');
            p2 = plot(nan,nan,'-','Color',th.slope,'LineWidth',1.8);
            p3 = plot(nan,nan,'--','Color',th.slope,'LineWidth',1.4);
            p4 = plot(nan,nan,':','Color',th.slope,'LineWidth',1.6);
            legend([p1 p2 p3 p4],{'measured (reality)','train model', ...
                'var/mean inversion','inversion, 3-pt median'}, ...
                'Location','best','Box','off','FontSize',7.0);
        end
    end
    try
        exportgraphics(f,fname,'Resolution',150);
    catch
        print(f,fname,'-dpng','-r150');
    end
    close(f); fprintf('  wrote %s\n',fname);
end

function printScores(D)
    n = numel(D);
    getf = @(fld,sub) arrayfun(@(d) d.(fld).(sub), D);
    fprintf('\n%-11s | P: R^2 mod/inv/med | N: R^2 mod/inv/med | PPR vs measured: inv / model / (model vs inv)\n','condition');
    fprintf('%s\n',repmat('-',1,104));
    for c = 1:n
        q = D(c).PPR;
        fprintf('%-11s | %5.2f/%5.2f/%5.2f  | %5.2f/%5.2f/%5.2f  |  %5.2fx / %5.2fx   R^2 %5.2f /%5.2f / %5.2f\n', ...
            D(c).label,D(c).P.r2model,D(c).P.r2inv,D(c).P.r2med, ...
            D(c).N.r2model,D(c).N.r2inv,D(c).N.r2med, ...
            q.ratioInv,q.ratioModel,q.r2inv,q.r2model,q.r2modelVsInv);
    end
    fprintf('%s\n',repmat('-',1,104));
    fprintf(['"model" = train fit whose N/P came from the MLE chain (quantal\n', ...
             'conversion + failure detection + binomial MLE). "inv" = recovered\n', ...
             'from variance and mean amplitude only (no Q, no failures, no MLE).\n', ...
             '"med" = inversion after a 3-point running median.\n', ...
             'PPR block: inv/meas and model/meas are median fold-differences vs the\n', ...
             'directly measured amplitude ratio; the last R^2 compares the two\n', ...
             'independent chains against EACH OTHER.\n']);
end

function fillCI(k,y,sd,col)
    k=k(:)'; y=y(:)'; sd=sd(:)';
    good = isfinite(y)&isfinite(sd); k=k(good); y=y(good); sd=sd(good);
    if numel(k)<2, return; end
    fill([k fliplr(k)],[y-sd fliplr(y+sd)],col,'FaceAlpha',0.10, ...
        'EdgeColor','none','HandleVisibility','off');
end

function v = r2_(meas,pred)
    m = ~isnan(meas)&~isnan(pred); meas=meas(m); pred=pred(m);
    if numel(meas)<2, v=NaN; return; end
    ss = sum((meas-mean(meas)).^2);
    if ss<eps, v=NaN; return; end
    v = 1 - sum((meas-pred).^2)/ss;
end

function [ca,fr] = parseLabel(lab)
    tok = regexp(lab,'([\d.]+)mM_(\d+)Hz','tokens','once');
    ca = str2double(tok{1}); fr = str2double(tok{2});
end
