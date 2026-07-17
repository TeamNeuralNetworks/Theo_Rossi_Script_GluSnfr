function s = mpfa_gui_style()
% MPFA_GUI_STYLE  Shared visual theme matching the mpfa_npq_simulator GUI.
%
%   s = mpfa_gui_style() returns a struct with:
%     s.theme    the GUI color palette (fields bg, card, text, axis, grid,
%                blue, green, orange, slope, fail, purple, train, ... )
%     s.style    function handle style(ax,xLabel,yLabel,titleText) applying the
%                exact GUI axis styling (Helvetica, thin grey axes, light grid,
%                box off, bold dark labels).
%     s.viridis  function handle viridis(n) -> n x 3 colormap used for train
%                trajectories in the GUI.
%     s.line     canonical line spec {'-o','MarkerFaceColor',col,'LineWidth',1.8,
%                'MarkerSize',5.5} via s.line(col).
%
%   These are copied from mpfa_npq_simulator.m so the standalone figures match
%   the interactive tool pixel-for-pixel in look.

    s.theme   = guiTheme();
    s.style   = @(ax,xl,yl,tt) styleAxes(ax,xl,yl,tt,guiTheme());
    s.viridis = @viridisMap;
    s.line    = @(col) {'-o','Color',col,'MarkerFaceColor',col, ...
                        'LineWidth',1.8,'MarkerSize',5.5};
end

function styleAxes(ax,xLabel,yLabel,plotTitle,theme)
    set(ax,'Color',theme.card,'FontName','Helvetica','FontSize',9, ...
        'LineWidth',0.9,'XColor',theme.axis,'YColor',theme.axis, ...
        'GridColor',theme.grid,'Layer','top');
    grid(ax,'on'); box(ax,'off');
    if ~isempty(xLabel), xlabel(ax,xLabel,'Color',theme.text,'FontWeight','bold','FontSize',9.5); end
    if ~isempty(yLabel), ylabel(ax,yLabel,'Color',theme.text,'FontWeight','bold','FontSize',8.4); end
    if ~isempty(plotTitle), title(ax,plotTitle,'Color',theme.text,'FontWeight','bold','FontSize',10.2); end
end

function theme = guiTheme()
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
end

function map = viridisMap(n)
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
