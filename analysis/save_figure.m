function p = save_figure(f, figdir, name)
%SAVE_FIGURE Write one figure to PNG at 300 dpi and close it.

p = string(fullfile(figdir, name));
exportgraphics(f, p, 'Resolution', 300);
close(f);
end
