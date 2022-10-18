from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cmocean

def imshow(q, cbar=True, location='right', cbar_label=None, ax=None, cmap=None, 
    vmax = None, aspect=True, pct=99, axes=False, interpolation='none', **kwargs):

    if q.min() < 0:
        cmap=cmocean.cm.balance if cmap is None else cmap
        vmax = np.percentile(np.abs(q), pct) if vmax is None else vmax
        vmin = -vmax
    else:
        cmap = 'inferno' if cmap is None else cmap
        vmax = np.percentile(q, pct) if vmax is None else vmax
        vmin = 0
    
    kw = dict(vmin=vmin, vmax=vmax, cmap=cmap, interpolation=interpolation)
    
    if ax is None:
        ax = plt.gca()

    # flipud because imshow inverts vertical axis
    im = ax.imshow(np.flipud(q), **kw, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    if aspect:
        ax.set_box_aspect(1)
    if axes:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    if cbar:
        divider = make_axes_locatable(ax)
        if location == 'right':
            cax = divider.append_axes('right', size="5%", pad=0.1)
            cbar_kw = dict()
        elif location == 'bottom':
            cax = divider.append_axes('bottom', size="5%", pad=0.1)
            cbar_kw = dict(orientation='horizontal')
        cb = plt.colorbar(im, cax = cax, label=cbar_label, **cbar_kw)
        cb.ax.ticklabel_format(scilimits=(-1,1), useMathText=True) # scientific notation for numbers outside range 0.01..100
    
    # Return axis to initial image
    plt.sca(ax)
    return im

def show_fonts():
    import matplotlib.font_manager
    flist = matplotlib.font_manager.get_fontconfig_fonts()
    return [matplotlib.font_manager.FontProperties(fname=fname).get_name() for fname in flist]

def show_rcparams(name):
    for key, val in matplotlib.rcParams.items():
        if name in key:
            print(key, val)

def default_rcParams(kw={}):
    rcParams = matplotlib.rcParamsDefault.copy()
    rcParams.pop('backend') # can break inlining
    matplotlib.rcParams.update(rcParams)
    matplotlib.rcParams.update({
        'font.family': 'MathJax_Main',
        'mathtext.fontset': 'cm',

        'figure.figsize': (12, 4),
        
        'font.size': 16,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    matplotlib.rcParams.update(**kw)