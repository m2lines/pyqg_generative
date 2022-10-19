from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cmocean

def show_fonts():
    import matplotlib.font_manager
    flist = matplotlib.font_manager.get_fontconfig_fonts()
    return [matplotlib.font_manager.FontProperties(fname=fname).get_name() for fname in flist]

def show_rcparams(name):
    for key, val in matplotlib.rcParams.items():
        if name in key:
            print(key, val)

def default_rcParams(kw={}):
    '''
    Also matplotlib.rcParamsDefault contains the default values,
    but:
    - backend is changed
    - without plotting something as initialization,
    inline does not work
    '''
    plt.plot()
    plt.close()
    rcParams = matplotlib.rcParamsDefault.copy()
    rcParams.pop('backend') # can break inlining
    matplotlib.rcParams.update(rcParams)
    
    matplotlib.rcParams.update({
        'font.family': 'MathJax_Main',
        'mathtext.fontset': 'cm',

        'figure.figsize': (10, 2),

        'figure.subplot.wspace': 0.3,
        
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,

        'axes.formatter.limits': (-1,2),
        'axes.formatter.use_mathtext': True,
        'axes.labelpad': -2,
        'axes.titlelocation' : 'right',
        
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    matplotlib.rcParams.update(**kw)

def set_letters():
    fig = plt.gcf()
    axes = fig.axes
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
    j = 0
    for ax in axes:
        subplot=False
        for key in ax.__dict__.keys():
            if 'subplot' in key:
                subplot = True
        if subplot:
            ax.text(0.03,0.9,f'({letters[j]})', transform = ax.transAxes, fontweight='bold', fontsize=11)
            j += 1

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
    
    # Return axis to initial image
    plt.sca(ax)
    return im