import numpy
from galpy.util import bovy_plot
from matplotlib import pyplot
from matplotlib.ticker import NullFormatter
from pixelFitDens import pixelAfeFeh
from pixelFitDF import get_options, _REFR0, _REFV0
from dfResultsTable import _GFIT, _KFIT
from calcDFResults import calcDFResults
def plotDFResults1D(options,args):
    #First load the results
    #Load g and k fits
    options.sample= 'g'
    options.select= 'all'
    gfits= calcDFResults(options,[_GFIT])
    options.sample= 'k'
    options.select= 'program'   
    kfits= calcDFResults(options,[_KFIT])
    #Setup plane
    tightbinned= pixelAfeFeh(None,dfeh=options.dfeh,dafe=options.dafe,
                             fehmin=-1.6,fehmax=0.5,afemin=-0.05,
                             afemax=0.55) 
    #Setup plotting
    bovy_plot.bovy_print(fig_height=8.,fig_width=8.,
                         axes_labelsize=11,xtick_labelsize=7,
                         ytick_labelsize=7,
                         xtick_major_size=2,
                         xtick_minor_size=1,
                         ytick_major_size=2,
                         ytick_minor_size=1)
    nullfmt   = NullFormatter()         # no labels
    if options.subtype.lower() == 'regular':
        npanels= 5
        keys= ['rd','zh','dlnvcdlnr','plhalo','vc']
        scale= [_REFR0,_REFR0*1000.,1.,1.,1.]
        ranges=[[1.5,4.5],
                [100.,800.],
                [-0.3,0.1],
                [0.,3.],
                [180.,260.]]
        labels=[r'$R_d^{\mathrm{MN}}\ [\mathrm{kpc}]$',
                r'$z_h^{\mathrm{MN}}\ [\mathrm{pc}]$',
                r'$\mathrm{d}\ln V_c / \mathrm{d}\ln R\, (R_0)$',
                r'$\alpha\ \mathrm{in}\ \rho_{\mathrm{halo}} \propto 1/r^\alpha$',
                r'$V_c\ [\mathrm{km\,s}^{-1}]$']                
        ticks= [[2.,3.,4.],
                [100.,300.,500.,700.],
                [-0.3,-0.1,0.1],
                [0.,1.5,3.],
                [180.,220.,260.]]
    elif options.subtype.lower() == 'valueadded':
        npanels= 4
        keys= ['rdexp','zhexp','surfzdisk','rhodm']
        scale= [_REFR0,_REFR0*1000.,1.,1.]
        ranges=[[1.5,4.5],
                [100.,800.],
                [40.,80.],
                [0.,0.02]]
        ticks= [[2.,3.,4.],
                [100.,300.,500.,700.],
                [40.,60.,80.],
                [0.,0.01,0.02]]
        labels=[r'$R_d^{\mathrm{Exp}}\ [\mathrm{kpc}]$',
                r'$z_h^{\mathrm{Exp}}\ [\mathrm{pc}]$',
                r'$\Sigma_{\mathrm{disk}}(R_0)\ [M_{\odot}\,\mathrm{pc}^{-2}]$',
                r'$\rho_{\mathrm{DM}}\,(R_0,0)\ [M_{\odot}\,\mathrm{pc}^{-3}]$']
    nrows= 2
    xrange=[-1.6,0.5]
    yrange=[-0.05,0.55]
    dx= 0.8/npanels
    dy= dx#*(yrange[1]-yrange[0])/(xrange[1]-xrange[0])
    allaxes= []
    for jj in range(nrows):
        for ii in range(npanels):
            if jj == 0: thesefits= gfits
            else: thesefits= kfits
            #Gather results
            plotthis= numpy.zeros((tightbinned.npixfeh(),tightbinned.npixafe()))
            key= keys[ii]
            for tt in range(tightbinned.npixfeh()):
                for rr in range(tightbinned.npixafe()):
                    tfeh= tightbinned.feh(tt)
                    tafe= tightbinned.afe(rr)
                    if numpy.amin((thesefits['feh']-tfeh)**2./options.dfeh**2.+(thesefits['afe']-tafe)**2./options.dafe**2.) > 0.:
                        plotthis[tt,rr]= numpy.nan
                        continue
                    tindx= numpy.argmin((thesefits['feh']-tfeh)**2./options.dfeh**2.+(thesefits['afe']-tafe)**2./options.dafe**2.)
                    plotthis[tt,rr]= thesefits[key][tindx]
            left, bottom, width, height= 0.1+ii*dx, 0.1+dy*(1-jj), dx, dy
            ax= pyplot.axes([left,bottom,width,height])
            allaxes.append(ax)
            fig= pyplot.gcf()
            fig.sca(ax)
            out= bovy_plot.bovy_dens2d(plotthis.T*scale[ii],
                                       origin='lower',cmap='jet',
                                       interpolation='nearest',
                                       xrange=xrange,yrange=yrange,
                                       vmin=ranges[ii][0],vmax=ranges[ii][1],
                                       contours=False,overplot=True,
                                       colorbar=False,aspect='auto')
            if jj == 1:
                bovy_plot._add_axislabels(r'$[\mathrm{Fe/H}]$',None)
            if ii == 0:
                bovy_plot._add_axislabels(None,r'$[\alpha/\mathrm{Fe}]$')
            if jj == 0:
                ax.xaxis.set_major_formatter(nullfmt)
            if ii > 0:
                ax.yaxis.set_major_formatter(nullfmt)
            if jj == 0:
                #Add colorbar
                left, bottom, width, height= 0.1+ii*dx, 0.1+2.*dy*(1-jj), dx, dy
                ax= pyplot.axes([left,bottom,width,height],frameon=False)
                ax.xaxis.set_major_formatter(nullfmt)
                ax.yaxis.set_major_formatter(nullfmt)
                ax.yaxis.set_tick_params(size=0)
                ax.xaxis.set_tick_params(size=0)
                allaxes.append(ax)
                fig.sca(ax)
                CB1= pyplot.colorbar(out,shrink=0.85,orientation='horizontal',
                                     fraction=0.25,ticks=ticks[ii])
                CB1.set_label(labels[ii],labelpad=-30)
    bovy_plot.bovy_end_print(options.outfilename)
    return None

def plotDFResults2D(options,args):
    pass

if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    if options.type.lower() == '1d':
        plotDFResults1D(options,args)
    elif options.type.lower() == '2d':
        plotDFResults2D(options,args)
