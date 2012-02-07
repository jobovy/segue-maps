import os, os.path
import numpy
import isodist
from galpy.util import bovy_plot
from segueSelect import _mr_gi, _gi_gr, _mr_ri_bright, _mr_ri_faint, \
    _ri_gr
_EXT='ps'
def plotAnIvezicDiff(dir):
    #Load An isochrones
    a= isodist.AnIsochrone()
    #Plot difference for a few metallicities
    fehs= [0.,-0.1,-0.2,-0.3,-0.5,-1.,-1.5]
    colors= ['b','c','g','y','orange','m','r']
    #Set up plot
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot([-100.,-100.],[-100.,-100],'k,',
                        xrange=[0.47,0.58],
                        yrange=[-0.25,.25],
                        xlabel=r'$g-r\ [\mathrm{mag}]$',
                        ylabel=r'$\mathrm{DM}_{\mathrm{An}}-\mathrm{DM}_{\mathrm{Ivezi\acute{c}}}\ [\mathrm{mag}]$')
    xlegend, ylegend, dy= 0.545, 0.2,-0.03
    for ii in range(len(fehs)):
        iso= a(numpy.log10(10.),feh=fehs[ii])
        #Get G dwarfs
        indx= (iso['g']-iso['r'] <= 0.55)*(iso['g']-iso['r'] >= 0.48)\
            *(iso['logg'] > 4.1)
        y= -1.*(iso['r'][indx]-_mr_gi(_gi_gr(iso['g'][indx]
                                             -iso['r'][indx]),fehs[ii]))
        bovy_plot.bovy_plot(iso['g'][indx]-iso['r'][indx],
                            y,
                            '-',color=colors[ii],
                            overplot=True)
        bovy_plot.bovy_text(xlegend,ylegend+ii*dy,
                            r'$[\mathrm{Fe/H]=%+4.1f}$' % fehs[ii],
                            color=colors[ii])
    bovy_plot.bovy_end_print(os.path.join(dir,'dm_an_ivezic.'+_EXT))

def plotJuricIvezicDiff(dir):
    #Plot difference for a few metallicities
    fehs= [0.,-0.1,-0.2,-0.3,-0.5,-1.,-1.5]
    colors= ['b','c','g','y','orange','m','r']
    #Set up plot
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot([-100.,-100.],[-100.,-100],'k,',
                        xrange=[0.47,0.58],
                        yrange=[-1.,1.],
                        xlabel=r'$g-r\ [\mathrm{mag}]$',
                        ylabel=r'$\mathrm{DM}_{\mathrm{Juri\acute{c}}}-\mathrm{DM}_{\mathrm{Ivezi\acute{c}}}\ [\mathrm{mag}]$')
    xlegend, ylegend, dy= 0.55, 0.8,-0.12
    grs= numpy.linspace(0.48,0.55,1001)
    for ii in range(len(fehs)):
        ybright= -1.*(_mr_ri_bright(_ri_gr(grs))-_mr_gi(_gi_gr(grs),fehs[ii]))
        yfaint= -1.*(_mr_ri_faint(_ri_gr(grs))-_mr_gi(_gi_gr(grs),fehs[ii]))
        bovy_plot.bovy_plot(grs,
                            ybright,
                            '-',color=colors[ii],
                            overplot=True)
        bovy_plot.bovy_plot(grs,
                            yfaint,
                            '--',color=colors[ii],
                            overplot=True)
        bovy_plot.bovy_text(xlegend,ylegend+ii*dy,
                            r'$[\mathrm{Fe/H]=%+4.1f}$' % fehs[ii],
                            color=colors[ii])
    bovy_plot.bovy_end_print(os.path.join(dir,'dm_juric_ivezic.'+_EXT))

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 2:
        plotJuricIvezicDiff(sys.argv[1])
    else:
        plotAnIvezicDiff(sys.argv[1])
