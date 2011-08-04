import numpy
import isodist
from galpy.util import bovy_plot
from segueSelect import _mr_gi, _gi_gr
_EXT='png'
def plotAnIvezicDiff():
    #Load An isochrones
    a= isodist.AnIsochrone()
    #Plot difference for a few metallicities
    fehs= [0.2,0.1,0.,-0.1,-0.2,-0.3,-0.5,-1.]
    colors= ['b','c','g','y','orange','m','r','k']
    #Set up plot
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot([-100.,-100.],[-100.,-100],'k,',
                        xrange=[0.47,0.58],
                        yrange=[-0.5,.5],
                        xlabel=r'$g-r\ [\mathrm{mag}]$',
                        ylabel=r'$\mathrm{DM}_{\mathrm{An}}-\mathrm{DM}_{\mathrm{Ivezic}}\ [\mathrm{mag}]$')
    xlegend, ylegend, dy= 0.55, 0.4,-0.06
    for ii in range(len(fehs)):
        iso= a(numpy.log10(10.),feh=fehs[ii])
        #Get G dwarfs
        indx= (iso['g']-iso['r'] <= 0.55)*(iso['g']-iso['r'] >= 0.48)\
            *(iso['logg'] > 4.1)
        y= -2.*(iso['r'][indx]-_mr_gi(_gi_gr(iso['g'][indx]
                                             -iso['r'][indx]),fehs[ii]))
        bovy_plot.bovy_plot(iso['g'][indx]-iso['r'][indx],
                            y,
                            '-',color=colors[ii],
                            overplot=True)
        bovy_plot.bovy_text(xlegend,ylegend+ii*dy,
                            r'$[\mathrm{Fe/H]=%+4.1f}$' % fehs[ii],
                            color=colors[ii])
    bovy_plot.bovy_end_print('dm_an_ivezic.'+_EXT)

if __name__ == '__main__':
    plotAnIvezicDiff()
