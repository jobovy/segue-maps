import os, os.path
import math
import numpy
from scipy import interpolate
import cPickle as pickle
from optparse import OptionParser
import monoAbundanceMW
from galpy.util import bovy_plot, save_pickles
from matplotlib import pyplot, cm
from galpy import potential
from galpy.actionAngle_src.actionAngleAdiabaticGrid import  actionAngleAdiabaticGrid
from galpy.df_src.quasiisothermaldf import quasiisothermaldf
from galpy.potential import MiyamotoNagaiPotential, LogarithmicHaloPotential, MWPotential
def plot_hrhrvshr(options,args):
    """Plot hr^out/hr^in as a function of hr for various sr"""
    if len(args) == 0.:
        print "Must provide a savefilename ..."
        print "Returning ..."
        return None
    if os.path.exists(args[0]):
        #Load
        savefile= open(args[0],'rb')
        plotthis= pickle.load(savefile)
        hrs= pickle.load(savefile)
        srs= pickle.load(savefile)
        savefile.close()
    else:
        #Grid of models to test
        hrs= numpy.linspace(options.hrmin,options.hrmax,options.nhr)
        srs= numpy.linspace(options.srmin,options.srmax,options.nsr)
        #Tile
        hrs= numpy.tile(hrs,(options.nsr,1)).T
        srs= numpy.tile(srs,(options.nhr,1))
        plotthis= numpy.zeros((options.nhr,options.nsr))
        #Setup potential and aA
        pot= MWPotential
        aA=actionAngleAdiabaticGrid(pot=pot,nR=16,nEz=16,nEr=31,nLz=31,zmax=1.,
                                    Rmax=5.)
        for ii in range(options.nhr):
            for jj in range(options.nsr):
                qdf= quasiisothermaldf(hrs[ii,jj]/8.,srs[ii,jj]/220.,
                                       srs[ii,jj]/2./220.,7./8.,7./8.,
                                       pot=pot,aA=aA)
                plotthis[ii,jj]= qdf.estimate_hr(1.,z=None,nR=101,nmc=100,
                                                 dR=numpy.amin([6./8.,
                                                                3./8.*hrs[ii,jj]]))/hrs[ii,jj]*8.
                print ii*options.nsr+jj+1, options.nsr*options.nhr, \
                    hrs[ii,jj], srs[ii,jj], plotthis[ii,jj]
        #Save
        save_pickles(args[0],plotthis,hrs,srs)
    #Now plot
    bovy_plot.bovy_print()
    indx= 0
    lines= []
    colors= [cm.jet(ii/float(options.nhr)*1.+0.) for ii in range(options.nhr)]
    lss= ['-' for ii in range(options.nhr)]#,'--','-.','..']
    labels= []
    lines.append(bovy_plot.bovy_plot(hrs[:,indx],plotthis[:,indx],
                                     color=colors[indx],ls=lss[indx],
                                     xrange=[0.,8.1],
                                     yrange=[0.8,1.4],
                                     xlabel=r'$h^{\mathrm{in}}_R\ \mathrm{at}\ 8\,\mathrm{kpc}$',
                                     ylabel=r'$h^{\mathrm{out}}_R / h^{\mathrm{in}}_R$'))
    labels.append(r'$\sigma_R = %.0f \,\mathrm{km\,s}^{-1}$' % srs[0,indx])
    for indx in range(1,options.nhr):
        lines.append(bovy_plot.bovy_plot(hrs[:,indx],plotthis[:,indx],
                                         color=colors[indx],ls=lss[indx],
                                         overplot=True))
        labels.append(r'$\sigma_R = %.0f \,\mathrm{km\,s}^{-1}$' % srs[0,indx])
    """
    #Legend
    pyplot.legend(lines,#(line1[0],line2[0],line3[0],line4[0]),
                  labels,#(r'$v_{bc} = 0$',
#                   r'$v_{bc} = 1\,\sigma_{bc}$',
#                   r'$v_{bc} = 2\,\sigma_{bc}$',
#                   r'$v_{bc} = 3\,\sigma_{bc}$'),
                  loc='lower right',#bbox_to_anchor=(.91,.375),
                  numpoints=2,
                  prop={'size':14},
                  frameon=False)
    """
     #Add colorbar
    map = cm.ScalarMappable(cmap=cm.jet)
    map.set_array(srs[0,:])
    map.set_clim(vmin=numpy.amin(srs[0,:]),vmax=numpy.amax(srs[0,:]))
    cbar= pyplot.colorbar(map,fraction=0.15)
    cbar.set_clim(numpy.amin(srs[0,:]),numpy.amax(srs[0,:]))
    cbar.set_label(r'$\sigma_R \,[\mathrm{km\,s}^{-1}]$')
    bovy_plot.bovy_end_print(options.plotfilename)

def plot_hzszq(options,args):
    """Plot sz,hz, q"""
    if len(args) == 0.:
        print "Must provide a savefilename ..."
        print "Returning ..."
        return None
    nqs, nszs, nhzs= 31, 51, 51
    #nqs, nszs, nhzs= 5,5,5
    if os.path.exists(args[0]):
        #Load
        savefile= open(args[0],'rb')
        hzs= pickle.load(savefile)
        szs= pickle.load(savefile)
        qs= pickle.load(savefile)
        savefile.close()
    else:
        qs= numpy.linspace(0.5,1.,nqs)
        szs= numpy.linspace(15.,50.,nszs)
        hzs= numpy.zeros((nqs,nszs))
        for ii in range(nqs):
            print "Working on potential %i / %i ..." % (ii+1,nqs)
            #Setup potential
            lp= LogarithmicHaloPotential(normalize=1.,q=qs[ii])
            aA= actionAngleAdiabaticGrid(pot=lp,nR=16,
                                         nEz=16,nEr=31,
                                         nLz=31,
                                         zmax=1.,
                                         Rmax=5.)
            for jj in range(nszs):
                qdf= quasiisothermaldf(options.hr/8.,2.*szs[jj]/220.,
                                       szs[jj]/220.,7./8.,7./8.,pot=lp,
                                       aA=aA,cutcounter=True)    
                hzs[ii,jj]= qdf.estimate_hz(1.)
        #Save
        save_pickles(args[0],hzs,szs,qs)
    #Re-sample
    hzsgrid= numpy.linspace(50.,1500.,nhzs)/8000.
    qs2d= numpy.zeros((nhzs,nszs))
    for ii in range(nszs):
        interpQ= interpolate.UnivariateSpline(hzs[:,ii],qs,k=3)
        qs2d[:,ii]= interpQ(hzsgrid)
        qs2d[(hzsgrid < hzs[0,ii]),ii]= numpy.nan
        qs2d[(hzsgrid > hzs[-1,ii]),ii]= numpy.nan
    #Now plot
    bovy_plot.bovy_print()
    bovy_plot.bovy_dens2d(qs2d.T,origin='lower',cmap='jet',
                          interpolation='gaussian',
#                          interpolation='nearest',
                          ylabel=r'$\sigma_z\ [\mathrm{km\,s}^{-1}]$',
                          xlabel=r'$h_z\ [\mathrm{pc}]$',
                          zlabel=r'$\mathrm{flattening}\ q$',
                          yrange=[szs[0],szs[-1]],
                          xrange=[8000.*hzsgrid[0],8000.*hzsgrid[-1]],
#                          vmin=0.5,vmax=1.,
                           contours=False,
                           colorbar=True,shrink=0.78)
    _OVERPLOTMAPS= True
    if _OVERPLOTMAPS:
        fehs= monoAbundanceMW.fehs()
        afes= monoAbundanceMW.afes()
        npops= len(fehs)
        mapszs= []
        maphzs= []
        for ii in range(npops):
            thissz, thiserr= monoAbundanceMW.sigmaz(fehs[ii],afes[ii],err=True)
            if thiserr/thissz > 0.1:
                continue
            thishz, thiserr= monoAbundanceMW.hz(fehs[ii],afes[ii],err=True)
            if thiserr/thishz > 0.1:
                continue
            mapszs.append(thissz)
            maphzs.append(thishz)
        mapszs= numpy.array(mapszs)
        maphzs= numpy.array(maphzs)
        bovy_plot.bovy_plot(maphzs,mapszs,'ko',overplot=True,mfc='none',mew=1.5)
    bovy_plot.bovy_text(r'$h_R = %i\,\mathrm{kpc}$' % int(options.hr),
                        bottom_right=True)
    bovy_plot.bovy_end_print(options.plotfilename)
                           

def get_options():
    usage = "usage: %prog [options] <savefile>\n\nsavefile= name of the file that will hold the data to be plotted"
    parser = OptionParser(usage=usage)
    parser.add_option("-t","--type",dest='type',default=None,
                      help="Type of thing to do")
    parser.add_option("-o",dest='plotfilename',default=None,
                      help="Name for output plot")
    parser.add_option("--rmin",dest='rmin',type='float',
                      default=4.,
                      help="Minimum radius")
    parser.add_option("--rmax",dest='rmax',type='float',
                      default=8.,
                      help="Maximum radius")
    parser.add_option("--hr",dest='hr',type='float',
                      default=3.,
                      help="Scale length (kpc)")
    parser.add_option("--hrmin",dest='hrmin',type='float',
                      default=1.,
                      help="Minimum scale length")
    parser.add_option("--hrmax",dest='hrmax',type='float',
                      default=5.,
                      help="Maximum scale length")
    parser.add_option("--srmin",dest='srmin',type='float',
                      default=20.,
                      help="Minimum scale length")
    parser.add_option("--srmax",dest='srmax',type='float',
                      default=80.,
                      help="Maximum scale length")
    parser.add_option("--nr",dest='nr',default=2,type='int',
                      help="Number of r to use")
    parser.add_option("--nhr",dest='nhr',default=11,type='int',
                      help="Number of hr to use")
    parser.add_option("--nsr",dest='nsr',default=11,type='int',
                      help="Number of sr to use")
    return parser

if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    numpy.random.seed(1)
    if options.type.lower() == 'hrhrhr':
        plot_hrhrvshr(options,args)
    elif options.type.lower() == 'hrhrr':
        plot_hrhrvsr(options,args)
    elif options.type.lower() == 'hzszq':
        plot_hzszq(options,args)

