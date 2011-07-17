###############################################################################
#   Procedures to compare data and model
###############################################################################
_NRS= 1001
import numpy
from scipy import ndimage
from galpy.util import bovy_coords, bovy_plot
from fitDensz import _ivezic_dist, _ZSUN
###############################################################################
#   Density
###############################################################################
def comparerdistPlate(densfunc,params,sf,colordist,data,plate,
                      rmin=14.5,rmax=20.2,grmin=0.48,grmax=0.55,feh=-0.15,
                      convolve=0.,xrange=None,yrange=None,
                      overplot=False,bins=21,color='k'):
    """
    NAME:
       comparerdistPlate
    PURPOSE:
       compare the observed distribution of r-band mags to the model prediction
    INPUT:
       densfunc - underlying density function densfunc(R,z,params)
       params - array of parameters of the density function
       sf - segueSelect instance
       colordist - color g-r distribution
       data - data recarray (.dered_r is used)
       plate - plate number(s), or 'all', 'faint', or 'bright'
       rmin, rmax= minimum and maximum r
       grmin, grmax= minimum and maximum g-r
       feh= metallicity to use in d->r
       convolve= set to convolution kernel size (convolve predicted 
                 distribution with Gaussian of this width)
       xrange=, yrange= ranges for plot
       overplot= if True, overplot
       bins= hist number of bins
       color= color for model
    OUTPUT:
       plot to output
       return rdist, datahist, dataedges
    HISTORY:
       2011-07-15 - Written - Bovy (NYU)
    """
    rs= numpy.linspace(rmin,rmax,_NRS)
    platelb= bovy_coords.radec_to_lb(sf.platestr.ra,sf.platestr.dec,
                                     degree=True)
    allplates, faintplates, brightplates = False, False, False
    if isinstance(plate,str) and plate.lower() == 'all':
        plate= sf.plates
        allplates= True
    elif isinstance(plate,str) and plate.lower() == 'bright':
        plate= []
        for ii in range(len(sf.plates)):
            if not 'faint' in sf.platestr[ii].programname:
                plate.append(sf.plates[ii])
        brightplates= True
    elif isinstance(plate,str) and plate.lower() == 'faint':
        plate= []
        for ii in range(len(sf.plates)):
            if 'faint' in sf.platestr[ii].programname:
                plate.append(sf.plates[ii])
        faintplates= True
    elif isinstance(plate,int):
        plate= [plate]
    if isinstance(params,list): #list of samples
        pass
    else: #single value
        rdist= numpy.zeros(_NRS)
        platels, platebs= [], []
        for p in plate:
            #l and b?
            pindx= (sf.plates == p)
            platel= platelb[pindx,0][0]
            plateb= platelb[pindx,1][0]
            platels.append(platel)
            platebs.append(plateb)
            thisrdist= _predict_rdist_plate(rs,densfunc,params,rmin,rmax,
                                            platel,
                                            plateb,grmin,grmax,
                                            feh,colordist,sf,p)
            if 'faint' in sf.platestr[pindx].programname[0]:
                thisrdist[(rs < 17.8)]= 0.
            else:
                thisrdist[(rs > 17.8)]= 0.
            rdist+= thisrdist
        norm= numpy.nansum(rdist*(rs[1]-rs[0]))
        rdist/= norm
        if convolve > 0.:
            ndimage.filters.gaussian_filter1d(rdist,convolve/(rs[1]-rs[0]),
                                              output=rdist)
        if xrange is None:
            xrange= [numpy.amin(rs)-0.2,numpy.amax(rs)+0.1]
        if yrange is None:
            yrange= [0.,1.2*numpy.amax(rdist)]
        bovy_plot.bovy_plot(rs,rdist,ls='-',color=color,
                            xrange=xrange,yrange=yrange,
                            xlabel='$r_0\ [\mathrm{mag}]$',
                            ylabel='$\mathrm{density}$',overplot=overplot)
        #Plot the data
        data_dered_r= []
        for p in plate:
            data_dered_r.extend(data[(data.plate == p)].dered_r)
        hist= bovy_plot.bovy_hist(data_dered_r,
                                  normed=True,bins=bins,ec='k',
                                  histtype='step',
                                  overplot=True,range=xrange)
        if len(plate) > 1 and len(plate) < 10:
            platestr= '\mathrm{plates}\ \ '
            for ii in range(len(plate)-1):
                platestr= platestr+'%i, ' % plate[ii]
            platestr+= '%i' % plate[-1]
            lbstr= '$l = %i^\circ \pm %i^\circ$' % (int(numpy.mean(platels)),int(numpy.std(platels)))+'\n'\
                +'$b = %i^\circ \pm %i^\circ$' % (int(numpy.mean(platebs)),
            int(numpy.std(platebs)))
        elif allplates:
            platestr= '\mathrm{all\ plates}'
            bovy_plot.bovy_text(r'$'+platestr+'$'
                                +'\n'+
                                '$%i \ \ \mathrm{stars}$' % 
                                len(data_dered_r),top_right=True)
        elif brightplates:
            platestr= '\mathrm{bright\ plates}'
            bovy_plot.bovy_text(r'$'+platestr+'$'
                                +'\n'+
                                '$%i \ \ \mathrm{stars}$' % 
                                len(data_dered_r),top_right=True)
        elif faintplates:
            platestr= '\mathrm{faint\ plates}'
            bovy_plot.bovy_text(r'$'+platestr+'$'
                                +'\n'+
                                '$%i \ \ \mathrm{stars}$' % 
                                len(data_dered_r),top_right=True)
        elif len(plate) >= 10:
            platestr= '\mathrm{many\ plates}'
            lbstr= '$l = %i^\circ \pm %i^\circ$' % (
                int(numpy.mean(platels)),int(numpy.std(platels)))+'\n'\
                +'$b = %i^\circ\pm%i^\circ$' % (int(numpy.mean(platebs)),
            int(numpy.std(platebs)))
        else:
            platestr= '\mathrm{plate}\ \ %i' % plate[0]
            lbstr= '$l = %i^\circ$' % int(platel)+'\n'\
                +'$b = %i^\circ$' % int(plateb)           
        if not (allplates or brightplates or faintplates):
            bovy_plot.bovy_text(r'$'+platestr+'$'
                                +'\n'+
                                '$%i \ \ \mathrm{stars}$' % 
                                len(data_dered_r)
                                +'\n'+
                                lbstr,top_right=True)
        return (rdist, hist[0], hist[1])

###############################################################################
#            FORWARD MODELING FOR MODEL--DATA COMPARISON
###############################################################################
def _predict_rdist(rs,densfunc,params,rmin,rmax,platelb,grmin,grmax,
                   feh,sf,colordist):
    """BOVY ADAPT"""
    """Predict the r distribution for the sample"""
    plates= sf.plates
    out= numpy.zeros(len(rs))
    for ii in range(len(plates)):
        if 'faint' in sf.platestr[ii].programname:
            out+= sf(numpy.array([plates[ii] for jj in range(len(rs))]),
                     r=rs)*_predict_rdist_plate(rs,densfunc,params,17.8,
                                                rmax,platelb[ii,0],
                                                platelb[ii,1],
                                                grmin,grmax,
                                                feh,colordist)
        else:
            out+= sf(numpy.array([plates[ii] for jj in range(len(rs))]),
                     r=rs)*_predict_rdist_plate(rs,densfunc,params,rmin,
                                                      17.8,platelb[ii,0],
                                                      platelb[ii,1],
                                                      grmin,grmax,
                                                      feh,colordist)
    return out

def _predict_rdist_plate(rs,densfunc,params,rmin,rmax,l,b,grmin,grmax,
                         feh,colordist,sf,plate):
    """Predict the r distribution for a plate"""
    #BOVY: APPROXIMATELY INTEGRATE OVER GR
    ngr= 11
    grs= numpy.linspace(grmin,grmax,ngr)
    out= numpy.zeros(len(rs))
    norm= 0.
    for jj in range(ngr):
       #Calculate distances
        ds= _ivezic_dist(grs[jj],rs,feh)
        #Calculate (R,z)s
        XYZ= bovy_coords.lbd_to_XYZ(numpy.array([l for ii in range(len(ds))]),
                                    numpy.array([b for ii in range(len(ds))]),
                                    ds,degree=True)
        XYZ= XYZ.astype(numpy.float64)
        R= ((8.-XYZ[:,0])**2.+XYZ[:,1]**2.)**(0.5)
        XYZ[:,2]+= _ZSUN
        out+= ds**3.*densfunc(R,XYZ[:,2],params)*colordist(grs[jj])
        norm+= colordist(grs[jj])
    select= sf(numpy.array([plate for jj in range(len(rs))]),r=rs)
    out*= select
    out/= norm
    out[(rs < rmin)]= 0.
    out[(rs > rmax)]= 0.
    return out
