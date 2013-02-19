###############################################################################
#   Procedures to compare data and model
###############################################################################
_NRS= 1001
_NZS= 1001
_legendsize= 16
import sys
import os, os.path
import time
import math as m
import numpy
from scipy import ndimage
from galpy.util import bovy_coords, bovy_plot, multi
import matplotlib
from fitDensz import _ivezic_dist, _ZSUN, _DEGTORAD, _gi_gr, _mr_gi
from segueSelect import _ERASESTR, _SEGUESELECTDIR, _load_fits
from fitSigz import _APOORFEHRANGE, _ARICHFEHRANGE
###############################################################################
#   Density
###############################################################################
def comparerdistPlate(densfunc,params,sf,colordist,fehdist,data,plate,
                      rmin=14.5,rmax=20.2,grmin=0.48,grmax=0.55,fehmin=-0.4,
                      fehmax=0.5,feh=-0.15,
                      convolve=0.,xrange=None,yrange=None,
                      overplot=False,bins=21,color='k',ls='-',
                      left_legend=None,right_legend=None):
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
       fehdist - FeH distribution
       data - data recarray (.dered_r is used)
       plate - plate number(s), or 'all', 'faint', or 'bright'
       rmin, rmax= minimum and maximum r
       grmin, grmax= minimum and maximum g-r
       fehmin, fehmax= minimum and maximum [Fe/H]
       feh= metallicity to use in d->r
       convolve= set to convolution kernel size (convolve predicted 
                 distribution with Gaussian of this width)
       xrange=, yrange= ranges for plot
       overplot= if True, overplot
       bins= hist number of bins
       color= color for model
       left_legend = if set, legend to put at the left top
       right_legend = if set, legend to put at the right top
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
    allfaint, allbright= True, True
    if isinstance(params,list): #list of samples
        pass
    else: #single value
        rdist= numpy.zeros(_NRS)
        platels, platebs= [], []
        cnt= 0
        for p in plate:
            cnt+= 1
            #l and b?
            sys.stdout.write('\r'+"Working on plate %i (%i/%i)" % (p,cnt,len(plate)))
            sys.stdout.flush()
            pindx= (sf.plates == p)
            platel= platelb[pindx,0][0]
            plateb= platelb[pindx,1][0]
            platels.append(platel)
            platebs.append(plateb)
            thisrdist= _predict_rdist_plate(rs,densfunc,params,rmin,rmax,
                                            platel,
                                            plateb,grmin,grmax,
                                            fehmin,fehmax,
                                            feh,colordist,fehdist,sf,p)
            if 'faint' in sf.platestr[pindx].programname[0]:
                thisrdist[(rs < 17.8)]= 0.
                allbright= False
            else:
                thisrdist[(rs > 17.8)]= 0.
                allfaint= False
            rdist+= thisrdist
        sys.stdout.write('\r'+_ERASESTR+'\r')
        sys.stdout.flush()
        norm= numpy.nansum(rdist*(rs[1]-rs[0]))
        rdist/= norm
        if convolve > 0.:
            ndimage.filters.gaussian_filter1d(rdist,convolve/(rs[1]-rs[0]),
                                              output=rdist)
        if xrange is None:
            xrange= [numpy.amin(rs)-0.2,numpy.amax(rs)+0.6]
        if yrange is None:
            yrange= [0.,1.6*numpy.amax(rdist)]
        bovy_plot.bovy_plot(rs,rdist,ls=ls,color=color,
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
        if not right_legend is None:
            bovy_plot.bovy_text(right_legend,top_right=True,size=_legendsize)
        if len(plate) > 1 and len(plate) < 9:
            platestr= '\mathrm{plates}\ \ '
            for ii in range(len(plate)-1):
                platestr= platestr+'%i, ' % plate[ii]
            platestr+= '%i' % plate[-1]
            lbstr= '$l = %i^\circ \pm %i^\circ$' % (int(numpy.mean(platels)),int(numpy.std(platels)))+'\n'\
                +'$b = %i^\circ \pm %i^\circ$' % (int(numpy.mean(platebs)),
            int(numpy.std(platebs)))
        elif allplates:
            platestr= '\mathrm{all\ plates}'
            if right_legend is None:
                bovy_plot.bovy_text(r'$'+platestr+'$'
                                    +'\n'+
                                    '$%i \ \ \mathrm{stars}$' % 
                                    len(data_dered_r),top_right=True,
                                    size=_legendsize)
        elif brightplates:
            platestr= '\mathrm{bright\ plates}'
            if right_legend is None:
                bovy_plot.bovy_text(r'$'+platestr+'$'
                                    +'\n'+
                                    '$%i \ \ \mathrm{stars}$' % 
                                    len(data_dered_r),top_right=True,
                                    size=_legendsize)
        elif faintplates:
            platestr= '\mathrm{faint\ plates}'
            if right_legend is None:
                bovy_plot.bovy_text(r'$'+platestr+'$'
                                    +'\n'+
                                    '$%i \ \ \mathrm{stars}$' % 
                                    len(data_dered_r),top_right=True,
                                    size=_legendsize)
        elif len(plate) >= 9:
            platestr= '%i\ \mathrm{plates}' % len(plate)
            lbstr= '$l = %i^\circ \pm %i^\circ$' % (
                int(numpy.mean(platels)),int(numpy.std(platels)))+'\n'\
                +'$b = %i^\circ\pm%i^\circ$' % (int(numpy.mean(platebs)),
            int(numpy.std(platebs)))
        else:
            platestr= '\mathrm{plate}\ \ %i' % plate[0]
            lbstr= '$l = %i^\circ$' % int(platel)+'\n'\
                +'$b = %i^\circ$' % int(plateb)           
        if not (allplates or brightplates or faintplates):
            if right_legend is None:
                bovy_plot.bovy_text(r'$'+platestr+'$'
                                    +'\n'+
                                    '$%i \ \ \mathrm{stars}$' % 
                                    len(data_dered_r)
                                    +'\n'+
                                    lbstr,top_right=True,
                                    size=_legendsize)
            #Overplot direction in (R,z) plane
            ax= matplotlib.pyplot.gca()
            yrange= ax.get_ylim()
            dy= yrange[1]-yrange[0]
            rx, ry,dr, dz= xrange[1]-2.1, yrange[1]-0.5*dy, 2., 0.4*dy
            #x-axis
            bovy_plot.bovy_plot([rx-0.2,rx-0.2+dr],
                                [ry,ry],
                                'k-',overplot=True)
            #y-axis
            bovy_plot.bovy_plot([rx,rx],
                                [ry-dz/2.,ry+dz/2.],
                                'k-',overplot=True)
            #Sun's position
            bovy_plot.bovy_plot([rx+dr/2.],[ry],'ko',overplot=True)
            #Draw los
            gr= (grmax+grmin)/2.
            if allbright:
                thisrmin= rmin
                thisrmax= 17.8
            if allfaint:
                thisrmin= 17.8
                thisrmax= rmax
            dmin, dmax= _ivezic_dist(gr,thisrmin,feh), _ivezic_dist(gr,thisrmax,feh)
            ds= numpy.linspace(dmin,dmax,101)
            xyzs= bovy_coords.lbd_to_XYZ(numpy.array([numpy.mean(platels)+numpy.std(platels) for ii in range(len(ds))]),
                                         numpy.array([numpy.mean(platebs) for ii in range(len(ds))]),
                                         ds,degree=True).astype('float')
            rs= (((8.-xyzs[:,0])**2.+xyzs[:,1]**2.)**0.5)/8.*dr/2.+rx
            zs= xyzs[:,2]/8.*dz/2.+ry
            bovy_plot.bovy_plot(rs,zs,'-',color='0.75',overplot=True)
            xyzs= bovy_coords.lbd_to_XYZ(numpy.array([numpy.mean(platels)-numpy.std(platels) for ii in range(len(ds))]),
                                         numpy.array([numpy.mean(platebs) for ii in range(len(ds))]),
                                         ds,degree=True).astype('float')
            rs= (((8.-xyzs[:,0])**2.+xyzs[:,1]**2.)**0.5)/8.*dr/2.+rx
            zs= xyzs[:,2]/8.*dz/2.+ry
            bovy_plot.bovy_plot(rs,zs,'-',color='0.75',overplot=True)
            xyzs= bovy_coords.lbd_to_XYZ(numpy.array([numpy.mean(platels) for ii in range(len(ds))]),
                                         numpy.array([numpy.mean(platebs)+numpy.std(platebs) for ii in range(len(ds))]),
                                         ds,degree=True).astype('float')
            rs= (((8.-xyzs[:,0])**2.+xyzs[:,1]**2.)**0.5)/8.*dr/2.+rx
            zs= xyzs[:,2]/8.*dz/2.+ry
            bovy_plot.bovy_plot(rs,zs,'-',color='0.75',overplot=True)
            xyzs= bovy_coords.lbd_to_XYZ(numpy.array([numpy.mean(platels) for ii in range(len(ds))]),
                                         numpy.array([numpy.mean(platebs)-numpy.std(platebs) for ii in range(len(ds))]),
                                         ds,degree=True).astype('float')
            rs= (((8.-xyzs[:,0])**2.+xyzs[:,1]**2.)**0.5)/8.*dr/2.+rx
            zs= xyzs[:,2]/8.*dz/2.+ry
            bovy_plot.bovy_plot(rs,zs,'-',color='0.75',overplot=True)
            xyzs= bovy_coords.lbd_to_XYZ(numpy.array([numpy.mean(platels) for ii in range(len(ds))]),
                                         numpy.array([numpy.mean(platebs) for ii in range(len(ds))]),
                                         ds,degree=True).astype('float')
            rs= (((8.-xyzs[:,0])**2.+xyzs[:,1]**2.)**0.5)/8.*dr/2.+rx
            zs= xyzs[:,2]/8.*dz/2.+ry
            bovy_plot.bovy_plot(rs,zs,'k-',overplot=True)
            bovy_plot.bovy_text(rx+3./4.*dr,ry-0.1*dz,r'$R$')
            bovy_plot.bovy_text(rx-0.2,ry+3./4.*dz/2.,r'$Z$')
        if not left_legend is None:
            bovy_plot.bovy_text(left_legend,top_left=True,size=_legendsize)
        return (rdist, hist[0], hist[1])

def comparezdistPlate(densfunc,params,sf,colordist,fehdist,data,plate,
                      rmin=14.5,rmax=20.2,grmin=0.48,grmax=0.55,fehmin=-0.4,
                      fehmax=0.5,feh=-0.15,
                      convolve=0.,xrange=None,yrange=None,
                      overplot=False,bins=21,color='k',ls='-',
                      left_legend=None,right_legend=None):
    """
    NAME:
       comparezdistPlate
    PURPOSE:
       compare the observed distribution of heights zto the model prediction
    INPUT:
       densfunc - underlying density function densfunc(R,z,params)
       params - array of parameters of the density function
       sf - segueSelect instance
       colordist - color g-r distribution
       fehdist - Fe/H distribution of the sample
       data - data recarray (.dered_r is used)
       plate - plate number(s), or 'all', 'faint', or 'bright'
       rmin, rmax= minimum and maximum r
       grmin, grmax= minimum and maximum g-r
       fehmin, fehmax= minimum and maximum [Fe/H]
       feh= metallicity to use in d->r
       convolve= set to convolution kernel size (convolve predicted 
                 distribution with Gaussian of this width)
       xrange=, yrange= ranges for plot
       overplot= if True, overplot
       bins= hist number of bins
       color= color for model
       left_legend = if set, legend to put at the left top
       right_legend = if set, legend to put at the right top
    OUTPUT:
       plot to output
       return rdist, datahist, dataedges
    HISTORY:
       2011-07-29 - Written - Bovy (NYU)
    """
    #Set up plates
    platelb= bovy_coords.radec_to_lb(sf.platestr.ra,sf.platestr.dec,
                                     degree=True)
    platelb= platelb.astype('float')
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
    allfaint, allbright= True, True
    #Zmin and Zmax for this rmin, rmax
    bs= []
    for p in plate:
        #l and b?
        pindx= (sf.plates == p)
        plateb= platelb[pindx,1][0]
        bs.append(plateb)
        if 'faint' in sf.platestr[pindx].programname[0]:
            allbright= False
        else:
            allfaint= False
    bs= numpy.array(bs)
    bmin, bmax= numpy.amin(numpy.fabs(bs)),numpy.amax(numpy.fabs(bs))
    if allbright:
        thisrmin, thisrmax= rmin, 17.8
    elif allfaint:
        thisrmin, thisrmax= 17.8, rmax
    else:
        thisrmin, thisrmax= rmin, rmax
    _NGR, _NFEH= 51, 51
    grs= numpy.zeros((_NGR,_NFEH))
    fehs= numpy.zeros((_NGR,_NFEH))
    for ii in range(_NGR):
        if feh > -0.5: #rich, actually only starts at 0.05
            fehs[ii,:]= numpy.linspace(fehmin,0.05,_NFEH)
        else:
            fehs[ii,:]= numpy.linspace(fehmin,fehmax,_NFEH)
    for ii in range(_NFEH):
        grs[:,ii]= numpy.linspace(grmin,grmax,_NGR)
    sys.stdout.write('\r'+"Determining minimal and maximal distance")
    sys.stdout.flush()
    dmin= numpy.amin(_ivezic_dist(grs,thisrmin,fehs))
    dmax= numpy.amax(_ivezic_dist(grs,thisrmax,fehs))
    sys.stdout.write('\r'+_ERASESTR+'\r')
    sys.stdout.flush()
    zmin, zmax= dmin*numpy.sin(bmin*_DEGTORAD), dmax*numpy.sin(bmax*_DEGTORAD)
    zs= numpy.linspace(zmin,zmax,_NZS)
    if isinstance(params,list): #list of samples
        pass
    else: #single value
        zdist= numpy.zeros(_NZS)
        platels, platebs= [], []
        cnt= 0
        for p in plate:
            cnt+= 1
            sys.stdout.write('\r'+"Working on plate %i (%i/%i)" % (p,cnt,len(plate)))
            sys.stdout.flush()
            #l and b?
            pindx= (sf.plates == p)
            platel= platelb[pindx,0][0]
            plateb= platelb[pindx,1][0]
            platels.append(platel)
            platebs.append(plateb)
            thiszdist= _predict_zdist_plate(zs,densfunc,params,rmin,rmax,
                                            platel,
                                            plateb,grmin,grmax,
                                            fehmin,fehmax,
                                            feh,colordist,fehdist,sf,p)
            zdist+= thiszdist
        sys.stdout.write('\r'+_ERASESTR+'\r')
        sys.stdout.flush()
        norm= numpy.nansum(zdist*(zs[1]-zs[0]))
        zdist/= norm
        if convolve > 0.:
            ndimage.filters.gaussian_filter1d(zdist,convolve/(zs[1]-zs[0]),
                                              output=zdist)
        if xrange is None:
            if allplates or brightplates:
                xrange=[-0.2,5]
            else:
                if feh > -0.5:
                    xrange= [numpy.amin(zs)-0.2,
                             0.7*(numpy.amax(zs)-numpy.amin(zs)+0.5)+numpy.amin(zs)-0.2]
                else:
                    xrange= [numpy.amin(zs)-0.2,
                             1*(numpy.amax(zs)-numpy.amin(zs)+0.5)+numpy.amin(zs)-0.2]
        if yrange is None:
            yrange= [0.,1.65*numpy.amax(zdist)]
        bovy_plot.bovy_plot(zs,zdist,ls=ls,color=color,
                            xrange=xrange,yrange=yrange,
                            xlabel='$|Z|\ [\mathrm{kpc}]$',
                            ylabel='$\mathrm{density}$',overplot=overplot)
        #Plot the data
        data_z= []
        for p in plate:
            data_z.extend(numpy.fabs(data[(data.plate == p)].zc))
        hist= bovy_plot.bovy_hist(data_z,
                                  normed=True,bins=bins,ec='k',
                                  histtype='step',
                                  overplot=True,range=[zmin,zmax])
        if not right_legend is None:
            bovy_plot.bovy_text(right_legend,top_right=True,size=_legendsize)
        if len(plate) > 1 and len(plate) < 9:
            platestr= '\mathrm{plates}\ \ '
            for ii in range(len(plate)-1):
                platestr= platestr+'%i, ' % plate[ii]
            platestr+= '%i' % plate[-1]
            lbstr= '$l = %i^\circ \pm %i^\circ$' % (int(numpy.mean(platels)),int(numpy.std(platels)))+'\n'\
                +'$b = %i^\circ \pm %i^\circ$' % (int(numpy.mean(platebs)),
            int(numpy.std(platebs)))
        elif allplates:
            platestr= '\mathrm{all\ plates}'
            if right_legend is None:
                bovy_plot.bovy_text(r'$'+platestr+'$'
                                    +'\n'+
                                    '$%i \ \ \mathrm{stars}$' % 
                                    len(data_z),top_right=True,
                                    size=_legendsize)
        elif brightplates:
            platestr= '\mathrm{bright\ plates}'
            if right_legend is None:
                bovy_plot.bovy_text(r'$'+platestr+'$'
                                    +'\n'+
                                    '$%i \ \ \mathrm{stars}$' % 
                                    len(data_z),top_right=True,
                                    size=_legendsize)
        elif faintplates:
            platestr= '\mathrm{faint\ plates}'
            if right_legend is None:
                bovy_plot.bovy_text(r'$'+platestr+'$'
                                    +'\n'+
                                    '$%i \ \ \mathrm{stars}$' % 
                                    len(data_z),top_right=True,
                                    size=_legendsize)
        elif len(plate) >= 9:
            platestr= '%i\ \mathrm{plates}' % len(plate)
            lbstr= '$l = %i^\circ \pm %i^\circ$' % (
                int(numpy.mean(platels)),int(numpy.std(platels)))+'\n'\
                +'$b = %i^\circ\pm%i^\circ$' % (int(numpy.mean(platebs)),
            int(numpy.std(platebs)))
        else:
            platestr= '\mathrm{plate}\ \ %i' % plate[0]
            lbstr= '$l = %i^\circ$' % int(platel)+'\n'\
                +'$b = %i^\circ$' % int(plateb)           
        if not (allplates or brightplates or faintplates):
            if right_legend is None:
                bovy_plot.bovy_text(r'$'+platestr+'$'
                                    +'\n'+
                                    '$%i \ \ \mathrm{stars}$' % 
                                    len(data_z)
                                    +'\n'+
                                    lbstr,top_right=True,
                                    size=_legendsize)
            #Overplot direction in (R,z) plane
            ax= matplotlib.pyplot.gca()
            yrange= ax.get_ylim()
            dy= yrange[1]-yrange[0]
            xfac= 1./(20.8-14.5)*(xrange[1]-xrange[0])
            rx, ry,dr, dz= xrange[1]-2.1*xfac,yrange[1]-0.5*dy, 2.*xfac, 0.4*dy
            #x-axis
            bovy_plot.bovy_plot([rx-0.2*xfac,rx-0.2*xfac+dr],
                                [ry,ry],
                                'k-',overplot=True)
            #y-axis
            bovy_plot.bovy_plot([rx,rx],
                                [ry-dz/2.,ry+dz/2.],
                                'k-',overplot=True)
            #Sun's position
            bovy_plot.bovy_plot([rx+dr/2.],[ry],'ko',overplot=True)
            #Draw los
            gr= (grmax+grmin)/2.
            if allbright:
                thisrmin= rmin
                thisrmax= 17.8
            if allfaint:
                thisrmin= 17.8
                thisrmax= rmax
            dmin, dmax= _ivezic_dist(gr,thisrmin,feh), _ivezic_dist(gr,thisrmax,feh)
            ds= numpy.linspace(dmin,dmax,101)
            xyzs= bovy_coords.lbd_to_XYZ(numpy.array([numpy.mean(platels)+numpy.std(platels) for ii in range(len(ds))]),
                                         numpy.array([numpy.mean(platebs) for ii in range(len(ds))]),
                                         ds,degree=True).astype('float')
            rs= (((8.-xyzs[:,0])**2.+xyzs[:,1]**2.)**0.5)/8.*dr/2.+rx
            zs= xyzs[:,2]/8.*dz/2.+ry
            bovy_plot.bovy_plot(rs,zs,'-',color='0.75',overplot=True)
            xyzs= bovy_coords.lbd_to_XYZ(numpy.array([numpy.mean(platels)-numpy.std(platels) for ii in range(len(ds))]),
                                         numpy.array([numpy.mean(platebs) for ii in range(len(ds))]),
                                         ds,degree=True).astype('float')
            rs= (((8.-xyzs[:,0])**2.+xyzs[:,1]**2.)**0.5)/8.*dr/2.+rx
            zs= xyzs[:,2]/8.*dz/2.+ry
            bovy_plot.bovy_plot(rs,zs,'-',color='0.75',overplot=True)
            xyzs= bovy_coords.lbd_to_XYZ(numpy.array([numpy.mean(platels) for ii in range(len(ds))]),
                                         numpy.array([numpy.mean(platebs)+numpy.std(platebs) for ii in range(len(ds))]),
                                         ds,degree=True).astype('float')
            rs= (((8.-xyzs[:,0])**2.+xyzs[:,1]**2.)**0.5)/8.*dr/2.+rx
            zs= xyzs[:,2]/8.*dz/2.+ry
            bovy_plot.bovy_plot(rs,zs,'-',color='0.75',overplot=True)
            xyzs= bovy_coords.lbd_to_XYZ(numpy.array([numpy.mean(platels) for ii in range(len(ds))]),
                                         numpy.array([numpy.mean(platebs)-numpy.std(platebs) for ii in range(len(ds))]),
                                         ds,degree=True).astype('float')
            rs= (((8.-xyzs[:,0])**2.+xyzs[:,1]**2.)**0.5)/8.*dr/2.+rx
            zs= xyzs[:,2]/8.*dz/2.+ry
            bovy_plot.bovy_plot(rs,zs,'-',color='0.75',overplot=True)
            xyzs= bovy_coords.lbd_to_XYZ(numpy.array([numpy.mean(platels) for ii in range(len(ds))]),
                                         numpy.array([numpy.mean(platebs) for ii in range(len(ds))]),
                                         ds,degree=True).astype('float')
            rs= (((8.-xyzs[:,0])**2.+xyzs[:,1]**2.)**0.5)/8.*dr/2.+rx
            zs= xyzs[:,2]/8.*dz/2.+ry
            bovy_plot.bovy_plot(rs,zs,'k-',overplot=True)
            bovy_plot.bovy_text(rx+3./4.*dr,ry-0.1*dz,r'$R$')
            bovy_plot.bovy_text(rx-0.25*xfac,ry+3./4.*dz/2.,r'$Z$')
        if not left_legend is None:
            bovy_plot.bovy_text(left_legend,top_left=True,size=_legendsize)
        return (zdist, hist[0], hist[1])

def compareRdistPlate(densfunc,params,sf,colordist,fehdist,data,plate,
                      rmin=14.5,rmax=20.2,grmin=0.48,grmax=0.55,fehmin=-0.4,
                      fehmax=0.5,feh=-0.15,
                      convolve=0.05,xrange=None,yrange=None,
                      overplot=False,bins=21,color='k',ls='-',
                      left_legend=None,right_legend=None):
    """
    NAME:
       compareRdistPlate
    PURPOSE:
       compare the observed distribution of radii R to the model prediction
    INPUT:
       densfunc - underlying density function densfunc(R,z,params)
       params - array of parameters of the density function
       sf - segueSelect instance
       colordist - color g-r distribution
       fehdist - Fe/H distribution of the sample
       data - data recarray (.dered_r is used)
       plate - plate number(s), or 'all', 'faint', or 'bright'
       rmin, rmax= minimum and maximum r
       grmin, grmax= minimum and maximum g-r
       fehmin, fehmax= minimum and maximum [Fe/H]
       feh= metallicity to use in d->r
       convolve= set to convolution kernel size (convolve predicted 
                 distribution with Gaussian of this width)
       xrange=, yrange= ranges for plot
       overplot= if True, overplot
       bins= hist number of bins
       color= color for model
       left_legend = if set, legend to put at the left top
       right_legend = if set, legend to put at the right top
    OUTPUT:
       plot to output
       return rdist, datahist, dataedges
    HISTORY:
       2011-07-30 - Written - Bovy (NYU)
    """
    #Set up plates
    platelb= bovy_coords.radec_to_lb(sf.platestr.ra,sf.platestr.dec,
                                     degree=True)
    platelb= platelb.astype('float')
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
    allfaint, allbright= True, True
    #Rmin and Rmax for this rmin, rmax
    bs= []
    for p in plate:
        #l and b?
        pindx= (sf.plates == p)
        plateb= platelb[pindx,1][0]
        bs.append(plateb)
        if 'faint' in sf.platestr[pindx].programname[0]:
            allbright= False
        else:
            allfaint= False
    bs= numpy.array(bs)
    bmin, bmax= numpy.amin(numpy.fabs(bs)),numpy.amax(numpy.fabs(bs))
    if allbright:
        thisrmin, thisrmax= rmin, 17.8
    elif allfaint:
        thisrmin, thisrmax= 17.8, rmax
    else:
        thisrmin, thisrmax= rmin, rmax
    _NGR, _NFEH= 51, 51
    grs= numpy.zeros((_NGR,_NFEH))
    fehs= numpy.zeros((_NGR,_NFEH))
    for ii in range(_NGR):
        fehs[ii,:]= numpy.linspace(fehmin,fehmax,_NFEH)
    for ii in range(_NFEH):
        grs[:,ii]= numpy.linspace(grmin,grmax,_NGR)
    sys.stdout.write('\r'+"Determining minimal and maximal distance")
    sys.stdout.flush()
    dmin= numpy.amin(_ivezic_dist(grs,thisrmin,fehs))
    dmax= numpy.amax(_ivezic_dist(grs,thisrmax,fehs))
    sys.stdout.write('\r'+_ERASESTR+'\r')
    sys.stdout.flush()
    Rmin, Rmax= 5., 14. #BOVY: HARD-CODED
    Rs= numpy.linspace(Rmin,Rmax,_NZS)
    if isinstance(params,list): #list of samples
        pass
    else: #single value
        Rdist= numpy.zeros(_NZS)
        platels, platebs= [], []
        cnt= 0
        for p in plate:
            cnt+= 1
            sys.stdout.write('\r'+"Working on plate %i (%i/%i)" % (p,cnt,len(plate)))
            sys.stdout.flush()
            #l and b?
            pindx= (sf.plates == p)
            platel= platelb[pindx,0][0]
            plateb= platelb[pindx,1][0]
            platels.append(platel)
            platebs.append(plateb)
            thisRdist= _predict_Rdist_plate(Rs,densfunc,params,rmin,rmax,
                                            platel,
                                            plateb,grmin,grmax,
                                            fehmin,fehmax,
                                            feh,colordist,fehdist,sf,p)
            Rdist+= thisRdist
        sys.stdout.write('\r'+_ERASESTR+'\r')
        sys.stdout.flush()
        norm= numpy.nansum(Rdist*(Rs[1]-Rs[0]))
        Rdist/= norm
        if convolve > 0.:
            ndimage.filters.gaussian_filter1d(Rdist,convolve/(Rs[1]-Rs[0]),
                                              output=Rdist)
        else:
            #Convolve around 8 anyway
            eightindx= (Rs > 7.8)*(Rs < 8.2)
            eightRdist= numpy.zeros(numpy.sum(eightindx))
            ndimage.filters.gaussian_filter1d(Rdist[eightindx],
                                              0.1/(Rs[1]-Rs[0]),
                                              output=eightRdist,
                                              mode='nearest')
            Rdist[eightindx]= eightRdist
        if xrange is None:
            if allbright and feh <= -0.5:
                xrange= [numpy.amin(Rs)-0.2,numpy.amax(Rs)+0.7]
            else:
                xrange= [numpy.amin(Rs)-0.2,numpy.amax(Rs)+0.3]
        if yrange is None:
            yrange= [0.,1.6*numpy.amax(Rdist)]
        bovy_plot.bovy_plot(Rs,Rdist,ls=ls,color=color,
                            xrange=xrange,yrange=yrange,
                            xlabel='$R\ [\mathrm{kpc}]$',
                            ylabel='$\mathrm{density}$',overplot=overplot)
        #Plot the data
        data_R= []
        for p in plate:
            data_R.extend(((8.-data[(data.plate == p)].xc)**2.+data[(data.plate == p)].yc**2.)**0.5)
        hist= bovy_plot.bovy_hist(data_R,
                                  normed=True,bins=bins,ec='k',
                                  histtype='step',
                                  overplot=True,range=[Rmin,Rmax])
        if not right_legend is None:
            bovy_plot.bovy_text(right_legend,top_right=True,size=_legendsize)
        if len(plate) > 1 and len(plate) < 9:
            platestr= '\mathrm{plates}\ \ '
            for ii in range(len(plate)-1):
                platestr= platestr+'%i, ' % plate[ii]
            platestr+= '%i' % plate[-1]
            lbstr= '$l = %i^\circ \pm %i^\circ$' % (int(numpy.mean(platels)),int(numpy.std(platels)))+'\n'\
                +'$b = %i^\circ \pm %i^\circ$' % (int(numpy.mean(platebs)),
            int(numpy.std(platebs)))
        elif allplates:
            platestr= '\mathrm{all\ plates}'
            if right_legend is None:
                bovy_plot.bovy_text(r'$'+platestr+'$'
                                    +'\n'+
                                    '$%i \ \ \mathrm{stars}$' % 
                                    len(data_R),top_right=True,
                                    size=_legendsize)
        elif brightplates:
            platestr= '\mathrm{bright\ plates}'
            if right_legend is None:
                bovy_plot.bovy_text(r'$'+platestr+'$'
                                    +'\n'+
                                    '$%i \ \ \mathrm{stars}$' % 
                                    len(data_R),top_right=True,
                                    size=_legendsize)
        elif faintplates:
            platestr= '\mathrm{faint\ plates}'
            if right_legend is None:
                bovy_plot.bovy_text(r'$'+platestr+'$'
                                    +'\n'+
                                    '$%i \ \ \mathrm{stars}$' % 
                                    len(data_R),top_right=True,
                                    size=_legendsize)
        elif len(plate) >= 9:
            platestr= '%i\ \mathrm{plates}' % len(plate)
            lbstr= '$l = %i^\circ \pm %i^\circ$' % (
                int(numpy.mean(platels)),int(numpy.std(platels)))+'\n'\
                +'$b = %i^\circ\pm%i^\circ$' % (int(numpy.mean(platebs)),
            int(numpy.std(platebs)))
        else:
            platestr= '\mathrm{plate}\ \ %i' % plate[0]
            lbstr= '$l = %i^\circ$' % int(platel)+'\n'\
                +'$b = %i^\circ$' % int(plateb)           
        if not (allplates or brightplates or faintplates):
            if right_legend is None:
                bovy_plot.bovy_text(r'$'+platestr+'$'
                                    +'\n'+
                                    '$%i \ \ \mathrm{stars}$' % 
                                    len(data_R)
                                    +'\n'+
                                    lbstr,top_right=True,
                                    size=_legendsize)
            #Overplot direction in (R,z) plane
            ax= matplotlib.pyplot.gca()
            yrange= ax.get_ylim()
            dy= yrange[1]-yrange[0]
            xfac= 1./(20.8-14.5)*(xrange[1]-xrange[0])
            rx, ry,dr, dz= xrange[1]-2.1*xfac,yrange[1]-0.5*dy, 2.*xfac, 0.4*dy
            #x-axis
            bovy_plot.bovy_plot([rx-0.2*xfac,rx-0.2*xfac+dr],
                                [ry,ry],
                                'k-',overplot=True)
            #y-axis
            bovy_plot.bovy_plot([rx,rx],
                                [ry-dz/2.,ry+dz/2.],
                                'k-',overplot=True)
            #Sun's position
            bovy_plot.bovy_plot([rx+dr/2.],[ry],'ko',overplot=True)
            #Draw los
            gr= (grmax+grmin)/2.
            if allbright:
                thisrmin= rmin
                thisrmax= 17.8
            if allfaint:
                thisrmin= 17.8
                thisrmax= rmax
            dmin, dmax= _ivezic_dist(gr,thisrmin,feh), _ivezic_dist(gr,thisrmax,feh)
            ds= numpy.linspace(dmin,dmax,101)
            xyzs= bovy_coords.lbd_to_XYZ(numpy.array([numpy.mean(platels)+numpy.std(platels) for ii in range(len(ds))]),
                                         numpy.array([numpy.mean(platebs) for ii in range(len(ds))]),
                                         ds,degree=True).astype('float')
            rs= (((8.-xyzs[:,0])**2.+xyzs[:,1]**2.)**0.5)/8.*dr/2.+rx
            zs= xyzs[:,2]/8.*dz/2.+ry
            bovy_plot.bovy_plot(rs,zs,'-',color='0.75',overplot=True)
            xyzs= bovy_coords.lbd_to_XYZ(numpy.array([numpy.mean(platels)-numpy.std(platels) for ii in range(len(ds))]),
                                         numpy.array([numpy.mean(platebs) for ii in range(len(ds))]),
                                         ds,degree=True).astype('float')
            rs= (((8.-xyzs[:,0])**2.+xyzs[:,1]**2.)**0.5)/8.*dr/2.+rx
            zs= xyzs[:,2]/8.*dz/2.+ry
            bovy_plot.bovy_plot(rs,zs,'-',color='0.75',overplot=True)
            xyzs= bovy_coords.lbd_to_XYZ(numpy.array([numpy.mean(platels) for ii in range(len(ds))]),
                                         numpy.array([numpy.mean(platebs)+numpy.std(platebs) for ii in range(len(ds))]),
                                         ds,degree=True).astype('float')
            rs= (((8.-xyzs[:,0])**2.+xyzs[:,1]**2.)**0.5)/8.*dr/2.+rx
            zs= xyzs[:,2]/8.*dz/2.+ry
            bovy_plot.bovy_plot(rs,zs,'-',color='0.75',overplot=True)
            xyzs= bovy_coords.lbd_to_XYZ(numpy.array([numpy.mean(platels) for ii in range(len(ds))]),
                                         numpy.array([numpy.mean(platebs)-numpy.std(platebs) for ii in range(len(ds))]),
                                         ds,degree=True).astype('float')
            rs= (((8.-xyzs[:,0])**2.+xyzs[:,1]**2.)**0.5)/8.*dr/2.+rx
            zs= xyzs[:,2]/8.*dz/2.+ry
            bovy_plot.bovy_plot(rs,zs,'-',color='0.75',overplot=True)
            xyzs= bovy_coords.lbd_to_XYZ(numpy.array([numpy.mean(platels) for ii in range(len(ds))]),
                                         numpy.array([numpy.mean(platebs) for ii in range(len(ds))]),
                                         ds,degree=True).astype('float')
            rs= (((8.-xyzs[:,0])**2.+xyzs[:,1]**2.)**0.5)/8.*dr/2.+rx
            zs= xyzs[:,2]/8.*dz/2.+ry
            bovy_plot.bovy_plot(rs,zs,'k-',overplot=True)
            bovy_plot.bovy_text(rx+3./4.*dr,ry-0.1*dz,r'$R$')
            bovy_plot.bovy_text(rx-0.25*xfac,ry+3./4.*dz/2.,r'$Z$')
        if not left_legend is None:
            bovy_plot.bovy_text(left_legend,top_left=True,size=_legendsize)
        return (Rdist, hist[0], hist[1])

def comparernumberPlate(densfunc,params,sf,colordist,fehdist,data,plate,
                        rmin=14.5,rmax=20.2,grmin=0.48,grmax=0.55,
                        fehmin=-0.4,
                        fehmax=0.5,feh=-0.15,
                        zmax=None,
                        vsx='|sinb|',
                        xrange=None,yrange=None,
                        overplot=False,color='k',marker='v',cumul=False,
                        runavg=0,noplot=False,nodata=False,distfac=1.,
                        R0=8.,numcores=None,
                        colorfehfac=None,normR=None,normZ=None):
    """
    NAME:
       comparernumberPlate
    PURPOSE:
       compare the observed number of objects per plate to the model prediction
    INPUT:
       densfunc - underlying density function densfunc(R,z,params)
       params - array of parameters of the density function
       sf - segueSelect instance
       colordist - color g-r distribution
       data - data recarray (.dered_r is used)
       plate - plate numbers, or 'all', 'faint', or 'bright'
       rmin, rmax= minimum and maximum r
       grmin, grmax= minimum and maximum g-r
       feh= metallicity to use in d->r
       vsx= what to plot as x ('b', 'l', 'plate', '|b|', '|sinb|')
       xrange=, yrange= ranges for plot
       overplot= if True, overplot
       color= color for model
       marker= marker
       cumul= if True, plot cumulative distribution
       runavg= if > 0, also plot a running average (only for cumul=False)
       numcores= if set to an integer, use this number of cores
       colorfehfac= the summation of color and FeH for each plate [plates,nzs]
       normR=, normZ= radii and heights for the integral
    OUTPUT:
       plot to output
       return numbers, data_numbers, xs
    HISTORY:
       2011-07-18 - Written - Bovy (NYU)
    """
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
    #Zmin and Zmax for this rmin, rmax
    bs= []
    allbright, allfaint= False, False
    for p in plate:
        #l and b?
        pindx= (sf.plates == p)
        plateb= platelb[pindx,1][0]
        bs.append(plateb)
        if 'faint' in sf.platestr[pindx].programname[0]:
            allbright= False
        else:
            allfaint= False
    bs= numpy.array(bs)
    bmin, bmax= numpy.amin(numpy.fabs(bs)),numpy.amax(numpy.fabs(bs))
    if allbright:
        thisrmin, thisrmax= rmin, 17.8
    elif allfaint:
        thisrmin, thisrmax= 17.8, rmax
    else:
        thisrmin, thisrmax= rmin, rmax
    _NGR, _NFEH= 51, 51
    grs= numpy.zeros((_NGR,_NFEH))
    fehs= numpy.zeros((_NGR,_NFEH))
    for ii in range(_NGR):
        if feh > -0.5: #rich, actually only starts at 0.05
            fehs[ii,:]= numpy.linspace(fehmin,fehmax,_NFEH)
        else:
            fehs[ii,:]= numpy.linspace(fehmin,fehmax,_NFEH)
    for ii in range(_NFEH):
        grs[:,ii]= numpy.linspace(grmin,grmax,_NGR)
    dmin= numpy.amin(_ivezic_dist(grs,thisrmin,fehs))*distfac
    dmax= numpy.amax(_ivezic_dist(grs,thisrmax,fehs))*distfac
    if zmax is None:
        zmax= dmax*numpy.sin(bmax*_DEGTORAD)
#        zmax+= 2.*_ZSUN #Just to be sure we have the North covered
    zmin= dmin*numpy.sin(bmin*_DEGTORAD)
    zmin-= 2.*_ZSUN #Just to be sure we have the South covered
    zs= numpy.linspace(zmin,zmax,_NZS)
    if not noplot:
        #Set up x
        if vsx.lower() == 'b' or vsx.lower() == 'l' or vsx.lower() == '|b|' \
                or vsx.lower() == '|sinb|':
            platels, platebs= [], []
            for ii in range(len(plate)):
                p= plate[ii]
                #l and b?
                pindx= (sf.plates == p)
                platel= platelb[pindx,0][0]
                plateb= platelb[pindx,1][0]
                platels.append(platel)
                platebs.append(plateb)
            if vsx.lower() == 'b':
                xs= numpy.array(platebs)
                xlabel= r'$b\ [\mathrm{deg}]$'
            elif vsx.lower() == '|b|':
                xs= numpy.fabs(numpy.array(platebs))
                xlabel= r'$|b|\ [\mathrm{deg}]$'
            elif vsx.lower() == '|sinb|':
                xs= numpy.fabs(numpy.sin(numpy.array(platebs)))
                xlabel= r'$\sin |b|$'
            else:
                xs= numpy.array(platels)
                xlabel= r'$l\ [\mathrm{deg}]$'
            addx= 0.1
        elif vsx.lower() == 'plate':
            xs= numpy.array(plate)
            addx= 100
            xlabel= r'$\mathrm{plate}$'
    if isinstance(params,list): #list of samples
        pass
    else: #single value
        numbers= numpy.zeros(len(plate))
        platels, platebs= [], []
        if not numcores is None:
            numbers= numpy.array(multi.parallel_map((lambda x: _calc_numbers(plate[x],
                                                                 platelb,
                                                                 zs,
                                                                 densfunc,
                                                                 params,
                                                                 rmin,rmax,
                                                                 grmin,grmax,
                                                                 fehmin,fehmax,
                                                                 feh,colordist,
                                                                 fehdist,sf,
                                                                 distfac,R0,
                                                                             colorfehfac,normR,normZ)),
                                        range(len(plate)),numcores=numcores))
        else:
            for ii in range(len(plate)):
                p= plate[ii]
                #l and b?
                pindx= (sf.plates == p)
                platel= platelb[pindx,0][0]
                plateb= platelb[pindx,1][0]
                platels.append(platel)
                platebs.append(plateb)
                thiszdist= _predict_zdist_plate(zs,densfunc,params,rmin,rmax,
                                                platel,
                                                plateb,grmin,grmax,
                                                fehmin,fehmax,
                                                feh,colordist,fehdist,sf,p,distfac,
                                                R0)
                numbers[ii]= numpy.nansum(thiszdist)
#        norm= numpy.nansum(numbers)
#        numbers/= norm
        numbers*= (zs[1]-zs[0])/(20.2-14.5)*1000 #backward compatibility
        if noplot and nodata:
            return numbers
        if xrange is None:
            xrange= [numpy.amin(xs)-addx,numpy.amax(xs)+addx]
        if yrange is None and not cumul:
            yrange= [0.,1.2*numpy.amax(numbers)]
        if yrange is None and cumul:
            yrange=[0.,1.1]
        #The data
        data_numbers= []
        for p in plate:
            data_numbers.append(numpy.sum((data.plate == p)))
        data_numbers= numpy.array(data_numbers,dtype='float')
        nstars= numpy.sum(data_numbers)
        data_numbers/= nstars
        if noplot:
            return (numbers, data_numbers, xs)
        #Sort the data and note the sort index
        sortindx= numpy.argsort(data_numbers)
        data_numbers= data_numbers[sortindx]
        numbers= numbers[sortindx]
        if cumul:
            data_numbers= numpy.cumsum(data_numbers)
            numbers= numpy.cumsum(numbers)
        xs= numpy.arange(len(plate))
        bovy_plot.bovy_plot(xs,numbers,marker=marker,color=color,ls='none',
                            yrange=yrange,
                            xlabel=r'$\mathrm{plates\ sorted\ by\ number}$',
                            ylabel='$\mathrm{relative\ number}$',
                            overplot=overplot)
        if runavg > 0 and not cumul:
            from matplotlib import mlab
            running_avg= mlab.movavg(numbers,runavg)
            runavg_xs= numpy.arange(len(running_avg))
            runavg_xs+= (len(numbers)-len(running_avg))/2
            bovy_plot.bovy_plot(runavg_xs,running_avg,color=color,ls='-',
                                overplot=True)
        bovy_plot.bovy_plot(xs,data_numbers,ls='-',
                            marker='.',color='k',
                            overplot=True)
        if len(plate) > 1 and len(plate) < 9:
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
                                nstars,top_left=True,
                                size=_legendsize)
        elif brightplates:
            platestr= '\mathrm{bright\ plates}'
            bovy_plot.bovy_text(r'$'+platestr+'$'
                                +'\n'+
                                '$%i \ \ \mathrm{stars}$' % 
                                nstars,top_left=True,
                                size=_legendsize)
        elif faintplates:
            platestr= '\mathrm{faint\ plates}'
            bovy_plot.bovy_text(r'$'+platestr+'$'
                                +'\n'+
                                '$%i \ \ \mathrm{stars}$' % 
                                nstars,top_left=True,
                                size=_legendsize)
        elif len(plate) >= 9:
            platestr= '%i\ \mathrm{plates}' % len(plate)
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
                                nstars
                                +'\n'+
                                lbstr,top_left=True,
                                size=_legendsize)
        return (numbers, data_numbers, xs)

#For multi evaluation of previous    
def _calc_numbers(p,platelb,zs,densfunc,params,rmin,rmax,
                  grmin,grmax,fehmin,fehmax,
                  feh,colordist,fehdist,sf,distfac,R0,colorfehfac,R,Z):
    #l and b?
    pindx= (sf.plates == p)
    platel= platelb[pindx,0][0]
    plateb= platelb[pindx,1][0]
    thiszdist= _predict_zdist_plate(zs,densfunc,params,rmin,rmax,
                                    platel,
                                    plateb,grmin,grmax,
                                    fehmin,fehmax,
                                    feh,colordist,fehdist,sf,p,distfac,
                                    R0,colorfehfac[pindx,:],
                                    R[pindx,:],Z[pindx,:])
    return numpy.nansum(thiszdist)

def comparezdistPlateMulti(densfunc,params,sf,colordist,fehdist,data,plate,
                           rmin=14.5,rmax=20.2,grmin=0.48,grmax=0.55,
                           fehmin=-0.4,fehmax=0.5,feh=-0.15,
                           convolve=0.,xrange=None,yrange=None,
                           overplot=False,bins=21,color='k',ls='-',
                           left_legend=None,right_legend=None):
    """
    NAME:
       comparezdistPlateMulti
    PURPOSE:
       compare the observed distribution of heights zto the model prediction (for M multiple populations
    INPUT:
       densfunc - underlying density function densfunc(R,z,params) LIST OF M
       params - array of parameters of the density function LIST OF M
       sf - segueSelect instance
       colordist - color g-r distribution LIST OF M
       fehdist - Fe/H distribution of the sample LIST OF M
       data - data recarray (.dered_r is used) LIST OF M
       plate - plate number(s), or 'all', 'faint', or 'bright'
       rmin, rmax= minimum and maximum r
       grmin, grmax= minimum and maximum g-r
       fehmin, fehmax= minimum and maximum [Fe/H] LIST OF M
       feh= metallicity to use in d->r LIST OF M
       convolve= set to convolution kernel size (convolve predicted 
                 distribution with Gaussian of this width)
       xrange=, yrange= ranges for plot
       overplot= if True, overplot
       bins= hist number of bins
       color= color for model
       left_legend = if set, legend to put at the left top
       right_legend = if set, legend to put at the right top
    OUTPUT:
       plot to output
       return rdist, datahist, dataedges
    HISTORY:
       2011-07-29 - Written - Bovy (NYU)
       2012-12-21 - Adapted for multiple populations - Bovy (IAS)
    """
    M= len(densfunc)
    #Set up plates
    platelb= bovy_coords.radec_to_lb(sf.platestr.ra,sf.platestr.dec,
                                     degree=True)
    platelb= platelb.astype('float')
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
    allfaint, allbright= True, True
    #Zmin and Zmax for this rmin, rmax
    bs= []
    for p in plate:
        #l and b?
        pindx= (sf.plates == p)
        plateb= platelb[pindx,1][0]
        bs.append(plateb)
        if 'faint' in sf.platestr[pindx].programname[0]:
            allbright= False
        else:
            allfaint= False
    bs= numpy.array(bs)
    bmin, bmax= numpy.amin(numpy.fabs(bs)),numpy.amax(numpy.fabs(bs))
    if allbright:
        thisrmin, thisrmax= rmin, 17.8
    elif allfaint:
        thisrmin, thisrmax= 17.8, rmax
    else:
        thisrmin, thisrmax= rmin, rmax
    _NGR, _NFEH= 51, 51
    grs= numpy.zeros((M,_NGR,_NFEH))
    fehs= numpy.zeros((M,_NGR,_NFEH))
    for jj in range(M):
        for ii in range(_NGR):
            if feh[jj] > -0.5: #rich, actually only starts at 0.05
                fehs[jj,ii,:]= numpy.linspace(fehmin[jj],0.05,_NFEH)
            else:
                fehs[jj,ii,:]= numpy.linspace(fehmin[jj],fehmax[jj],_NFEH)
        for ii in range(_NFEH):
            grs[jj,:,ii]= numpy.linspace(grmin,grmax,_NGR)
    sys.stdout.write('\r'+"Determining minimal and maximal distance")
    sys.stdout.flush()
    dmin= numpy.array([numpy.amin(_ivezic_dist(grs[jj,:,:],thisrmin,fehs[jj,:,:])) for jj in range(M)])
    dmax= numpy.array([numpy.amax(_ivezic_dist(grs[jj,:,:],thisrmax,fehs[jj,:,:])) for jj in range(M)])
    sys.stdout.write('\r'+_ERASESTR+'\r')
    sys.stdout.flush()
    zmin, zmax= dmin*numpy.sin(bmin*_DEGTORAD), dmax*numpy.sin(bmax*_DEGTORAD)
    zs= numpy.linspace(numpy.amin(zmin),numpy.amax(zmax),_NZS)
    if False: #remnant of old code
        pass
    else: #single value
        zdist= numpy.zeros((M,_NZS))
        platels, platebs= [], []
        for jj in range(M):
            cnt= 0
            for p in plate:
                cnt+= 1
                sys.stdout.write('\r'+"Working on plate %i (%i/%i)" % (p,cnt,len(plate)))
                sys.stdout.flush()
                #l and b?
                pindx= (sf.plates == p)
                platel= platelb[pindx,0][0]
                plateb= platelb[pindx,1][0]
                if jj == 0: #only do this once
                    platels.append(platel)
                    platebs.append(plateb)
                thiszdist= _predict_zdist_plate(zs,
                                                densfunc[jj],params[jj],
                                                rmin,rmax,
                                                platel,
                                                plateb,grmin,grmax,
                                                fehmin[jj],fehmax[jj],
                                                feh[jj],colordist[jj],
                                                fehdist[jj],sf,p)
                zdist[jj,:]+= thiszdist
        sys.stdout.write('\r'+_ERASESTR+'\r')
        sys.stdout.flush()
        #Fix relative normalization
        prednumbers= numpy.nansum(zdist,axis=1)
        obsnumbers= numpy.array([len(data[jj]) for jj in range(M)])
        relamp= obsnumbers/prednumbers
        for jj in range(M):
            zdist[jj:]*= relamp[jj]
        zdist= numpy.nansum(zdist,axis=0)
        norm= numpy.nansum(zdist*(zs[1]-zs[0]))
        zdist/= norm
        if convolve > 0.:
            ndimage.filters.gaussian_filter1d(zdist,convolve/(zs[1]-zs[0]),
                                              output=zdist)
        if xrange is None:
            if allplates or brightplates:
                xrange=[-0.2,5]
            else:
                if feh[0] > -0.5:
                    xrange= [numpy.amin(zs)-0.2,
                             0.7*(numpy.amax(zs)-numpy.amin(zs)+0.5)+numpy.amin(zs)-0.2]
                else:
                    xrange= [numpy.amin(zs)-0.2,
                             1*(numpy.amax(zs)-numpy.amin(zs)+0.5)+numpy.amin(zs)-0.2]
        if yrange is None:
            yrange= [0.,1.65*numpy.amax(zdist)]
        bovy_plot.bovy_plot(zs,zdist,ls=ls,color=color,
                            xrange=xrange,yrange=yrange,
                            xlabel='$|Z|\ [\mathrm{kpc}]$',
                            ylabel='$\mathrm{density}$',overplot=overplot)
        #Plot the data
        data_z= []
        for jj in range(M):
            for p in plate:
                data_z.extend(numpy.fabs(data[jj][(data[jj].plate == p)].zc))
        if len(data_z) > 0:
            hist= bovy_plot.bovy_hist(data_z,
                                      normed=True,bins=bins,ec='k',
                                      histtype='step',
                                      overplot=True,
                                      range=[numpy.amin(zmin),numpy.amax(zmax)])
        if not right_legend is None:
            bovy_plot.bovy_text(right_legend,top_right=True,size=_legendsize)
        if len(plate) > 1 and len(plate) < 9:
            platestr= '\mathrm{plates}\ \ '
            for ii in range(len(plate)-1):
                platestr= platestr+'%i, ' % plate[ii]
            platestr+= '%i' % plate[-1]
            lbstr= '$l = %i^\circ \pm %i^\circ$' % (int(numpy.mean(platels)),int(numpy.std(platels)))+'\n'\
                +'$b = %i^\circ \pm %i^\circ$' % (int(numpy.mean(platebs)),
            int(numpy.std(platebs)))
        elif allplates:
            platestr= '\mathrm{all\ plates}'
            if right_legend is None:
                bovy_plot.bovy_text(r'$'+platestr+'$'
                                    +'\n'+
                                    '$%i \ \ \mathrm{stars}$' % 
                                    len(data_z),top_right=True,
                                    size=_legendsize)
        elif brightplates:
            platestr= '\mathrm{bright\ plates}'
            if right_legend is None:
                bovy_plot.bovy_text(r'$'+platestr+'$'
                                    +'\n'+
                                    '$%i \ \ \mathrm{stars}$' % 
                                    len(data_z),top_right=True,
                                    size=_legendsize)
        elif faintplates:
            platestr= '\mathrm{faint\ plates}'
            if right_legend is None:
                bovy_plot.bovy_text(r'$'+platestr+'$'
                                    +'\n'+
                                    '$%i \ \ \mathrm{stars}$' % 
                                    len(data_z),top_right=True,
                                    size=_legendsize)
        elif len(plate) >= 9:
            platestr= '%i\ \mathrm{plates}' % len(plate)
            lbstr= '$l = %i^\circ \pm %i^\circ$' % (
                int(numpy.mean(platels)),int(numpy.std(platels)))+'\n'\
                +'$b = %i^\circ\pm%i^\circ$' % (int(numpy.mean(platebs)),
            int(numpy.std(platebs)))
        else:
            platestr= '\mathrm{plate}\ \ %i' % plate[0]
            lbstr= '$l = %i^\circ$' % int(platel)+'\n'\
                +'$b = %i^\circ$' % int(plateb)           
        if not (allplates or brightplates or faintplates):
            if right_legend is None:
                bovy_plot.bovy_text(r'$'+platestr+'$'
                                    +'\n'+
                                    '$%i \ \ \mathrm{stars}$' % 
                                    len(data_z)
                                    +'\n'+
                                    lbstr,top_right=True,
                                    size=_legendsize)
            #Overplot direction in (R,z) plane
            ax= matplotlib.pyplot.gca()
            yrange= ax.get_ylim()
            dy= yrange[1]-yrange[0]
            xfac= 1./(20.8-14.5)*(xrange[1]-xrange[0])
            rx, ry,dr, dz= xrange[1]-2.1*xfac,yrange[1]-0.5*dy, 2.*xfac, 0.4*dy
            #x-axis
            bovy_plot.bovy_plot([rx-0.2*xfac,rx-0.2*xfac+dr],
                                [ry,ry],
                                'k-',overplot=True)
            #y-axis
            bovy_plot.bovy_plot([rx,rx],
                                [ry-dz/2.,ry+dz/2.],
                                'k-',overplot=True)
            #Sun's position
            bovy_plot.bovy_plot([rx+dr/2.],[ry],'ko',overplot=True)
            #Draw los
            gr= (grmax+grmin)/2.
            if allbright:
                thisrmin= rmin
                thisrmax= 17.8
            if allfaint:
                thisrmin= 17.8
                thisrmax= rmax
            dmin, dmax= _ivezic_dist(gr,thisrmin,numpy.mean(numpy.array(feh))), _ivezic_dist(gr,thisrmax,numpy.mean(numpy.array(feh)))
            ds= numpy.linspace(dmin,dmax,101)
            xyzs= bovy_coords.lbd_to_XYZ(numpy.array([numpy.mean(platels)+numpy.std(platels) for ii in range(len(ds))]),
                                         numpy.array([numpy.mean(platebs) for ii in range(len(ds))]),
                                         ds,degree=True).astype('float')
            rs= (((8.-xyzs[:,0])**2.+xyzs[:,1]**2.)**0.5)/8.*dr/2.+rx
            zs= xyzs[:,2]/8.*dz/2.+ry
            bovy_plot.bovy_plot(rs,zs,'-',color='0.75',overplot=True)
            xyzs= bovy_coords.lbd_to_XYZ(numpy.array([numpy.mean(platels)-numpy.std(platels) for ii in range(len(ds))]),
                                         numpy.array([numpy.mean(platebs) for ii in range(len(ds))]),
                                         ds,degree=True).astype('float')
            rs= (((8.-xyzs[:,0])**2.+xyzs[:,1]**2.)**0.5)/8.*dr/2.+rx
            zs= xyzs[:,2]/8.*dz/2.+ry
            bovy_plot.bovy_plot(rs,zs,'-',color='0.75',overplot=True)
            xyzs= bovy_coords.lbd_to_XYZ(numpy.array([numpy.mean(platels) for ii in range(len(ds))]),
                                         numpy.array([numpy.mean(platebs)+numpy.std(platebs) for ii in range(len(ds))]),
                                         ds,degree=True).astype('float')
            rs= (((8.-xyzs[:,0])**2.+xyzs[:,1]**2.)**0.5)/8.*dr/2.+rx
            zs= xyzs[:,2]/8.*dz/2.+ry
            bovy_plot.bovy_plot(rs,zs,'-',color='0.75',overplot=True)
            xyzs= bovy_coords.lbd_to_XYZ(numpy.array([numpy.mean(platels) for ii in range(len(ds))]),
                                         numpy.array([numpy.mean(platebs)-numpy.std(platebs) for ii in range(len(ds))]),
                                         ds,degree=True).astype('float')
            rs= (((8.-xyzs[:,0])**2.+xyzs[:,1]**2.)**0.5)/8.*dr/2.+rx
            zs= xyzs[:,2]/8.*dz/2.+ry
            bovy_plot.bovy_plot(rs,zs,'-',color='0.75',overplot=True)
            xyzs= bovy_coords.lbd_to_XYZ(numpy.array([numpy.mean(platels) for ii in range(len(ds))]),
                                         numpy.array([numpy.mean(platebs) for ii in range(len(ds))]),
                                         ds,degree=True).astype('float')
            rs= (((8.-xyzs[:,0])**2.+xyzs[:,1]**2.)**0.5)/8.*dr/2.+rx
            zs= xyzs[:,2]/8.*dz/2.+ry
            bovy_plot.bovy_plot(rs,zs,'k-',overplot=True)
            bovy_plot.bovy_text(rx+3./4.*dr,ry-0.1*dz,r'$R$')
            bovy_plot.bovy_text(rx-0.25*xfac,ry+3./4.*dz/2.,r'$Z$')
        if not left_legend is None:
            bovy_plot.bovy_text(left_legend,top_left=True,size=_legendsize)
        if len(data_z) > 0:
            return (zdist, hist[0], hist[1])
        else:
            return (zdist, None,None)

###############################################################################
#   Good sets of plates to run comparerdistPlate for
###############################################################################
def similarPlatesDirection(l,b,dr,sf,data=None,bright=True,faint=True):
    """
    NAME:
       similarPlatesDirection
    PURPOSE:
       Find good sets of plates to run comparerdistPlate for
    INPUT:
       l,b - desired (l,b) center, but only start of iteration
       dr - radius of circle to consider plates in (deg)
       sf - segueSelect instance
       bright=, faint= if false, don't include bright/faint plates 
                       (default: all)
    OUTPUT:
      list of plates that can be sent to comparerdistPlate
    HISTORY:
       2011-07-25 - Written - Bovy (NYU)
    """
    lrad= l*_DEGTORAD
    brad= b*_DEGTORAD
    racen, deccen= bovy_coords.lb_to_radec(lrad,brad)
    drrad= dr*_DEGTORAD
    cosdr= numpy.cos(drrad)
    if sf is None:
        #Just load plates from file
        platestr= _load_fits(os.path.join(_SEGUESELECTDIR,
                                          'segueplates.fits'))
        plates= platestr.plate
    elif bright and faint:
        plates= sf.plates
        platestr= sf.platestr
    elif bright:
        plates= sf.plates[sf.brightplateindx]
        platestr= sf.platestr[sf.brightplateindx]
    elif faint:
        plates= sf.plates[sf.faintplateindx]
        platestr= sf.platestr[sf.faintplateindx]
    iterating= True
    finalplates= []
    while iterating:
        #Calculate distance between current center and all plates
        cosdist= [cos_sphere_dist(platestr[ii].ra*_DEGTORAD,
                                  platestr[ii].dec*_DEGTORAD,
                                  racen,deccen) for ii in range(len(plates))]
        indx= (cosdist >= cosdr)
        theseplates= plates[indx]
        if sorted(theseplates) == sorted(finalplates): iterating= False
        finalplates= theseplates
        racen= numpy.mean(platestr[indx].ra)*_DEGTORAD
        deccen= numpy.mean(platestr[indx].dec)*_DEGTORAD
    return finalplates
    

###############################################################################
#            FORWARD MODELING FOR MODEL--DATA COMPARISON
###############################################################################
def _predict_rdist(rs,densfunc,params,rmin,rmax,platelb,grmin,grmax,
                   feh,sf,colordist):
    """Predict the r distribution for the sample NOT USED AND NOT CORRECT"""
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
                         fehmin,fehmax,
                         feh,colordist,fehdist,
                         sf,plate,dontmarginalizecolorfeh=False,
                         ngr=11,nfeh=11,
                         dontmultiplycolorfehdist=False):
    """Predict the r distribution for a plate"""
    #BOVY: APPROXIMATELY INTEGRATE OVER GR AND FEH
    grs= numpy.linspace(grmin,grmax,ngr)
    fehs= numpy.linspace(fehmin,fehmax,nfeh)
    if dontmarginalizecolorfeh:
        out= numpy.zeros((len(rs),ngr,nfeh))
    else:
        out= numpy.zeros(len(rs))
    norm= 0.
    for kk in range(nfeh):
        for jj in range(ngr):
            #Calculate distances
            ds= _ivezic_dist(grs[jj],rs,fehs[kk])
            #Calculate (R,z)s
            XYZ= bovy_coords.lbd_to_XYZ(numpy.array([l for ii in range(len(ds))]),
                                        numpy.array([b for ii in range(len(ds))]),
                                        ds,degree=True)
            XYZ= XYZ.astype(numpy.float64)
            R= ((8.-XYZ[:,0])**2.+XYZ[:,1]**2.)**(0.5)
            XYZ[:,2]+= _ZSUN
            if dontmarginalizecolorfeh:
                out[:,jj,kk]= ds**3.*densfunc(R,XYZ[:,2],params)*colordist(grs[jj])*fehdist(fehs[kk])
            elif dontmultiplycolorfehdist:
                out+= ds**3.*densfunc(R,XYZ[:,2],params)
                norm+= 1.
            else:
                out+= ds**3.*densfunc(R,XYZ[:,2],params)*colordist(grs[jj])\
                    *fehdist(fehs[kk])
                norm+= colordist(grs[jj])*fehdist(fehs[kk])
    select= sf(plate,r=rs)
    if dontmarginalizecolorfeh:
        for jj in range(ngr):
            for kk in range(nfeh):
                out[:,jj,kk]*= select
    else:
        out*= select
        out/= norm
        out[(rs < rmin)]= 0.
        out[(rs > rmax)]= 0.
    return out

def _predict_zdist_plate(zs,densfunc,params,rmin,rmax,l,b,grmin,grmax,
                         fehmin,fehmax,
                         feh,colordist,fehdist,sf,plate,distfac=1.,R0=8.,
                         colorfehfac=None,R=None,Z=None):
    """Predict the Z distribution for a plate"""
    if b > 0.:
        ds= (zs-_ZSUN)/numpy.fabs(numpy.sin(b*_DEGTORAD))
    else:
        ds= (zs+_ZSUN)/numpy.fabs(numpy.sin(b*_DEGTORAD))
    if colorfehfac is None:
        #BOVY: APPROXIMATELY INTEGRATE OVER GR
        ngr, nfeh= 11, 11
        grs= numpy.linspace(grmin,grmax,ngr)
        fehs= numpy.linspace(fehmin,fehmax,nfeh)
        out= numpy.zeros(len(zs))
        norm= 0.
        logds= 5.*numpy.log10(ds/distfac)+10.
        for kk in range(nfeh):
            for jj in range(ngr):
            #What rs do these zs correspond to
                gi= _gi_gr(grs[jj])
                mr= _mr_gi(gi,fehs[kk])
                rs= logds+mr
                select= numpy.array(sf(plate,r=rs))
                out+= colordist(grs[jj])*fehdist(fehs[kk])\
                    *select
                norm+= colordist(grs[jj])*fehdist(fehs[kk])
        out/= norm
    else:
        out= colorfehfac
    #Calculate (R,z)s
    if R is None:
        XYZ= bovy_coords.lbd_to_XYZ(numpy.array([l for ii in range(len(ds))]),
                                    numpy.array([b for ii in range(len(ds))]),
                                    ds,degree=True)
        R= ((R0-XYZ[:,0])**2.+XYZ[:,1]**2.)**(0.5)
        Z= XYZ[:,2]+_ZSUN
    out*= ds**2.*densfunc(R,Z,params)/numpy.fabs(numpy.sin(b*_DEGTORAD))
    return out

def _predict_Rdist_plate(Rs,densfunc,params,rmin,rmax,l,b,grmin,grmax,
                         fehmin,fehmax,
                         feh,colordist,fehdist,sf,plate):
    """Predict the R distribution for a plate"""
    #BOVY: APPROXIMATELY INTEGRATE OVER GR
    ngr, nfeh= 11, 11
    grs= numpy.linspace(grmin,grmax,ngr)
    fehs= numpy.linspace(fehmin,fehmax,nfeh)
    out= numpy.zeros(len(Rs))
    norm= 0.
    ds= numpy.zeros(len(Rs))
    zero_indx= []
    for ii in range(len(Rs)):
        if Rs[ii] < 8.:
            if m.cos(l*_DEGTORAD) < 0.:
                ds[ii]= 1.
                zero_indx.append(True)
                continue
            elif (l < 90. and Rs[ii] < 8.*numpy.cos((90.-l-1.49*0.)*_DEGTORAD)) \
                    or (l > 270. and Rs[ii] < 8.*numpy.cos((l-270.-0.*1.49)*_DEGTORAD)):
                ds[ii]= 1.
                zero_indx.append(True)
                continue
            theta= m.asin(8.*m.sin(l*_DEGTORAD)/Rs[ii])-l*_DEGTORAD
        else:
            theta= m.pi-m.asin(8.*m.sin(l*_DEGTORAD)/Rs[ii])-l*_DEGTORAD
        ds[ii]= m.sqrt(Rs[ii]**2.+8.**2.-16.*Rs[ii]*m.cos(theta))
        zero_indx.append(False)
    zero_indx= numpy.array(zero_indx,dtype='bool')
    ds/= numpy.fabs(numpy.cos(b*_DEGTORAD))
    for kk in range(nfeh):
        for jj in range(ngr):
            #What rs do these Rs correspond to
            gi= _gi_gr(grs[jj])
            mr= _mr_gi(gi,fehs[kk])
            rs= 5.*numpy.log10(ds)+10.+mr
            select= numpy.array(sf(plate,r=rs))
            out+= colordist(grs[jj])*select*fehdist(fehs[kk])
            norm+= colordist(grs[jj])*fehdist(fehs[kk])
    #Calculate (R,z)s
    XYZ= bovy_coords.lbd_to_XYZ(numpy.array([l for ii in range(len(ds))]),
                                numpy.array([b for ii in range(len(ds))]),
                                ds,degree=True)
    XYZ= XYZ.astype(numpy.float64)
    R= ((8.-XYZ[:,0])**2.+XYZ[:,1]**2.)**(0.5)
    #XYZ[:,2]+= _ZSUN #Not here because this is model
    out*= ds**2.*densfunc(R,XYZ[:,2],params)
    out/= norm
    out*= Rs/numpy.cos(b*_DEGTORAD)/numpy.fabs(ds*numpy.cos(b*_DEGTORAD)-8.*numpy.cos(l*_DEGTORAD))
    out[zero_indx]= 0.
    return out

def cos_sphere_dist(theta,phi,theta_o,phi_o):
    """
    NAME:
       cos_sphere_dist
    PURPOSE:
       computes the cosine of the spherical distance between two
       points on the sphere
    INPUT:
       theta  - polar angle [0,pi]
       phi    - azimuth [0,2pi]
       theta  - polar angle of center of the disk
       phi_0  - azimuth of the center of the disk
    OUTPUT:
       cos of spherical distance
    HISTORY:
       2010-04-29 -Written - Bovy (NYU)
    """
    return (numpy.sin(theta)*numpy.sin(theta_o)*(numpy.cos(phi_o)*\
                                                     numpy.cos(phi)+
                                                 numpy.sin(phi_o)\
                                                     *numpy.sin(phi))+
            numpy.cos(theta_o)*numpy.cos(theta))

def _add_coordinset(rx=None,ry=None,rmin=14.5,rmax=19.5,feh=-0.15,
                    allbright=False,allfaint=False,platels=None,
                    platebs=None,grmin=0.48,grmax=0.55):
    #Overplot direction in (R,z) plane
    ax= matplotlib.pyplot.gca()
    xrange= ax.get_xlim()
    yrange= ax.get_ylim()
    dy= yrange[1]-yrange[0]
    xfac= 1./(20.8-14.5)*(xrange[1]-xrange[0])
    if rx is None:
        rx= xrange[1]-2.1*xfac
    if ry is None:
        ry= yrange[1]-0.5*dy
    dr, dz= 2.*xfac, 0.4*dy
    #x-axis
    bovy_plot.bovy_plot([rx-0.2*xfac,rx-0.2*xfac+dr],
                        [ry,ry],
                        'k-',overplot=True)
    #y-axis
    bovy_plot.bovy_plot([rx,rx],
                        [ry-dz/2.,ry+dz/2.],
                        'k-',overplot=True)
    #Sun's position
    bovy_plot.bovy_plot([rx+dr/2.],[ry],'ko',overplot=True)
    #Draw los
    gr= (grmax+grmin)/2.
    if allbright:
        thisrmin= rmin
        thisrmax= 17.8
    elif allfaint:
        thisrmin= 17.8
        thisrmax= rmax
    else:
        thisrmin= rmin
        thisrmax= rmax
    dmin, dmax= _ivezic_dist(gr,thisrmin,feh), _ivezic_dist(gr,thisrmax,feh)
    ds= numpy.linspace(dmin,dmax,101)
    xyzs= bovy_coords.lbd_to_XYZ(numpy.array([numpy.mean(platels)+numpy.std(platels) for ii in range(len(ds))]),
                                 numpy.array([numpy.mean(platebs) for ii in range(len(ds))]),
                                 ds,degree=True).astype('float')
    rs= (((8.-xyzs[:,0])**2.+xyzs[:,1]**2.)**0.5)/8.*dr/2.+rx
    zs= xyzs[:,2]/8.*dz/2.+ry
    bovy_plot.bovy_plot(rs,zs,'-',color='0.75',overplot=True)
    xyzs= bovy_coords.lbd_to_XYZ(numpy.array([numpy.mean(platels)-numpy.std(platels) for ii in range(len(ds))]),
                                 numpy.array([numpy.mean(platebs) for ii in range(len(ds))]),
                                 ds,degree=True).astype('float')
    rs= (((8.-xyzs[:,0])**2.+xyzs[:,1]**2.)**0.5)/8.*dr/2.+rx
    zs= xyzs[:,2]/8.*dz/2.+ry
    bovy_plot.bovy_plot(rs,zs,'-',color='0.75',overplot=True)
    xyzs= bovy_coords.lbd_to_XYZ(numpy.array([numpy.mean(platels) for ii in range(len(ds))]),
                                 numpy.array([numpy.mean(platebs)+numpy.std(platebs) for ii in range(len(ds))]),
                                 ds,degree=True).astype('float')
    rs= (((8.-xyzs[:,0])**2.+xyzs[:,1]**2.)**0.5)/8.*dr/2.+rx
    zs= xyzs[:,2]/8.*dz/2.+ry
    bovy_plot.bovy_plot(rs,zs,'-',color='0.75',overplot=True)
    xyzs= bovy_coords.lbd_to_XYZ(numpy.array([numpy.mean(platels) for ii in range(len(ds))]),
                                 numpy.array([numpy.mean(platebs)-numpy.std(platebs) for ii in range(len(ds))]),
                                 ds,degree=True).astype('float')
    rs= (((8.-xyzs[:,0])**2.+xyzs[:,1]**2.)**0.5)/8.*dr/2.+rx
    zs= xyzs[:,2]/8.*dz/2.+ry
    bovy_plot.bovy_plot(rs,zs,'-',color='0.75',overplot=True)
    xyzs= bovy_coords.lbd_to_XYZ(numpy.array([numpy.mean(platels) for ii in range(len(ds))]),
                                 numpy.array([numpy.mean(platebs) for ii in range(len(ds))]),
                                 ds,degree=True).astype('float')
    rs= (((8.-xyzs[:,0])**2.+xyzs[:,1]**2.)**0.5)/8.*dr/2.+rx
    zs= xyzs[:,2]/8.*dz/2.+ry
    bovy_plot.bovy_plot(rs,zs,'k-',overplot=True)
    bovy_plot.bovy_text(rx+3./4.*dr,ry-0.1*dz,r'$R$')
    bovy_plot.bovy_text(rx-0.28*xfac,ry+3./4.*dz/2.,r'$Z$')
