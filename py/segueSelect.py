import os, os.path
import sys
import copy
import math
import numpy
from scipy import special, interpolate
import pyfits
try:
    from galpy.util import bovy_plot
except ImportError:
    pass #BOVY ADD LOCAL MODULE HERE IF RELEASED
import matplotlib
try:
    from galpy.util import bovy_coords
    _COORDSLOADED= True
except ImportError:
    _COORDSLOADED= False
########################SELECTION FUNCTION DETERMINATION#######################
_INTERPDEGREEBRIGHT= 1
_INTERPDEGREEFAINT= 3
_BINEDGES_G_FAINT= [0.,50.,70.,85.,200000000.]
_BINEDGES_G_BRIGHT= [0.,75.,150.,300.,200000000.]
###############################FILENAMES#######################################
_SEGUESELECTDIR=os.getenv('SEGUESELECTDIR')
_GDWARFALLFILE= os.path.join(_SEGUESELECTDIR,'gdwarfall_raw_nodups.fit')
_GDWARFFILE= os.path.join(_SEGUESELECTDIR,'gdwarf_raw_nodups.fit')
_KDWARFALLFILE= os.path.join(_SEGUESELECTDIR,'kdwarfall_raw_nodups.fit')
_KDWARFFILE= os.path.join(_SEGUESELECTDIR,'kdwarf_raw_nodups.fit')
_ERASESTR= "                                                                                "
class segueSelect:
    """Class that contains selection function for SEGUE targets"""
    def __init__(self,sample='G',plates=None,
                 select='all',
                 type='constant',dr=0.3,
                 interp_degree_bright=_INTERPDEGREEBRIGHT,
                 interp_degree_faint=_INTERPDEGREEFAINT,
                 ug=False,ri=False,sn=True,
                 ebv=False):
        """
        NAME:
           __init__
        PURPOSE:
           load the selection function for this sample
        INPUT:
           sample= sample to load ('G' or 'K', 'GK' loads all BOVY: GK NOT IMPLEMENTED)
           select= 'all' selects all SEGUE stars in the color-range; 
                   'program' only selects program stars
           plates= if set, only consider this plate, or list of plates,
                   or 'faint'/'bright'plates only,
                   or plates '>1000' or '<2000'
           type= type of selection function to determine ('constant' for 
                 constant per plate; 'r' universal function of r FOR FAINT ONLY)
           dr= when determining the selection function as a function of r,
               binsize to use

           SPECTROSCOPIC SAMPLE SELECTION:
              ug= if True, cut on u-g
              ri= if True, cut on r-i
              sn= if False, don't cut on SN
              ebv= if True, cut on E(B-V)
        OUTPUT:
           object
        HISTORY:
           2011-07-08 - Written - Bovy (NYU)
        """
        self.sample=sample.lower()
        #Load plates
        self.platestr= _load_fits(os.path.join(_SEGUESELECTDIR,
                                               'segueplates.fits'))
        #Add platesn_r to platestr
        platesn_r= (self.platestr.sn1_1+self.platestr.sn2_1)/2.
        self.platestr= _append_field_recarray(self.platestr,
                                              'platesn_r',platesn_r)
        if plates is None:
            self.plates= list(self.platestr.plate)
        else:
            if isinstance(plates,str):
                self.plates= self.platestr.plate
                if plates[0] == '>':
                    self.plates= self.plates[(self.plates > int(plates[1:len(plates)]))]
                elif plates[0] == '<':
                    self.plates= self.plates[(self.plates < int(plates[1:len(plates)]))]
                elif plates.lower() == 'faint':
                    indx= ['faint' in name for name in self.platestr.programname]
                    indx= numpy.array(indx,dtype='bool')
                    self.plates= self.plates[indx]
                elif plates.lower() == 'bright':
                    indx= [not 'faint' in name for name in self.platestr.programname]
                    indx= numpy.array(indx,dtype='bool')
                    self.plates= self.plates[indx]
                else:
                    print "'plates=' format not understood, check documentation"
                    return
                self.plates= list(self.plates)
            elif not isinstance(plates,(list,numpy.ndarray)):
                self.plates= [plates]
            elif isinstance(plates,numpy.ndarray):
                self.plates= list(plates)
            else:
                self.plates= plates
        #Remove 2820 for now BOVY DEAL WITH PLATE 2820, 2560, 2799, 2550
        if 2820 in self.plates:
            self.plates.remove(2820)
        if 2560 in self.plates:
            self.plates.remove(2560)
        if 2799 in self.plates:
            self.plates.remove(2799)
        if 2550 in self.plates:
            self.plates.remove(2550)
        #Remove duplicate plates
        self.plates= numpy.array(sorted(list(set(self.plates))))
        #Match platestr to plates again
        allIndx= numpy.arange(len(self.platestr),dtype='int')
        reIndx= numpy.zeros(len(self.plates),dtype='int')-1
        for ii in range(len(self.plates)):
            indx= (self.platestr.field('plate') == self.plates[ii])
            reIndx[ii]= (allIndx[indx][0])
        self.platestr= self.platestr[reIndx]
        #Build bright/faint dict
        self.platebright= {}
        for ii in range(len(self.plates)):
            p= self.plates[ii]
            if 'faint' in self.platestr[ii].programname:
                self.platebright[str(p)]= False
            else:
                self.platebright[str(p)]= True            
        #load the photometry for the SEGUE plates
        self.platephot= {}
        for ii in range(len(self.plates)):
            plate= self.plates[ii]
            sys.stdout.write('\r'+"Loading photometry for plate %i" % plate)
            sys.stdout.flush()
            platefile= os.path.join(_SEGUESELECTDIR,'segueplates',
                                    '%i.fit' % plate)
            self.platephot[str(plate)]= _load_fits(platefile)
            #Split into bright and faint
            if 'faint' in self.platestr[ii].programname:
                indx= (self.platephot[str(plate)].field('r') > 17.8)
                self.platephot[str(plate)]= self.platephot[str(plate)][indx]
            else:
                indx= (self.platephot[str(plate)].field('r') < 17.8)
                self.platephot[str(plate)]= self.platephot[str(plate)][indx]
        sys.stdout.write('\r'+_ERASESTR+'\r')
        sys.stdout.flush()
        #Flesh out samples
        for plate in self.plates:
            if self.sample == 'g':
                self.rmin= 14.5
                self.rmax= 20.2
                indx= ((self.platephot[str(plate)].field('g')\
                           -self.platephot[str(plate)].field('r')) < 0.55)\
                           *((self.platephot[str(plate)].field('g')\
                           -self.platephot[str(plate)].field('r')) > 0.48)\
                           *(self.platephot[str(plate)].field('r') < 20.2)\
                           *(self.platephot[str(plate)].field('r') > 14.5)
            elif self.sample == 'k':
                self.rmin= 14.5
                self.rmax= 19.
                indx= ((self.platephot[str(plate)].field('g')\
                            -self.platephot[str(plate)].field('r')) > 0.55)\
                           *((self.platephot[str(plate)].field('g')\
                           -self.platephot[str(plate)].field('r')) < 0.75)\
                           *(self.platephot[str(plate)].field('r') < 19.)\
                           *(self.platephot[str(plate)].field('r') > 14.5)
            if self.sample == 'gk':
                indx= ((self.platephot[str(plate)].field('g')\
                           -self.platephot[str(plate)].field('r')) < 0.75)\
                           *((self.platephot[str(plate)].field('g')\
                           -self.platephot[str(plate)].field('r')) > 0.48)\
                           *(self.platephot[str(plate)].field('r') < 20.2)\
                           *(self.platephot[str(plate)].field('r') > 14.5)
            self.platephot[str(plate)]= self.platephot[str(plate)][indx]
        #Now load the spectroscopic data
        sys.stdout.write('\r'+"Reading and parsing spectroscopic data ...\r")
        sys.stdout.flush()
        if sample.lower() == 'g':
            if select.lower() == 'all':
                self.spec= read_gdwarfs(ug=ug,ri=ri,sn=sn,
                                        ebv=ebv,nocoords=True)
            elif select.lower() == 'program':
                self.spec= read_gdwarfs(file=_GDWARFFILE,
                                        ug=ug,ri=ri,sn=sn,
                                        ebv=ebv,nocoords=True)
        elif sample.lower() == 'k':
            if select.lower() == 'all':
                self.spec= read_kdwarfs(ug=ug,ri=ri,sn=sn,
                                        ebv=ebv,nocoords=True)
            elif select.lower() == 'program':
                self.spec= read_kdwarfs(file=_KDWARFFILE,
                                        ug=ug,ri=ri,sn=sn,
                                        ebv=ebv,nocoords=True)
        self.platespec= {}
        for plate in self.plates:
            #Find spectra for each plate
            indx= (self.spec.field('plate') == plate)
            self.platespec[str(plate)]= self.spec[indx]
        sys.stdout.write('\r'+_ERASESTR+'\r')
        sys.stdout.flush()
        #Determine selection function
        sys.stdout.write('\r'+"Determining selection function ...\r")
        sys.stdout.flush()
        self._determine_select(type,dr=dr,
                               interp_degree_bright=interp_degree_bright,
                               interp_degree_faint=interp_degree_faint)
        sys.stdout.write('\r'+_ERASESTR+'\r')
        sys.stdout.flush()
        return None

    def __call__(self,plate,r=None,gr=None):
        """
        NAME:
           __call__
        PURPOSE:
           evaluate the selection function
        INPUT:
           plate - plate number
           r - dereddened r-band magnitude
           gr - dereddened g-r color
        OUTPUT:
           selection function
        HISTORY:
           2011-07-11 - Written - Bovy (NYU)
        BUGS: BOVY
           poorly written
           determine first whether r is in the range for this plate
           allow plate to be a single value, r many
        """
        try:
            if isinstance(plate,(list,numpy.ndarray)) \
                    or isinstance(r,(list,numpy.ndarray)):
                if isinstance(r,(list,numpy.ndarray)) \
                        and isinstance(plate,int):
                    plate= [plate for ii in range(len(r))]
                out= []
                for ii in range(len(plate)):
                    p= plate[ii]
                    if self.type.lower() == 'constant':
                        out.append(self.weight[str(p)])
                    elif self.type.lower() == 'r':
                        if r[ii] < 17.8 and not self.platebright[str(p)]:
                            out.append(0.)
                        elif r[ii] >= 17.8 and self.platebright[str(p)]:
                            out.append(0.)
                        elif r[ii] < 17.8 and r[ii] >= self.rmin:
                            out.append(self.weight[str(p)])
                            #BOVY: REMOVE R DEPENDENCE FOR NOW
                            #out.append(self.weight[str(p)]*self.s_one_r_bright_interpolate(r[ii])[0])
                        elif r[ii] >= 17.8 and r[ii] <= self.rmax:
                            out.append(self.weight[str(p)]*self.s_one_r_faint_interpolate(r[ii])) #different interpolator does not return array
                        else:
                            out.append(0.)
                if isinstance(plate,numpy.ndarray):
                    out= numpy.array(out)
                return out
            else:
                if self.type.lower() == 'constant':
                    return self.weight[str(plate)]
                elif self.type.lower() == 'r':
                    if r < 17.8 and r >= self.rmin:
                        return self.weight[str(plate)]
                    ###BOVY: REMOVE R-DEPENDENCE FOR NOW
                    #return self.weight[str(plate)]*self.s_one_r_bright_interpolate(r)[0]
                    elif r >= 17.8 and r <= self.rmax:
                        return self.weight[str(plate)]*self.s_one_r_faint_interpolate(r) #different interpolator does not return array
                    else:
                        return 0.
        except KeyError:
            raise IOError("Requested plate %i either not loaded or it does not exist" % plate)

    def plot(self,x='r',y='sf',plate='a bright plate',overplot=False):
        """
        NAME:
           plot
        PURPOSE:
           plot the derived selection function
        INPUT:
           x= what to plot on the x-axis (e.g., 'r')
           y= what to plot on the y-axis (default function value)
           plate= plate to plot (number or 'a bright plate' (default), 'a faint plate')
           overplot= if True, overplot
        OUTPUT:
           plot to output
        HISTORY:
           2011-07-18 - Written - Bovy (NYU)
        """
        _NXS= 1001
        if isinstance(plate,str) and plate.lower() == 'a bright plate':
            plate= 2964
        elif isinstance(plate,str) and plate.lower() == 'a faint plate':
            plate= 2965
        if x.lower() == 'r':
            xs= numpy.linspace(self.rmin,self.rmax,_NXS)
            xrange= [self.rmin,self.rmax]
            xlabel= r'$r_0\ [\mathrm{mag}]$'
        #Evaluate selection function
        zs= self(plate,r=xs)
        if y.lower() == 'sf':
            ys= zs
            ylabel= r'$\mathrm{selection\ function}$'
            yrange= [0.,1.2*numpy.amax(ys)]
        bovy_plot.bovy_plot(xs,ys,'k-',xrange=xrange,yrange=yrange,
                            xlabel=xlabel,ylabel=ylabel,
                            overplot=overplot)
        #Also plot data
        if self.type.lower() == 'r':
            pindx= (self.plates == plate)
            if self.platebright[str(plate)]:
                bovy_plot.bovy_plot(self.s_r_plate_rs_bright,
                                    self.s_r_plate_bright[:,pindx],
                                    color='k',
                                    marker='o',ls='none',overplot=True)
            else:
                bovy_plot.bovy_plot(self.s_r_plate_rs_faint,
                                    self.s_r_plate_faint[:,pindx],
                                    color='k',
                                    marker='o',ls='none',overplot=True)
        return None

    def plotColorMag(self,x='gr',y='r',plate='all',spec=False,scatterplot=True,
                     bins=None,specbins=None):
        """
        NAME:
           plotColorMag
        PURPOSE:
           plot the distribution of photometric/spectroscopic objects in color
           magnitude (or color-color) space
        INPUT:
           x= what to plot on the x-axis (combinations of ugriz as 'g', 
               or 'gr')
           y= what to plot on the y-axis (combinations of ugriz as 'g',  
               or 'gr')
           plate= plate(s) to plot, int or list/array, 'all', 'bright', 'faint'
           spec= if True, overlay spectroscopic objects as red contours and 
                 histograms
           scatterplot= if False, regular scatterplot, 
                        if True, hogg_scatterplot
           bins= number of bins to use in the histogram(s)
           specbins= number of bins to use in histograms of spectropscopic 
                     objects
       OUTPUT:
        HISTORY:
           2011-07-13 - Written - Bovy (NYU)
        """
        if isinstance(plate,str) and plate.lower() == 'all':
            plate= self.plates
        elif isinstance(plate,str) and plate.lower() == 'bright':
            plate= []
            for ii in range(len(self.plates)):
                if not 'faint' in self.platestr[ii].programname:
                    plate.append(self.plates[ii])
        elif isinstance(plate,str) and plate.lower() == 'faint':
            plate= []
            for ii in range(len(self.plates)):
                if 'faint' in self.platestr[ii].programname:
                    plate.append(self.plates[ii])
        elif isinstance(plate,(list,numpy.ndarray)):
            plate=plate
        else:
            plate= [plate]
        xs, ys= [], []
        specxs, specys= [], []
        for ii in range(len(plate)):
            p=plate[ii]
            thisplatephot= self.platephot[str(p)]
            thisplatespec= self.platespec[str(p)]
            if len(x) > 1: #Color
                xs.extend(thisplatephot.field(x[0])\
                              -thisplatephot.field(x[1])) #dereddened
                specxs.extend(thisplatespec.field('dered_'+x[0])\
                                  -thisplatespec.field('dered_'+x[1]))
            else:
                xs.extend(thisplatephot.field(x[0]))
                specxs.extend(thisplatespec.field('dered_'+x[0]))
            if len(y) > 1: #Color
                ys.extend(thisplatephot.field(y[0])\
                              -thisplatephot.field(y[1])) #dereddened
                specys.extend(thisplatespec.field('dered_'+y[0])\
                                  -thisplatespec.field('dered_'+y[1]))
            else:
                ys.extend(thisplatephot.field(y[0]))
                specys.extend(thisplatespec.field('dered_'+y[0]))
        xs= numpy.array(xs)
        xs= numpy.reshape(xs,numpy.prod(xs.shape))
        ys= numpy.array(ys)
        ys= numpy.reshape(ys,numpy.prod(ys.shape))
        specxs= numpy.array(specxs)
        specxs= numpy.reshape(specxs,numpy.prod(specxs.shape))
        specys= numpy.array(specys)
        specys= numpy.reshape(specys,numpy.prod(specys.shape))
        if len(x) > 1:
            xlabel= '('+x[0]+'-'+x[1]+')_0'
        else:
            xlabel= x[0]+'_0'
        xlabel= r'$'+xlabel+r'$'
        if len(y) > 1:
            ylabel= '('+y[0]+'-'+y[1]+')_0'
        else:
            ylabel= y[0]+'_0'
        ylabel= r'$'+ylabel+r'$'
        if len(xs) > 1: #color
            xrange= [numpy.amin(xs)-0.1,numpy.amax(xs)+0.1]
        else:
            xrange= [numpy.amin(xs)-0.7,numpy.amax(xs)+0.7]
        if len(ys) > 1: #color
            yrange= [numpy.amin(ys)-0.1,numpy.amax(ys)+0.1]
        else:
            yrange= [numpy.amin(ys)-0.7,numpy.amax(ys)+0.7]
        if bins is None:
            bins= int(numpy.ceil(0.3*numpy.sqrt(len(xs))))
        if specbins is None: specbins= bins
        if scatterplot:
            if len(xs) > 100000: symb= 'w,'
            else: symb= 'k,'
            if spec:
                #First plot spectroscopic sample
                cdict = {'red': ((.0, 1.0, 1.0),
                                 (1.0, 1.0, 1.0)),
                         'green': ((.0, 1.0, 1.0),
                                   (1.0, 1.0, 1.0)),
                         'blue': ((.0, 1.0, 1.0),
                                  (1.0, 1.0, 1.0))}
                allwhite = matplotlib.colors.LinearSegmentedColormap('allwhite',cdict,256)
                speclevels= list(special.erf(0.5*numpy.arange(1,4)))
                speclevels.append(1.01)#HACK TO REMOVE OUTLIERS
                bovy_plot.scatterplot(specxs,specys,symb,onedhists=True,
                                      levels=speclevels,
                                      onedhistec='r',
                                      cntrcolors='r',
                                      cmap=allwhite,
                                      xlabel=xlabel,ylabel=ylabel,
                                      xrange=xrange,yrange=yrange,
                                      bins=specbins)
                
            bovy_plot.scatterplot(xs,ys,symb,onedhists=True,
                                  xlabel=xlabel,ylabel=ylabel,
                                  xrange=xrange,yrange=yrange,bins=bins,
                                  overplot=spec)
        else:
            bovy_plot.bovy_plot(xs,ys,'k,',onedhists=True,
                                xlabel=xlabel,ylabel=ylabel,
                                xrange=xrange,yrange=yrange)
        return None                

    def _determine_select(self,type,dr=None,
                          interp_degree_bright=_INTERPDEGREEBRIGHT,
                          interp_degree_faint=_INTERPDEGREEFAINT):
        self.type= type
        #First determine the total weight for each plate
        self.weight= {}
        for ii in range(len(self.plates)):
            plate= self.plates[ii]
            self.weight[str(plate)]= len(self.platespec[str(plate)])\
                /float(len(self.platephot[str(plate)]))
        if type.lower() == 'constant':
            return
        if type.lower() == 'r':
            #Determine the selection function in bins in r, for bright/faint
            nbrightrbins= int(math.floor((17.8-self.rmin)/dr))+1
            nfaintrbins= int(math.floor((self.rmax-17.8)/dr))+2
            s_one_r_bright= numpy.zeros((nbrightrbins,len(self.plates)))
            s_one_r_faint= numpy.zeros((nfaintrbins,len(self.plates)))
            s_r_bright= numpy.zeros((nbrightrbins,len(self.plates)))
            s_r_faint= numpy.zeros((nfaintrbins,len(self.plates)))
            #Determine s_1(r) for each plate separately first
            nbrightplates, nfaintplates= 0, 0
            brightplateindx= numpy.empty(len(self.plates),dtype='bool') #BOVY: move this out of here
            faintplateindx= numpy.empty(len(self.plates),dtype='bool')
            weightbrights, weightfaints= 0., 0.
            for ii in range(len(self.plates)):
                plate= self.plates[ii]
                if 'faint' in self.platestr[ii].programname: #faint plate
                    thisrmin, thisrmax= 17.8, self.rmax+dr #slightly further to avoid out-of-range errors
                    thisbins= nfaintrbins
                    nfaintplates+= 1
                    faintplateindx[ii]= True
                    brightplateindx[ii]= False
                else:
                    thisrmin, thisrmax= self.rmin, 17.8
                    thisbins= nbrightrbins
                    nbrightplates+= 1
                    faintplateindx[ii]= False
                    brightplateindx[ii]= True
                nspecr, edges = numpy.histogram(self.platespec[str(plate)].dered_r,bins=thisbins,range=[thisrmin,thisrmax])
                nphotr, edges = numpy.histogram(self.platephot[str(plate)].r,
                                                bins=thisbins,
                                                range=[thisrmin,thisrmax])
                nspecr= numpy.array(nspecr,dtype='float64')
                nphotr= numpy.array(nphotr,dtype='float64')
                nonzero= (nspecr > 0.)*(nphotr > 0.)
                if 'faint' in self.platestr[ii].programname: #faint plate
                    s_r_faint[nonzero,ii]= nspecr[nonzero].astype('float64')/nphotr[nonzero]
                    weightfaints+= float(numpy.sum(nspecr))/float(numpy.sum(nphotr))
                else: #bright plate
                    s_r_bright[nonzero,ii]= nspecr[nonzero].astype('float64')/nphotr[nonzero]
                    weightbrights+= float(numpy.sum(nspecr))/float(numpy.sum(nphotr))
                nspecr/= float(numpy.sum(nspecr))
                nphotr/= float(numpy.sum(nphotr))
                if 'faint' in self.platestr[ii].programname: #faint plate
                    s_one_r_faint[nonzero,ii]= nspecr[nonzero]/nphotr[nonzero]
                else: #bright plate
                    s_one_r_bright[nonzero,ii]= nspecr[nonzero]/nphotr[nonzero]
            self.s_r_plate_rs_bright= \
                numpy.linspace(self.rmin+dr/2.,17.8-dr/2.,nbrightrbins)
            self.s_r_plate_rs_faint= \
                numpy.linspace(17.8+dr/2.,self.rmax-dr/2.,nfaintrbins)
            self.s_r_plate_bright= s_r_bright
            self.s_r_plate_faint= s_r_faint
            self.s_one_r_plate_bright= s_one_r_bright
            self.s_one_r_plate_faint= s_one_r_faint
            self.nbrightplates= nbrightplates
            self.nfaintplates= nfaintplates
            self.brightplateindx= brightplateindx
            self.faintplateindx= faintplateindx
            fromIndividual= False
            if fromIndividual:
                #Mean or median?
                median= False
                if median:
                    self.s_one_r_bright= numpy.median(self.s_one_r_plate_bright[:,self.brightplateindx],axis=1)
                    self.s_one_r_faint= numpy.median(self.s_one_r_plate_faint[:,self.faintplateindx],axis=1)
                else:
                    self.s_one_r_bright= numpy.sum(self.s_one_r_plate_bright,axis=1)/nbrightplates
                    self.s_one_r_faint= numpy.sum(self.s_one_r_plate_faint,axis=1)/nfaintplates
                print self.s_one_r_bright, self.s_one_r_faint
            else:
                self.s_one_r_bright= \
                    numpy.sum(self.s_r_plate_bright[:,self.brightplateindx],axis=1)\
                    /weightbrights
                self.s_one_r_faint= \
                    numpy.sum(self.s_r_plate_faint[:,self.faintplateindx],axis=1)\
                    /weightfaints
            self.interp_rs_bright= \
                numpy.linspace(self.rmin+5.*dr/2.,17.8-5.*dr/2.,nbrightrbins-4)
            self.s_one_r_bright_interpolate= interpolate.UnivariateSpline(\
                self.interp_rs_bright,self.s_one_r_bright[2:-2],k=_INTERPDEGREEBRIGHT)
            self.interp_rs_faint= \
                numpy.linspace(17.8+1.*dr/2.,self.rmax+dr/2.,nfaintrbins)
            self.s_one_r_faint_interpolate= interpolate.interp1d(\
                self.interp_rs_faint,self.s_one_r_faint,#[1:len(self.s_one_r_faint)],
                kind=_INTERPDEGREEFAINT,fill_value=self.s_one_r_faint[0],
                bounds_error=False)
            return

def ivezic_dist_gr(g,r,feh):
    """
    NAME:
       ivezic_dist_gr
    PURPOSE:
        Iveziv et al. (2008) distances in terms of g-r for <M0 stars
    INPUT:
       g, r, feh - dereddened g and r and metallicity
    OUTPUT:
       (dist,disterr) arrays in kpc
    HISTORY:
       2011-07-11 - Written - Bovy (NYU)
    """
    #First distances, then uncertainties
    gi= _gi_gr(g-r)
    mr= _mr_gi(gi,feh)
    ds= 10.**(0.2*(r-mr)-2.)
    #Now propagate the uncertainties
    derrs= ds/10. #BOVY: ASSUME 10% for now
    return (ds,derrs)

def read_gdwarfs(file=_GDWARFALLFILE,logg=False,ug=False,ri=False,sn=True,
                 ebv=False,nocoords=False):
    """
    NAME:
       read_gdwarfs
    PURPOSE:
       read the spectroscopic G dwarf sample
    INPUT:
       logg= if True, cut on logg
       ug= if True, cut on u-g
       ri= if True, cut on r-i
       sn= if False, don't cut on SN
       ebv= if True, cut on E(B-V)
       nocoords= if True, don't calculate distances or transform coordinates
    OUTPUT:
       cut data, returns numpy.recarray
    HISTORY:
       2011-07-08 - Written - Bovy (NYU)
    """
    raw= _load_fits(file)
    #First cut on r
    indx= (raw.field('dered_r') < 20.2)*(raw.field('dered_r') > 14.5)
    raw= raw[indx]
    #Then cut on g-r
    indx= ((raw.field('dered_g')-raw.field('dered_r')) < 0.55)\
        *((raw.field('dered_g')-raw.field('dered_r')) > .48)
    raw= raw[indx]
    #Cut on velocity errs
    indx= (raw.field('pmra_err') > 0.)*(raw.field('pmdec_err') > 0.)\
        *(raw.field('vr_err') > 0.)
    raw= raw[indx]
    #Cut on logg?
    if logg:
        indx= (raw.field('logga') > 3.75)
        raw= raw[indx]
    if ug:
        indx= ((raw.field('dered_u')-raw.field('dered_g')) < 2.)\
            *((raw.field('dered_u')-raw.field('dered_g')) > .6)
        raw= raw[indx]
    if ri:
        indx= ((raw.field('dered_r')-raw.field('dered_i')) < .4)\
            *((raw.field('dered_r')-raw.field('dered_i')) > -.1)
        raw= raw[indx]
    if sn:
        indx= (raw.field('sna') > 15.)
        raw= raw[indx]
    if ebv:
        indx= (raw.field('ebv') < .3)
        raw= raw[indx]
    if nocoords: return raw
    #BOVY: distances
    raw= _add_distances(raw)
    #velocities
    raw= _add_velocities(raw)
    return raw

def read_kdwarfs(file=_KDWARFALLFILE,logg=False,ug=False,ri=False,sn=True,
                 ebv=False,nocoords=False):
    """
    NAME:
       read_kdwarfs
    PURPOSE:
       read the spectroscopic K dwarf sample
    INPUT:
       logg= if True, cut on logg
       ug= if True, cut on u-g
       ri= if True, cut on r-i
       sn= if False, don't cut on SN
       ebv= if True, cut on E(B-V)
       nocoords= if True, don't calculate distances or transform coordinates
    OUTPUT:
       cut data, returns numpy.recarray
    HISTORY:
       2011-07-11 - Written - Bovy (NYU)
    """
    raw= _load_fits(file)
    #First cut on r
    indx= (raw.field('dered_r') < 19.)*(raw.field('dered_r') > 14.5)
    raw= raw[indx]
    #Then cut on g-r
    indx= ((raw.field('dered_g')-raw.field('dered_r')) < 0.75)\
        *((raw.field('dered_g')-raw.field('dered_r')) > .55)
    raw= raw[indx]
    #Cut on velocity errs
    indx= (raw.field('pmra_err') > 0.)*(raw.field('pmdec_err') > 0.)\
        *(raw.field('vr_err') > 0.)
    raw= raw[indx]
    #Cut on logg?
    if logg:
        indx= (raw.field('logga') > 3.75)
        raw= raw[indx]
    if ug: #BOVY UPDATE FOR K
        indx= ((raw.field('dered_u')-raw.field('dered_g')) < 2.)\
            *((raw.field('dered_u')-raw.field('dered_g')) > .6)
        raw= raw[indx]
    if ri: #BOVY: UPDATE FOR K
        indx= ((raw.field('dered_r')-raw.field('dered_i')) < .4)\
            *((raw.field('dered_r')-raw.field('dered_i')) > -.1)
        raw= raw[indx]
    if sn:
        indx= (raw.field('sna') > 15.)
        raw= raw[indx]
    if ebv:
        indx= (raw.field('ebv') < .3)
        raw= raw[indx]
    if nocoords: return raw
    #BOVY: distances
    raw= _add_distances(raw)
    #velocities
    raw= _add_velocities(raw)
    return raw

def _add_distances(raw):
    """Add distances"""
    ds,derrs= ivezic_dist_gr(raw.dered_g,raw.dered_r,raw.feh)
    raw= _append_field_recarray(raw,'dist',ds)
    raw= _append_field_recarray(raw,'dist_err',derrs)
    return raw

def _add_velocities(raw):
    if not _COORDSLOADED:
        print "galpy.util.bovy_coords failed to load ..."
        print "Install galpy for coordinate transformations ..."
        print "*not* adding velocities ..."
        return raw
    #We start from RA and Dec
    lb= bovy_coords.radec_to_lb(raw.ra,raw.dec,degree=True)
    XYZ= bovy_coords.lbd_to_XYZ(lb[:,0],lb[:,1],raw.dist,degree=True)
    pmllpmbb= bovy_coords.pmrapmdec_to_pmllpmbb(raw.pmra,raw.pmdec,
                                                raw.ra,raw.dec,degree=True)
    #print numpy.mean(pmllpmbb[:,0]-raw.pml), numpy.std(pmllpmbb[:,0]-raw.pml)
    #print numpy.mean(pmllpmbb[:,1]-raw.pmb), numpy.std(pmllpmbb[:,1]-raw.pmb)
    vxvyvz= bovy_coords.vrpmllpmbb_to_vxvyvz(raw.vr,pmllpmbb[:,0],
                                             pmllpmbb[:,1],lb[:,0],lb[:,1],
                                             raw.dist,degree=True)
    #Solar motion from Schoenrich & Binney
    vxvyvz[:,0]+= -11.1
    vxvyvz[:,1]+= 12.24
    vxvyvz[:,2]+= 7.25
    #print numpy.mean(vxvyvz[:,2]), numpy.std(vxvyvz[:,2])
    #Propagate uncertainties
    ndata= len(raw.ra)
    cov_pmradec= numpy.zeros((ndata,2,2))
    cov_pmradec[:,0,0]= raw.pmra_err**2.
    cov_pmradec[:,1,1]= raw.pmdec_err**2.
    cov_pmllbb= bovy_coords.cov_pmrapmdec_to_pmllpmbb(cov_pmradec,raw.ra,
                                                      raw.dec,degree=True)
    cov_vxvyvz= bovy_coords.cov_dvrpmllbb_to_vxyz(raw.dist,
                                                  raw.dist_err,
                                                  raw.vr_err,
                                                  pmllpmbb[:,0],pmllpmbb[:,1],
                                                  cov_pmllbb,lb[:,0],lb[:,1],
                                                  degree=True)
    #Cast
    XYZ= XYZ.astype(numpy.float64)
    vxvyvz= vxvyvz.astype(numpy.float64)
    cov_vxvyvz= cov_vxvyvz.astype(numpy.float64)
    #Append results to structure
    raw= _append_field_recarray(raw,'xc',XYZ[:,0])
    raw= _append_field_recarray(raw,'yc',XYZ[:,1])
    raw= _append_field_recarray(raw,'zc',XYZ[:,2])
    raw= _append_field_recarray(raw,'vxc',vxvyvz[:,0])
    raw= _append_field_recarray(raw,'vyc',vxvyvz[:,1])
    raw= _append_field_recarray(raw,'vzc',vxvyvz[:,2])
    raw= _append_field_recarray(raw,'vxc_err',numpy.sqrt(cov_vxvyvz[:,0,0]))
    raw= _append_field_recarray(raw,'vyc_err',numpy.sqrt(cov_vxvyvz[:,1,1]))
    raw= _append_field_recarray(raw,'vzc_err',numpy.sqrt(cov_vxvyvz[:,2,2]))
    raw= _append_field_recarray(raw,'vxvyc_rho',cov_vxvyvz[:,0,1]\
                                    /numpy.sqrt(cov_vxvyvz[:,0,0])\
                                    /numpy.sqrt(cov_vxvyvz[:,1,1]))
    raw= _append_field_recarray(raw,'vxvzc_rho',cov_vxvyvz[:,0,2]\
                                    /numpy.sqrt(cov_vxvyvz[:,0,0])\
                                    /numpy.sqrt(cov_vxvyvz[:,2,2]))
    raw= _append_field_recarray(raw,'vyvzc_rho',cov_vxvyvz[:,1,2]\
                                    /numpy.sqrt(cov_vxvyvz[:,1,1])\
                                    /numpy.sqrt(cov_vxvyvz[:,2,2]))
    return raw

def _load_fits(file,ext=1):
    """Loads fits file's data and returns it as a numpy.recarray with lowercase field names"""
    hdulist= pyfits.open(file)
    out= hdulist[ext].data
    hdulist.close()
    return _as_recarray(out)

def _append_field_recarray(recarray, name, new):
    new = numpy.asarray(new)
    newdtype = numpy.dtype(recarray.dtype.descr + [(name, new.dtype)])
    newrecarray = numpy.recarray(recarray.shape, dtype=newdtype)
    for field in recarray.dtype.fields:
        newrecarray[field] = recarray.field(field)
    newrecarray[name] = new
    return newrecarray

def _as_recarray(recarray):
    """go from FITS_rec to recarray"""
    newdtype = numpy.dtype(recarray.dtype.descr)
    newdtype.names= tuple([n.lower() for n in newdtype.names])
    newrecarray = numpy.recarray(recarray.shape, dtype=newdtype)
    for field in recarray.dtype.fields:
        newrecarray[field.lower()] = recarray.field(field)
    return newrecarray

#Ivezic distance functions
def _mr_gi(gi,feh):
    """Ivezic+08 photometric distance"""
    mro= -5.06+14.32*gi-12.97*gi**2.+6.127*gi**3.-1.267*gi**4.+0.0967*gi**5.
    dmr= 4.5-1.11*feh-0.18*feh**2.
    mr= mro+dmr
    return mr

def _gi_gr(gr):
    """(g-i) = (g-r)+(r-i), with Juric et al. (2008) stellar locus for g-r,
    BOVY: JUST USES LINEAR APPROXIMATION VALID FOR < M0"""
    ri= (gr-0.07)/2.34
    return gr+ri
