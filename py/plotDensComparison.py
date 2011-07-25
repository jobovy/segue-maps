import os, os.path
import numpy
from optparse import OptionParser
from galpy.util import bovy_plot
import segueSelect
from fitSigz import readData
from fitDensz import _HWRDensity, _FlareDensity, _const_colordist
import compareDataModel
def compareGRichRdist(options,args):
    #Set up density models and their parameters
    model1= _FlareDensity
    model2= _HWRDensity
    params1=  numpy.array([-1.20815403035,0.981614534466,-0.130775138707])
    params2= numpy.log(numpy.array([0.3,2.45,1.]))
    #Load sf
    sf= segueSelect.segueSelect(sample=options.sample,sn=True)
    #Load data
    XYZ,vxvyvz,cov_vxvyvz,data= readData(metal=options.metal,
                                         sample=options.sample)
    #We do bright/faint for 4 directions
    plate= compareDataModel.similarPlatesDirection(180.,0.,20.,sf,data,
                                                   faint=False)
    bins= 21
    bovy_plot.bovy_print()
    compareDataModel.comparerdistPlate(model1,params1,sf,_const_colordist,
                                       data,plate,color='k',bins=bins,ls='-')
    compareDataModel.comparerdistPlate(model2,params2,sf,_const_colordist,
                                       data,plate,color='k',bins=bins,
                                       overplot=True,ls='--')
    bovy_plot.bovy_end_print(os.path.join(args[0],'Flare_Dblexp_g_rich_l180_b0.eps'))

def get_options():
    usage = "usage: %prog [options] <savedir>\n\nsavedir= name of the directory that the comparisons will be saved to"
    parser = OptionParser(usage=usage)
    parser.add_option("--sample",dest='sample',default='g',
                      help="Use 'G' or 'K' dwarf sample")
    parser.add_option("--metal",dest='metal',default='rich',
                      help="Use metal-poor or rich sample ('poor', 'rich' or 'all')")
    return parser


if __name__ == '__main__':
    (options,args)= get_options().parse_args()
    if options.sample.lower() == 'g' and options.metal.lower() == 'rich':
        compareGRichRdist(options,args)
