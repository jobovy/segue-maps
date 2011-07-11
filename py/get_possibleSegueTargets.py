import re
import sys
import os.path
import subprocess
import numpy as nu
import astrometry.util.pyfits_utils as pyfits_utils
import pyfits
from optparse import OptionParser
_PYTHON='/usr/local/epd/bin/python'
_CASJOBS='/home/users/jb2777/scr/astrometry/src/astrometry/util/casjobs.py'
def get_possibleSegueTargets():
    """
    NAME:
       get_possibleSegueTargets
    PURPOSE:
       get the possible Segue targets from the CAS
    INPUT:
       parser - from optParser
    OUTPUT:
    HISTORY:
       2010-12-21 - Written - Bovy (NYU)
       2011-07-07 - Adapted for SEGUE - Bovy
    """
    caspwdfile= open('caspwd','r')
    casusr= caspwdfile.readline().rstrip()
    caspwd= caspwdfile.readline().rstrip()
    caspwdfile.close()
    #Read plate centers file
    platefile= os.path.join(os.getenv('DATADIR'),'bovy',
                            'segue-local','segueplates.fits')
    platecenters= pyfits_utils.table_fields(platefile)[::-1]
    nplates= len(platecenters.ra)
    done= []
    for plate in platecenters:
        print "Working on plate "+str(plate.plate)
        if plate.plate in done:
            print str(plate.plate)+" already done!"
            continue
        done.append(plate.plate)
        tmpsavefilename= os.path.join(os.getenv('DATADIR'),'bovy',
                                      'segue-local','segueplates',
                                      '%i.fit' % plate.plate)
        if os.path.exists(tmpsavefilename):
            print "file "+tmpsavefilename+" exists"
            print "Delete file "+tmpsavefilename+" before running this to update the sample from the CAS"
        else:
            dbname= prepare_sql(plate)
            subprocess.call([_PYTHON,_CASJOBS,casusr,caspwd,'querywait',
                             '@tmp.sql'])
            subprocess.call([_PYTHON,_CASJOBS,casusr,caspwd,
                             'outputdownloaddelete',dbname,tmpsavefilename])

def prepare_sql(plate):
    output_f = open('tmp1.sql', 'w')
    subprocess.call(["sed",'s/PLATERA/'+str(plate.ra).strip()+'/g',"../sql/possibleSegueTargets.sql"],stdout=output_f)
    output_f.close()
    output_f = open('tmp2.sql', 'w')
    subprocess.call(["sed",'s/PLATEDEC/'+str(plate.dec).strip()+'/g','tmp1.sql'],stdout=output_f)
    output_f.close()
    output_f = open('tmp.sql', 'w')
    dbname= 'platen'+str(plate.plate)
    d= ''
    for db in dbname:
        d+= db
    dbname=d
    if d[0] in '0123456789':
        d= d[1:-1]
    subprocess.call(["sed",'s/DBNAME/'+d+'/g','tmp2.sql'],stdout=output_f)
    output_f.close()
    return d
       
if __name__ == '__main__':
    get_possibleSegueTargets()
