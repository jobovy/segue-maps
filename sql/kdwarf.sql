select spp.mjd, spp.plate, spp.fiberID,
s.ra as RA, s.dec as Dec, s.b as b, s.l as l, spp.ebv,
s.psfMag_u as u, s.extinction_u as ext_u, 
s.psfMag_u-s.extinction_u as dered_u, s.psfMagErr_u as u_err,     
s.psfMag_g as g, s.extinction_g as ext_g, 
s.psfMag_g-s.extinction_g as dered_g, s.psfMagErr_g as g_err, 
s.psfMag_r as r, s.extinction_r as ext_r, 
s.psfMag_r-s.extinction_r as dered_r, s.psfMagErr_r as r_err, 
s.psfMag_i as i, s.extinction_i as ext_i, 
s.psfMag_i-s.extinction_i as dered_i, s.psfMagErr_i as i_err,
s.psfMag_g as z, s.extinction_z as ext_z, 
s.psfMag_z-s.extinction_z as dered_z, s.psfMagErr_z as z_err,
spp.ELODIERV as vr, spp.ELODIERVERR as vr_err, 
--pm.pmL, pm.pmB, pm.pmRa, pm.pmDec, pm.pmRaErr, pm.pmDecErr, 
spp.targ_pmmatch as match, spp.targ_pml as pml,
spp.targ_pmb as pmb, spp.targ_pmra as pmra, spp.targ_pmraerr as pmra_err,
spp.targ_pmdec as pmdec, spp.targ_pmdecerr as pmdec_err, 
--spp.FEHADOP as FeHa, spp.FEHADOPUNC as feha_err,
--spp.FEHSPEC as FeH_spe, spp.FEHSPECUNC as FeH_spe_err, 
--spp.LOGGADOP as log_g, spp.TEFFADOP as teff
spp.feha as feh, spp.fehaerr as feh_err, spp.teffa as Teff, 
spp.teffaerr as Tef_err, spp.logga as logga,
spp.loggaerr as logg_err,spp.alphafe as afe,spp.alphafeerr as afe_err,
   spp.zbsubclass as Type, spp.zbelodiesptype, soa.primtarget as
primtarget, spp.sna as sna,s.run, s.rerun, s.camcol, s.field, s.obj
into MyDB.kdwarf
from  sppParams as spp, SpecObjAll as soa, star as s, platex as p
--, ProperMotions pm
where  p.programname like '%segue%' 
and p.plate = soa.plate
and soa.primtarget = -2147450880
and spp.specobjid = soa.specobjid and soa.bestobjid = s.objid
--and s.objid = pm.objid
--and s.psfMag_r-s.extinction_r between 14.5 and 19.0
and s.psfMag_g-s.extinction_g-s.psfMag_r+s.extinction_r between .5 and .8
--and pm.pmRaErr > 0. and spp.SNR > 15.0