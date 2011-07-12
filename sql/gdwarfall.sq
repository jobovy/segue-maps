select spp.mjd, spp.plate, spp.fiberID,
   spp.ra as RA, spp.dec as Dec, spp.b, spp.l, spp.ebv,
   s.psfMag_u as u, s.extinction_u as ext_u, s.psfMag_u-s.extinction_u as
dered_u,
s.psfMagErr_u as u_err,
   s.psfMag_g as g, s.extinction_g as ext_g, s.psfMag_g-s.extinction_g as
dered_g,
s.psfMagErr_g as g_err,
   s.psfMag_r as r, s.extinction_r as ext_r, s.psfMag_r-s.extinction_r as
dered_r,
s.psfMagErr_r as r_err,
   s.psfMag_i as i, s.extinction_i as ext_i, s.psfMag_i-s.extinction_i as
dered_i,
s.psfMagErr_i as i_err,
   s.psfMag_g as z, s.extinction_z as ext_z, s.psfMag_z-s.extinction_z as
dered_z,
s.psfMagErr_z as z_err,
   spp.elodierv as vr, spp.elodierverr as vr_err, 
spp.targ_pmmatch as match, spp.targ_pml as pml,
spp.targ_pmb as pmb, spp.targ_pmra as pmra, spp.targ_pmraerr as pmra_err,
spp.targ_pmdec as pmdec, spp.targ_pmdecerr as pmdec_err, 
spp.feha as feh, spp.fehaerr as feh_err, spp.teffa as Teff, 
spp.teffaerr as Tef_err, spp.logga as logga,
spp.loggaerr as logg_err,spp.alphafe as afe,spp.alphafeerr as afe_err,
   spp.zbsubclass as Type, spp.zbelodiesptype, soa.primtarget as
primtarget, spp.sna as sna,s.run, s.rerun, s.camcol, s.field, s.obj
into MyDB.gdwarf
from  sppParams as spp, specObjAll as soa, star as s,  platex p
where  p.programname like '%segue%'
   and p.plate = soa.plate
   and spp.specobjid = soa.specobjid and soa.bestobjid = s.objid
   and s.psfMag_g-s.extinction_g-s.psfMag_r+s.extinction_r between 0.48
and 0.55