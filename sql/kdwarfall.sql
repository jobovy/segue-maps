select sp.specObjID, sp.mjd, sp.plate, sp.fiberID,sp.specClass, sp.primTarget,
sp.sci_sn,ph.objID, ph.psfMag_u as u, ph.psfMag_g as g, ph.psfMag_r as r, 
ph.psfMag_i as i, ph.psfMag_z as z, ph.ra, ph.dec, ph.raErr, ph.decErr,
ph.raDecCorr, ph.b, ph.l, ph.extinction_u, ph.extinction_g, ph.extinction_r, 
ph.extinction_i, ph.extinction_z, p.plate as plate, sspp.feha, sspp.fehaerr, 
sspp.teffa, sspp.teffaerr, sspp.logga, sspp.loggaerr, sspp.alphafe, 
sspp.alphafeerr, sspp.elodierv as vr, sspp.elodierverr as vr_err, sspp.sna, 
sspp.targ_pml as pmll, sspp.targ_pmb as pmbb, sspp.targ_pmra as pmra, 
sspp.targ_pmdec as pmdec, sspp.targ_pmraerr as pmra_err, 
sspp.targ_pmdecerr as pmdec_err, sspp.zbsubclass as type, 
sspp.zbelodiesptype as specTypeElodie,
sp.primtarget, sspp.sna
into mydb.Kdwarfs
from specObjAll sp, star ph, platex p, sppparams sspp
where p.plate = sp.plate
and sp.bestobjid = ph.objid
and sspp.specobjid = sp.specobjid
and sspp.targ_pmraerr > 0
and sspp.targ_pmdecerr > 0
and ((p.programname like '%segue%' and sp.primtarget = -2147450880) or
((ph.psfMag_r-ph.extinction_r between 14.5 and  19.0) and 
ph.psfMag_g-ph.extinction_g-ph.psfMag_r+ph.extinction_r between 0.55 and 0.75))
