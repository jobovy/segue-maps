OUTDIR=../figs
EXT=png

#Sig_z(z) for both metal rich and metal poor

$(OUTDIR)/HWR_rich_poor_g_func.$(EXT): plotSigzFunc.py
	$(PYTHON) plotSigzFunc.py HWR_rich_g.sav HWR_poor_g.sav \
	-o $@

#Metal rich G dwarf densities
$(OUTDIR)/HWR_rich_g_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g.sav -o $@ \
	--expd1 --expd2 -n 100 --colordist=binned \
	--xmin=0. --xmax=25. --ymin=0. --ymax=30. \
	--xlabel='h_z\ [\mathrm{kpc}]' \
	--ylabel='\nu(z;R=8\ [\mathrm{kpc}])' # \
#	--subsample=1000
$(OUTDIR)/HWR_rich_g_zfunc.$(EXT): fitDensz.py Makefile plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g.sav -o $@ \
	-n 100 --plotzfunc \
	--xlabel='z\ [\mathrm{kpc}]' \
	--ylabel='\ln \nu(z;R=8\ \mathrm{kpc})' # \
#	--subsample=1000
$(OUTDIR)/HWR_rich_g_rfunc.$(EXT): fitDensz.py Makefile plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g.sav -o $@ \
	-n 100 --plotrfunc \
	--xlabel='r_0\ [\mathrm{mag}]' \
	--ylabel='\mathrm{density}' # \
#	--subsample=1000

#TwoVertical model
$(OUTDIR)/HWR_rich_g_twovertical_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_twovertical.sav -o $@ \
	-n 100 --plotrfunc \
	--model=twovertical \
	--xlabel='r_0\ [\mathrm{mag}]' \
	--ylabel='\mathrm{density}' # \
#	--subsample=1000
#Flare model
$(OUTDIR)/HWR_rich_g_flare_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_flare.sav -o $@ \
	-n 100 --plotrfunc \
	--model=flare \
	--xlabel='r_0\ [\mathrm{mag}]' \
	--ylabel='\mathrm{density}' # \
#	--subsample=1000
#Tied Flare model
$(OUTDIR)/HWR_rich_g_tiedflare_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_tiedflare.sav -o $@ \
	-n 100 --plotrfunc \
	--model=tiedflare \
	--xlabel='r_0\ [\mathrm{mag}]' \
	--ylabel='\mathrm{density}' # \
#	--subsample=1000

#just bright
$(OUTDIR)/HWR_rich_gbright_rfunc.$(EXT): fitDensz.py Makefile plotData.py \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_gbright.sav -o $@ \
	-n 100 --plotrfunc --bright \
	--xlabel='r_0\ [\mathrm{mag}]' \
	--ylabel='\mathrm{density}' # \
#	--subsample=1000
$(OUTDIR)/TwoVertical_rich_gbright_rfunc.$(EXT): fitDensz.py Makefile \
	segueSelect.py
	$(PYTHON) fitDensz.py TwoVertical_dens_rich_gbright.sav -o $@ \
	-n 100 --plotrfunc --bright --model=twovertical \
	--xlabel='r_0\ [\mathrm{mag}]' \
	--ylabel='\mathrm{density}' # \
#	--subsample=1000
$(OUTDIR)/DblExp_rich_gbright_rfunc.$(EXT): fitDensz.py Makefile plotData.py \
	segueSelect.py
	$(PYTHON) fitDensz.py DblExp_dens_rich_gbright.sav -o $@ \
	-n 100 --plotrfunc --bright --model=dblexp \
	--xlabel='r_0\ [\mathrm{mag}]' \
	--ylabel='\mathrm{density}' # \
#	--subsample=1000


#Fake data
$(OUTDIR)/HWR_rich_fakedblexp_g_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_fakedblexp_g.sav -o $@ \
	-n 100 --plotrfunc --sel_faint=r --fake -i fakeThinDblExp.sav \
	--xlabel='r_0\ [\mathrm{mag}]' \
	--ylabel='\mathrm{density}' # \
#	--subsample=1000
$(OUTDIR)/HWR_rich_fakedblexp_gbright_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_fakedblexp_gbright.sav -o $@ \
	-n 100 --plotrfunc --sel_faint=r --fake -i fakeThinDblExp.sav \
	--xlabel='r_0\ [\mathrm{mag}]' --bright \
	--ylabel='\mathrm{density}' # \
#	--subsample=1000
$(OUTDIR)/Flare_rich_fakedblexp_g_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py \
	segueSelect.py
	$(PYTHON) fitDensz.py Flare_dens_rich_fakedblexp_g.sav -o $@ \
	-n 100 --plotrfunc --sel_faint=r --fake -i fakeThinDblExp.sav \
	--model=flare \
	--xlabel='r_0\ [\mathrm{mag}]' \
	--ylabel='\mathrm{density}' # \
#	--subsample=1000
$(OUTDIR)/Flare_rich_fakeflare_g_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py \
	segueSelect.py
	$(PYTHON) fitDensz.py Flare_dens_rich_fakedblexp_g.sav -o $@ \
	-n 100 --plotrfunc --sel_faint=r --fake -i fakeThinFlare.sav \
	--model=flare \
	--xlabel='r_0\ [\mathrm{mag}]' \
	--ylabel='\mathrm{density}' # \
#	--subsample=1000
$(OUTDIR)/HWR_rich_fakeflare_g_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_fakeflare_g.sav -o $@ \
	-n 100 --plotrfunc --sel_faint=r --fake -i fakeThinFlare.sav \
	--xlabel='r_0\ [\mathrm{mag}]' \
	--ylabel='\mathrm{density}' # \
#	--subsample=1000


#Metal poor G dwarf densities
$(OUTDIR)/HWR_poor_g_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g.sav -o $@ \
	--expd1 --expd2 -n 100 --metal=poor \
	--xmin=0. --xmax=25. --ymin=0. --ymax=30. \
	--xlabel='h_z\ [\mathrm{kpc}]' \
	--ylabel='h_R [\mathrm{kpc}]' #\
#	--subsample=1000
$(OUTDIR)/HWR_poor_g_zfunc.$(EXT): fitDensz.py Makefile plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g.sav -o $@ \
	-n 100 --plotzfunc --metal=poor \
	--xlabel='z\ [\mathrm{kpc}]' \
	--ylabel='\ln \nu(z;R=8\ \mathrm{kpc})' # \
#	--subsample=1000
$(OUTDIR)/HWR_poor_g_rfunc.$(EXT): fitDensz.py Makefile plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g.sav -o $@ \
	-n 100 --plotrfunc --metal=poor \
	--xlabel='r_0\ [\mathrm{mag}]' \
	--ylabel='\mathrm{density}' # \
#	--subsample=1000
#TwoVertical model
$(OUTDIR)/HWR_poor_g_twovertical_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g_twovertical.sav -o $@ \
	-n 100 --plotrfunc --metal=poor \
	--model=twovertical \
	--xlabel='r_0\ [\mathrm{mag}]' \
	--ylabel='\mathrm{density}' # \
#	--subsample=1000
#Flare model
$(OUTDIR)/HWR_poor_g_flare_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g_flare.sav -o $@ \
	-n 100 --plotrfunc --metal=poor \
	--model=flare \
	--xlabel='r_0\ [\mathrm{mag}]' \
	--ylabel='\mathrm{density}' # \
#	--subsample=1000
#just bright
$(OUTDIR)/HWR_poor_gbright_rfunc.$(EXT): fitDensz.py Makefile plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_gbright.sav -o $@ \
	-n 100 --plotrfunc --bright --metal=poor \
	--xlabel='r_0\ [\mathrm{mag}]' \
	--ylabel='\mathrm{density}' # \
#	--subsample=1000

#Metal rich G dwarf densities, flare
$(OUTDIR)/flare_rich_g_hRhz.$(EXT): fitDensz.py Makefile
	$(PYTHON) fitDensz.py flare_dens_rich_g.sav -o $@ \
	--expd1 --expd2 -n 100 --model=flare \
	--xmin=0. --xmax=25. --ymin=0. --ymax=30. \
	--xlabel='h_z\ [\mathrm{kpc}]' \
	--ylabel='h_R [\mathrm{kpc}]'

#Metal poor G dwarf densities, flare
$(OUTDIR)/flare_poor_g_hRhz.$(EXT): fitDensz.py Makefile
	$(PYTHON) fitDensz.py flare_dens_poor_g.sav -o $@ \
	--expd1 --expd2 -n 100 --metal=poor --model=flare \
	--xmin=0. --xmax=25. --ymin=0. --ymax=30. \
	--xlabel='h_z\ [\mathrm{kpc}]' \
	--ylabel='h_R [\mathrm{kpc}]'


#Metal rich G dwarf velocities
$(OUTDIR)/HWR_rich_g_sR.$(EXT): fitSigz.py Makefile
	$(PYTHON) fitSigz.py HWR_rich_g.sav -o $@ --expd1 --expd2 -n 10000 \
	--xmin=15. --xmax=25. --ymin=0. --ymax=30. \
	--xlabel='\sigma_z(z=500\ \mathrm{pc})\ [\mathrm{km\ s}^{-1}]' \
	--ylabel='R_\sigma\ [\mathrm{kpc}]'

$(OUTDIR)/HWR_rich_g_spbad.$(EXT): fitSigz.py Makefile
	$(PYTHON) fitSigz.py HWR_rich_g.sav -o $@ --expd1 -n 10000 \
	--xmin=15. --xmax=25. --ymin=0. --ymax=.05 --d2=0 \
	--xlabel='\sigma_z(z=500\ \mathrm{pc})\ [\mathrm{km\ s}^{-1}]' \
	--ylabel='P_{\mathrm{bad}}'

$(OUTDIR)/HWR_rich_g_func.$(EXT): fitSigz.py Makefile
	$(PYTHON) fitSigz.py HWR_rich_g.sav -o $@ -n 10000 \
	--plotfunc --plotnsamples=1000 \
	--xlabel='|z|\ [\mathrm{kpc}]' \
	--ylabel='\sigma_z\ [\mathrm{km\ s}^{-1}]' \
	--ymin=0. --ymax=50.

#Metal poor G dwarfs velocity
$(OUTDIR)/HWR_poor_g_sR.$(EXT): fitSigz.py Makefile
	$(PYTHON) fitSigz.py HWR_poor_g.sav -o $@ --expd1 --expd2 -n 10000 \
	--xmin=35. --xmax=45. --ymin=0. --ymax=30. \
	--xlabel='\sigma_z(z=500\ \mathrm{pc})\ [\mathrm{km\ s}^{-1}]' \
	--ylabel='R_\sigma\ [\mathrm{kpc}]' --metal=poor

$(OUTDIR)/HWR_poor_g_spbad.$(EXT): fitSigz.py Makefile
	$(PYTHON) fitSigz.py HWR_poor_g.sav -o $@ --expd1 -n 10000 \
	--xmin=35. --xmax=45. --ymin=0. --ymax=.05 --d2=0 \
	--xlabel='\sigma_z(z=500\ \mathrm{pc})\ [\mathrm{km\ s}^{-1}]' \
	--ylabel='P_{\mathrm{bad}}' --metal=poor

$(OUTDIR)/HWR_poor_g_func.$(EXT): fitSigz.py Makefile
	$(PYTHON) fitSigz.py HWR_poor_g.sav -o $@ -n 10000 \
	--plotfunc --plotnsamples=1000 \
	--xlabel='|z|\ [\mathrm{kpc}]' \
	--ylabel='\sigma_z\ [\mathrm{km\ s}^{-1}]' \
	--ymin=0. --ymax=50. --metal=poor

#Metal rich K dwarf velocities
$(OUTDIR)/HWR_rich_k_sR.$(EXT): fitSigz.py Makefile
	$(PYTHON) fitSigz.py HWR_rich_k.sav -o $@ --expd1 --expd2 -n 10000 \
	--xmin=15. --xmax=25. --ymin=0. --ymax=100. \
	--xlabel='\sigma_z(z=500\ \mathrm{pc})\ [\mathrm{km\ s}^{-1}]' \
	--ylabel='R_\sigma\ [\mathrm{kpc}]' --sample=K

$(OUTDIR)/HWR_rich_k_spbad.$(EXT): fitSigz.py Makefile
	$(PYTHON) fitSigz.py HWR_rich_k.sav -o $@ --expd1 -n 10000 \
	--xmin=15. --xmax=25. --ymin=0. --ymax=.05 --d2=0 \
	--xlabel='\sigma_z(z=500\ \mathrm{pc})\ [\mathrm{km\ s}^{-1}]' \
	--ylabel='P_{\mathrm{bad}}' --sample=K

$(OUTDIR)/HWR_rich_k_func.$(EXT): fitSigz.py Makefile
	$(PYTHON) fitSigz.py HWR_rich_k.sav -o $@ -n 10000 \
	--plotfunc --plotnsamples=1000 \
	--xlabel='|z|\ [\mathrm{kpc}]' \
	--ylabel='\sigma_z\ [\mathrm{km\ s}^{-1}]' \
	--ymin=0. --ymax=50. --sample=K

#Metal poor K dwarfs velocity
$(OUTDIR)/HWR_poor_k_sR.$(EXT): fitSigz.py Makefile
	$(PYTHON) fitSigz.py HWR_poor_k.sav -o $@ --expd1 --expd2 -n 10000 \
	--xmin=35. --xmax=45. --ymin=0. --ymax=100. \
	--xlabel='\sigma_z(z=500\ \mathrm{pc})\ [\mathrm{km\ s}^{-1}]' \
	--ylabel='R_\sigma\ [\mathrm{kpc}]' --metal=poor --sample=K

$(OUTDIR)/HWR_poor_k_spbad.$(EXT): fitSigz.py Makefile
	$(PYTHON) fitSigz.py HWR_poor_k.sav -o $@ --expd1 -n 10000 \
	--xmin=35. --xmax=45. --ymin=0. --ymax=.1 --d2=0 \
	--xlabel='\sigma_z(z=500\ \mathrm{pc})\ [\mathrm{km\ s}^{-1}]' \
	--ylabel='P_{\mathrm{bad}}' --metal=poor --sample=K

$(OUTDIR)/HWR_poor_k_func.$(EXT): fitSigz.py Makefile
	$(PYTHON) fitSigz.py HWR_poor_k.sav -o $@ -n 10000 \
	--plotfunc --plotnsamples=1000 \
	--xlabel='|z|\ [\mathrm{kpc}]' \
	--ylabel='\sigma_z\ [\mathrm{km\ s}^{-1}]' \
	--ymin=0. --ymax=50. --metal=poor --sample=K



##EXTRA

#Metal rich, isothermal model, G dwarfs, velocities
$(OUTDIR)/isotherm_rich_g_sR.$(EXT): fitSigz.py Makefile
	$(PYTHON) fitSigz.py isotherm_rich_g.sav -o $@ --expd1 --expd2 \
	-n 10000 --d2=2 --model=isotherm \
	--xmin=15. --xmax=25. --ymin=0. --ymax=30. \
	--xlabel='\sigma_z(z=500\ \mathrm{pc})\ [\mathrm{km\ s}^{-1}]' \
	--ylabel='R_\sigma\ [\mathrm{kpc}]'

$(OUTDIR)/isotherm_rich_g_spbad.$(EXT): fitSigz.py Makefile
	$(PYTHON) fitSigz.py isotherm_rich_g.sav -o $@ --expd1 -n 10000 \
	--xmin=15. --xmax=25. --ymin=0. --ymax=.05 --d2=0 \
	--xlabel='\sigma_z(z=500\ \mathrm{pc})\ [\mathrm{km\ s}^{-1}]' \
	--ylabel='P_{\mathrm{bad}}' --model=isotherm

$(OUTDIR)/isotherm_rich_g_func.$(EXT): fitSigz.py Makefile
	$(PYTHON) fitSigz.py isotherm_rich_g.sav -o $@ -n 10000 \
	--plotfunc --plotnsamples=1000 --model=isotherm \
	--xlabel='|z|\ [\mathrm{kpc}]' \
	--ylabel='\sigma_z\ [\mathrm{km\ s}^{-1}]' \
	--ymin=0. --ymax=50.
