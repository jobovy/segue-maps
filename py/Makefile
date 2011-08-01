OUTDIR=../figs
EXT=png

#Sig_z(z) for both metal rich and metal poor

$(OUTDIR)/HWR_rich_poor_g_func.$(EXT): plotSigzFunc.py
	$(PYTHON) plotSigzFunc.py HWR_rich_g.sav HWR_poor_g.sav \
	-o $@

#Metal rich G dwarf densities
$(OUTDIR)/HWR_rich_g_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g.sav -o $@ \
	-n 100
$(OUTDIR)/HWR_rich_g_mh_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_mh.sav -o $@ \
	-n 100 --metropolis

#Binned color distribution
$(OUTDIR)/HWR_rich_g_colorbinned_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_colorbinned.sav -o $@ \
	-n 100 --colordist=binned #\
#	--subsample=1000

#l,b ranges
$(OUTDIR)/HWR_rich_g_lcen180_bcen90_lbdr30_hRhz.$(EXT): fitDensz.py Makefile \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_lcen180_bcen90_lbdr30.sav -o $@ \
	-n 100 --lcen=180. --bcen=90 --lbdr=30
$(OUTDIR)/HWR_rich_g_lcen180_bcen0_lbdr30_hRhz.$(EXT): fitDensz.py Makefile \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_lcen180_bcen0_lbdr30.sav -o $@ \
	-n 100 --lcen=180. --bcen=0 --lbdr=30
$(OUTDIR)/HWR_rich_g_lcen45_bcen-23_lbdr20_hRhz.$(EXT): fitDensz.py Makefile \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_lcen45_bcen-23_lbdr20.sav -o $@ \
	-n 100 --lcen=45. --bcen=-23 --lbdr=20

#Min KS
$(OUTDIR)/HWR_rich_g_minks0.1_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_minks_0.1.sav -o $@ \
	-n 100 --minks=0.1
$(OUTDIR)/HWR_rich_g_minks-2_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_minks_-2.sav -o $@ \
	-n 100 --minks=-2
#Min plateSN_r
$(OUTDIR)/HWR_rich_g_minplatesn70_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_minplatesn_70.sav -o $@ \
	-n 100 --minplatesn=70
$(OUTDIR)/HWR_rich_g_minplatesn85_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_minplatesn_85.sav -o $@ \
	-n 100 --minplatesn=85
$(OUTDIR)/HWR_rich_g_minplatesn70_maxplatesn85hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_minplatesn_70_maxplatesn85.sav -o $@ \
	-n 100 --minplatesn_faint=70 --maxplatesn_faint=85 \
	--minplatesn_bright=0.
$(OUTDIR)/HWR_rich_g_minplatesn10_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_minplatesn_10.sav -o $@ \
	-n 100 --minplatesn=10

#TwoVertical model
$(OUTDIR)/HWR_rich_g_twovertical_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_twovertical.sav -o $@ \
	-n 100 --plotrfunc \
	--model=twovertical
#Flare model
$(OUTDIR)/HWR_rich_g_flare_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_flare.sav -o $@ \
	-n 100 --plotrfunc \
	--model=flare
#Min KS
$(OUTDIR)/HWR_rich_g_flare_minks0.1_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_flare_minks_0.1.sav -o $@ \
	-n 100 --minks=0.1 --model=flare
#Tied Flare model
$(OUTDIR)/HWR_rich_g_tiedflare_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_tiedflare.sav -o $@ \
	-n 100 --plotrfunc \
	--model=tiedflare

#just bright
$(OUTDIR)/HWR_rich_gbright_rfunc.$(EXT): fitDensz.py Makefile plotData.py \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_gbright.sav -o $@ \
	-n 100 --plotrfunc --bright
$(OUTDIR)/TwoVertical_rich_gbright_rfunc.$(EXT): fitDensz.py Makefile \
	segueSelect.py
	$(PYTHON) fitDensz.py TwoVertical_dens_rich_gbright.sav -o $@ \
	-n 100 --plotrfunc --bright --model=twovertical
$(OUTDIR)/Flare_rich_gbright_rfunc.$(EXT): fitDensz.py Makefile \
	segueSelect.py
	$(PYTHON) fitDensz.py Flare_dens_rich_gbright.sav -o $@ \
	-n 100 --plotrfunc --bright --model=flare
$(OUTDIR)/DblExp_rich_gbright_rfunc.$(EXT): fitDensz.py Makefile plotData.py \
	segueSelect.py
	$(PYTHON) fitDensz.py DblExp_dens_rich_gbright.sav -o $@ \
	-n 100 --plotrfunc --bright --model=dblexp

#Fake data
$(OUTDIR)/HWR_rich_fakedblexp_g_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_fakedblexp_g.sav -o $@ \
	-n 100 --plotrfunc --sel_faint=r --fake -i fakeThinDblExp.sav
$(OUTDIR)/HWR_rich_fakedblexp_gbright_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_fakedblexp_gbright.sav -o $@ \
	-n 100 --plotrfunc --sel_faint=r --fake -i fakeThinDblExp.sav
$(OUTDIR)/Flare_rich_fakedblexp_g_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py \
	segueSelect.py
	$(PYTHON) fitDensz.py Flare_dens_rich_fakedblexp_g.sav -o $@ \
	-n 100 --plotrfunc --sel_faint=r --fake -i fakeThinDblExp.sav \
	--model=flare
$(OUTDIR)/Flare_rich_fakeflare_g_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py \
	segueSelect.py
	$(PYTHON) fitDensz.py Flare_dens_rich_fakedblexp_g.sav -o $@ \
	-n 100 --plotrfunc --sel_faint=r --fake -i fakeThinFlare.sav \
	--model=flare
$(OUTDIR)/HWR_rich_fakeflare_g_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_fakeflare_g.sav -o $@ \
	-n 100 --plotrfunc --sel_faint=r --fake -i fakeThinFlare.sav

#Metal poor G dwarf densities
$(OUTDIR)/HWR_poor_g_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g.sav -o $@ \
	-n 100 --metal=poor
$(OUTDIR)/HWR_poor_g_zfunc.$(EXT): fitDensz.py Makefile plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g.sav -o $@ \
	-n 100 --plotzfunc --metal=poor
$(OUTDIR)/HWR_poor_g_rfunc.$(EXT): fitDensz.py Makefile plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g.sav -o $@ \
	-n 100 --plotrfunc --metal=poor
#l,b ranges
$(OUTDIR)/HWR_poor_g_lcen180_bcen90_lbdr30_hRhz.$(EXT): fitDensz.py Makefile \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g_lcen180_bcen90_lbdr30.sav -o $@ \
	-n 100 --lcen=180. --bcen=90 --lbdr=30 --metal=poor
$(OUTDIR)/HWR_poor_g_lcen180_bcen0_lbdr30_hRhz.$(EXT): fitDensz.py Makefile \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g_lcen180_bcen0_lbdr30.sav -o $@ \
	-n 100 --lcen=180. --bcen=0 --lbdr=30 --metal=poor
$(OUTDIR)/HWR_poor_g_lcen45_bcen-23_lbdr20_hRhz.$(EXT): fitDensz.py Makefile \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g_lcen45_bcen-23_lbdr20.sav -o $@ \
	-n 100 --lcen=45. --bcen=-23 --lbdr=20 --metal=poor
#min KS
$(OUTDIR)/HWR_poor_g_minks0.1_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g_minks_0.1.sav -o $@ \
	-n 100 --minks=0.1 --metal=poor
$(OUTDIR)/HWR_poor_g_minks0_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g_minks_0.sav -o $@ \
	-n 100 --minks=0 --metal=poor
#TwoVertical model
$(OUTDIR)/HWR_poor_g_twovertical_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g_twovertical.sav -o $@ \
	-n 100 --plotrfunc --metal=poor \
	--model=twovertical
#Flare model
$(OUTDIR)/HWR_poor_g_flare_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g_flare.sav -o $@ \
	-n 100 --plotrfunc --metal=poor \
	--model=flare
#Min KS
$(OUTDIR)/HWR_poor_g_flare_minks0.1_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g_flare_minks_0.1.sav -o $@ \
	-n 100 --minks=0.1 --model=flare --metal=poor
#just bright
$(OUTDIR)/HWR_poor_gbright_rfunc.$(EXT): fitDensz.py Makefile plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_gbright.sav -o $@ \
	-n 100 --plotrfunc --bright --metal=poor
#just bright
$(OUTDIR)/HWR_poor_gfaint_rfunc.$(EXT): fitDensz.py Makefile plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_gfaint.sav -o $@ \
	-n 100 --plotrfunc --faint --metal=poor

#Metal rich G dwarf densities, flare
$(OUTDIR)/flare_rich_g_hRhz.$(EXT): fitDensz.py Makefile
	$(PYTHON) fitDensz.py flare_dens_rich_g.sav -o $@ \
	-n 100 --model=flare

#Metal poor G dwarf densities, flare
$(OUTDIR)/flare_poor_g_hRhz.$(EXT): fitDensz.py Makefile
	$(PYTHON) fitDensz.py flare_dens_poor_g.sav -o $@ \
	-n 100 --metal=poor --model=flare

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
