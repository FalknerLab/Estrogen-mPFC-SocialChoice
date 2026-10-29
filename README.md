This repository contains the main analysis pipeline supporting the figures in the manuscript:

**Estrogen remodels large-scale cortical dynamics to promote behavioral persistence**<br>
Meenakshi M. Asokan, Alissa Le, Dexter Tsin, Lucy Sirrs, Madeline Cramer, Stefan N. Oline, Forrest D. Rogers, Annegret L. Falkner.

### Overview

**`UMAP_density_map_and_clusters_template.m`**
Behavioral features embedded in a 2D UMAP manifold, with a smoothed occupancy map 
over the manifold and the boundaries of the behavioral clusters overlaid.

**`fit_hmm.py`**
Hidden Markov model fit to binned population spiking, giving the neural state sequences.

**`Energy_landscape_Kramers_template.m`**
Neural state dwell times converted to attractor well depths via Kramers' escape-rate relation
and rendered as a metastable energy landscape.

**`LDA_distance_and_decoding_template.m`**
Population spike counts projected onto a linear discriminant space that maximally
separates behavioral states, and cross-validated decoding of state from the same population.

**`Effective_dimensionality_bootstrap_template.m`**
Effective dimensionality of the neural manifold occupied during each behavioral state,
measured as the participation ratio of the PCA eigenvalue spectrum of the bins labeled with that state.

**`Lifetime_sparseness_tuning_template.m`**
Single-unit tuning across the three behavioral states, and computation of 
a lifetime sparseness value per unit reflecting tuning selectivity.

**`Intrinsic_timescale_autocorr_template.m`**
Single-unit intrinsic timescale from the autocorrelation of spontaneous spiking, with
single- and multi-exponential fits.

**`Functional_connectivity_crosscov_template.m`**
Pairwise functional connectivity from simultaneously recorded spike trains using spike cross-covariance, 
graph visualization, and connectivity broken down both by signal correlation and by ensemble membership.

**`Two_node_mutual_inhibition_model_template.m`**
Two-population mutual-inhibition mean-field model with its recurrent weights set from the
measured connection probabilities.

**`utils/`**
Helper functions called by the scripts above and shipped with them.
