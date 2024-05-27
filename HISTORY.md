v0.4.0 / 2024-05-27
-------------------
* Try using jax.random.binomial by @dachengx in https://github.com/XENONnT/appletree/pull/148
* Turn off add_eps_to_hist in NR and ER by @FaroutYLq in https://github.com/XENONnT/appletree/pull/152
* Debug when `bins_type` is not set by @dachengx in https://github.com/XENONnT/appletree/pull/153
* Specifically install `lxml_html_clean` by @dachengx in https://github.com/XENONnT/appletree/pull/157
* Initialize context from backend by @zihaoxu98 in https://github.com/XENONnT/appletree/pull/156
* Add `parameter_alias` to translate parameters in `par_config` by @dachengx in https://github.com/XENONnT/appletree/pull/160
* Allow user to aggressively use memory by @dachengx in https://github.com/XENONnT/appletree/pull/164
* Fix a bug of plotter which contains inf by @zihaoxu98 in https://github.com/XENONnT/appletree/pull/165

New Contributors
* @FaroutYLq made their first contribution in https://github.com/XENONnT/appletree/pull/152

**Full Changelog**: https://github.com/XENONnT/appletree/compare/v0.3.2...v0.4.0


v0.3.2 / 2024-03-06
-------------------
* Remove `scikit-learn` version requirement by @dachengx in https://github.com/XENONnT/appletree/pull/131
* Use trusted publisher because username/password authentication is no longer supported by @dachengx in https://github.com/XENONnT/appletree/pull/132
* Be compatible with `JAX_ENABLE_X64=1` by @dachengx in https://github.com/XENONnT/appletree/pull/134
* Raise more information when file can not be found by @dachengx in https://github.com/XENONnT/appletree/pull/135
* Preserve dtype of results in `multiple_simulations` by @dachengx in https://github.com/XENONnT/appletree/pull/137
* Copy memory from GPU to CPU by @dachengx in https://github.com/XENONnT/appletree/pull/139
* Bug fix when using `force_no_eff` with tuple `data_names` by @dachengx in https://github.com/XENONnT/appletree/pull/141
* Add Gamma, Negative Binomial, and Generalized Poisson Distribution by @dachengx in https://github.com/XENONnT/appletree/pull/145
* Add support for 1D fitting by @dachengx in https://github.com/XENONnT/appletree/pull/144
* Correct the sigmas in TwoHalfNorm by @zihaoxu98 in https://github.com/XENONnT/appletree/pull/143
* Plotter for MCMC diagnostics by @zihaoxu98 in https://github.com/XENONnT/appletree/pull/146


**Full Changelog**: https://github.com/XENONnT/appletree/compare/v0.3.1...v0.3.2


v0.3.1 / 2024-01-12
-------------------
* Smarter SigmaMap handling and needed_parameters by @xzh19980906 in https://github.com/XENONnT/appletree/pull/116
* Move messages of used parameter in `SigmaMap.build` by @dachengx in https://github.com/XENONnT/appletree/pull/117
* Set required_parameter as method of Config by @dachengx in https://github.com/XENONnT/appletree/pull/119
* Optional applying efficiency in `multiple_simulations` by @dachengx in https://github.com/XENONnT/appletree/pull/123
* Fix S1/S2 correction and gas gain when simulating S1/S2PE by @mhliu0001 in https://github.com/XENONnT/appletree/pull/122
* Prevent already cached functions from being changed by @dachengx in https://github.com/XENONnT/appletree/pull/125
* Update docstring to google style by @dachengx in https://github.com/XENONnT/appletree/pull/126
* Update conf, add napoleon by @dachengx in https://github.com/XENONnT/appletree/pull/127
* Small bug fix when no llh_name is used by @dachengx in https://github.com/XENONnT/appletree/pull/129

New Contributors
* @mhliu0001 made their first contribution in https://github.com/XENONnT/appletree/pull/122

**Full Changelog**: https://github.com/XENONnT/appletree/compare/v0.3.0...v0.3.1


v0.3.0 / 2023-08-31
-------------------
* Fix function name to `simulate_weighted_data`, nothing else changed by @dachengx in https://github.com/XENONnT/appletree/pull/99
* Stop jax from preallocating memory by @dachengx in https://github.com/XENONnT/appletree/pull/100
* fix meshgrid binning by @hoetzsch in https://github.com/XENONnT/appletree/pull/101
* Binning is not required by Component by @dachengx in https://github.com/XENONnT/appletree/pull/103
* Upper clipping on binomial randgen in normal approx by @xzh19980906 in https://github.com/XENONnT/appletree/pull/107
* Rename fake maps by @xzh19980906 in https://github.com/XENONnT/appletree/pull/106
* Update fake maps filenames by @dachengx in https://github.com/XENONnT/appletree/pull/108
* Skip mongo DB when finding files by @xzh19980906 in https://github.com/XENONnT/appletree/pull/111
* Raise error if file does not exist by @xzh19980906 in https://github.com/XENONnT/appletree/pull/110
* Rename lce to correction by @dachengx in https://github.com/XENONnT/appletree/pull/109
* Add function to check the usage of configs, check_unused_configs by @dachengx in https://github.com/XENONnT/appletree/pull/112
* Proposal to use pre-commit for continuous integration by @dachengx in https://github.com/XENONnT/appletree/pull/113

New Contributors
* @hoetzsch made their first contribution in https://github.com/XENONnT/appletree/pull/101

**Full Changelog**: https://github.com/XENONnT/appletree/compare/v0.2.3...v0.3.0


v0.2.3 / 2023-05-29
-------------------
* Add pip install user by @dachengx in https://github.com/XENONnT/appletree/pull/96
* Installation with various `extras_require`s for different CUDA support by @dachengx in https://github.com/XENONnT/appletree/pull/97

**Full Changelog**: https://github.com/XENONnT/appletree/compare/v0.2.2...v0.2.3


v0.2.2 / 2023-05-25
-------------------
* Stop using MANIFEST.in, move to a modern way of file system configuration by @dachengx in https://github.com/XENONnT/appletree/pull/94

**Full Changelog**: https://github.com/XENONnT/appletree/compare/v0.2.1...v0.2.2


v0.2.1 / 2023-05-22
-------------------
* Change variables `s1` `s2` to `s1_area` `s2_area` by @dachengx in https://github.com/XENONnT/appletree/pull/86
* Remove duplicated import pd by @xzh19980906 in https://github.com/XENONnT/appletree/pull/87
* Generate number of events in the defined ROI by @dachengx in https://github.com/XENONnT/appletree/pull/88
* Update DOI link by @dachengx in https://github.com/XENONnT/appletree/pull/89
* Loosen requirement after beta phase and specify jax find-links by @dachengx in https://github.com/XENONnT/appletree/pull/90
* Update notebooks by @dachengx in https://github.com/XENONnT/appletree/pull/91
* Add more tests by @dachengx in https://github.com/XENONnT/appletree/pull/92

**Full Changelog**: https://github.com/XENONnT/appletree/compare/v0.2.0...v0.2.1


v0.2.0 / 2023-03-15
-------------------
* Add NESTv2 yields parameterization and literature constrain by @dachengx in https://github.com/XENONnT/appletree/pull/82
* Add uncertainty on electron lifetime by @dachengx in https://github.com/XENONnT/appletree/pull/83
* Add NESTv2 NR parameters of quanta distribution's width by @dachengx in https://github.com/XENONnT/appletree/pull/85

**Full Changelog**: https://github.com/XENONnT/appletree/compare/v0.1.0...v0.2.0


v0.1.0 / 2023-02-12
-------------------
* Add corner as dependency by @dachengx in https://github.com/XENONnT/appletree/pull/55
* Add .h5 backend to store MCMC results by @dachengx in https://github.com/XENONnT/appletree/pull/57
* Add document by @xzh19980906 in https://github.com/XENONnT/appletree/pull/61
* More docs by @xzh19980906 in https://github.com/XENONnT/appletree/pull/62
* Deduce and compile codes to generate Ly & Qy curve by @dachengx in https://github.com/XENONnT/appletree/pull/60
* Convert PDF spectrum to CDF by default by @xzh19980906 in https://github.com/XENONnT/appletree/pull/72
* Add R dimension for template generation by @xzh19980906 in https://github.com/XENONnT/appletree/pull/74
* Use specific config for certain llh by @dachengx in https://github.com/XENONnT/appletree/pull/78
* Config can read map and assign itself a mapping function by @dachengx in https://github.com/XENONnT/appletree/pull/79
* Stricter needed parameters check and save meta data to samples by @dachengx in https://github.com/XENONnT/appletree/pull/80
* Irregular binning option in likelihood by @xzh19980906 in https://github.com/XENONnT/appletree/pull/81

**Full Changelog**: https://github.com/XENONnT/appletree/compare/v0.0.0...v0.1.0


v0.0.0 / 2022-10-13
-------------------
* Add more notebooks by @dachengx in https://github.com/XENONnT/appletree/pull/52
* Introduce numpyro as dependency by @dachengx in https://github.com/XENONnT/appletree/pull/53
* Try upload to pypi by @dachengx in https://github.com/XENONnT/appletree/pull/54

**Full Changelog**: https://github.com/XENONnT/appletree/commits/v0.0.0
