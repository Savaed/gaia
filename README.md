# GAIA

**An exoplanet detection system based on stellar brightness time series and stellar parameters using convolutional neural networks.**

> **This project is still under active development.**

[![workflow](https://github.com/Savaed/gaia/actions/workflows/main.yaml/badge.svg)](https://github.com/Savaed/gaia/actions)
[![codecov](https://codecov.io/gh/Savaed/gaia/branch/main/graph/badge.svg?token=D482CSZ7MJ)](https://codecov.io/gh/Savaed/gaia)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://docs.astral.sh/ruff/)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)

> **GAIA allows the downloading, processing and visualisation of stellar brightness time series and stellar parameters from missions such as Kepler/K2 or TESS. A separate module allows the training of deep learning models and the detection of exoplanet candidates. The whole system can be run locally or partially on the Google Cloud Platform.**


## How it works

![Stellar transit](https://github.com/Savaed/gaia/blob/main/docs/img/nasa-transit.gif)
<p align="center">Figure. 1. Visualisation of changes in stellar brightness during transit</p>

Detection of exoplanets is possible using *light curves* (time series of stellar brightness), for which so-called *TCEs* (Threshold Cross Events) are identified, i.e. repeated signals associated with *period* (the amount of time between successive signals), *duration* (the time from the beginning to the end of the signal) and *epoch* (the time of first observation). TCEs can represent transits of exoplanets, but can also be the result of other events such as [binary systems](https://en.wikipedia.org/wiki/Binary_star) or measurement errors. Such a transit is shown in Figure 1. In addition to the primary transit (when the object obscures the star), a secondary transit (when the star obscures the object) is also distinguished. The purpose of training the models is to teach them to recognise transits of exoplanets and to distinguish them from other phenomena.

### Light curve preprocessing

Data processing mainly involves processing light curves and creating *local* and *global views* from them. The local view contains information about the averaged brightness of the star during the transit and is created separately for each TCE. The global view, also created for each TCE, covers the entire TCE period.

Views are created for:
  - stellar brightness,
  - even and odd transits separately,
  - light centroids (centres of the light source),
  - secondary transits.

Other data processing steps include standardisation of time series and stellar parameters, among others.

Raw Kepler time series downloaded from the MAST archive contain observations divided into several periods (quarters).

XXX
<p align="center">Figure. 2. *Raw light curves for Kepler-90 system. Data observed over entire mission - about 4 years.*</p>

Fig. 2. clearly shows the transits of two planets: Kepler-90 h - the largest planet in the system with the greatest transit depth, and Kepler-90 g - the second largest planet in size and transit depth.

XXX
<p align="center">Figure. 3. Raw light curves for Kepler-90 g transit</p>

It can be noticed that the brightness of the star even far away from transits is not constant - brightness fluctuations result from the natural variability of the observed star. This small variability significantly complicates exoplanet detection and must therefore be removed. In order to remove noise a normalization curve is fitted for each series and the transits are interpolated linearly. Such interpolation allows the curve to be fitted only to noise without removing any changes caused by planets or other objects. An example of the described fitting is shown in Fig. 4.

XXX
<p align="center">Figure. 4. Normalization curve for Kepler-90 g light curve segment. Transit is linearly interpolated</p>

Then the original series is divided by the normalization curve. The result of such an operation is a light curve with noise removed and transits preserved, as shown in Fig. 5.

XXX
<p align="center">Figure. 5. Flettened light curve for Kepler-90 g light curve segment</p>

The final stage of processing is the creation of views, global and local. Both are *phase-folded*, which means that all periods of the detected TCE are combined into one curve in which the detected event is centered and values are averged.

XXX
<p align="center">Figure. 6. Global light curve view (left) and local view (right) for Kepler-90 g.</p>

Centroid series are processed in a similar way, but only a local and global view is created for them. Even and odd transits are extracted from the normalized curves and local views of star brightness are created from them. A local view is also created for the second transit brightness.

## General system architecture

## Main modules

### Easy download data

### Dashborad and visualizations

### Data processing

### Deep learning models
