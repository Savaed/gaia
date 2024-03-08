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

Detection of exoplanets is possible using *light curves* (time series of stellar brightness), for which so-called *TCEs* (Threshold Cross Events) are identified, i.e. repeated signals associated with *period* (the amount of time between successive signals), *duration* (the time from the beginning to the end of the signal) and *epoch* (the time of first observation). TCEs can represent transits of exoplanets, but can also be the result of other events such as [binary systems](https://en.wikipedia.org/wiki/Binary_star) or measurement errors. Such a transit is shown above. In addition to the primary transit (when the object obscures the star), a secondary transit (when the star obscures the object) is also distinguished. The purpose of training the models is to teach them to recognise transits of exoplanets and to distinguish them from other phenomena.

### Light curve preprocessing

Data processing mainly involves processing light curves and creating *local* and *global views* from them. The local view contains information about the averaged brightness of the star during the transit and is created separately for each TCE. The global view, also created for each TCE, covers the entire TCE period.

Views are created for:
  - stellar brightness,
  - even and odd transits separately,
  - light centroids (centres of the light source),
  - secondary transits.

Other data processing steps include standardisation of time series and stellar parameters, among others.

Raw Kepler time series downloaded from the MAST archive contain observations divided into several periods (quarters).

![kepler full mission light curve](https://github.com/Savaed/gaia/blob/main/docs/img/kepler-90-full-mission.png)

The plot above clearly shows the transits of two planets: Kepler-90 h - the largest planet in the system with the greatest transit depth, and Kepler-90 g - the second largest planet in size and transit depth.

![kepler full quarter 4 light curve](https://github.com/Savaed/gaia/blob/main/docs/img/kepler-90-q4.png)

It can be noticed that the brightness of the star even far away from transits is not constant - brightness fluctuations result from the natural variability of the observed star. This small variability significantly complicates exoplanet detection and must therefore be removed. In order to remove noise a normalization curve is fitted for each series and the transits are interpolated linearly. Such interpolation allows the curve to be fitted only to noise without removing any changes caused by planets or other objects.

![kepler normalization](https://github.com/Savaed/gaia/blob/main/docs/img/kepler-90-q4-normalization.png)

Then the original series is divided by the normalization curve. The result of such an operation is a light curve with noise removed and transits preserved, as shown below.

![kepler normalized](https://github.com/Savaed/gaia/blob/main/docs/img/kepler-90-q4-flattened.png)

The final stage of processing is the creation of views, global and local. Both are *phase-folded*, which means that all periods of the detected TCE are combined into one curve in which the detected event is centered and values are averged.

![kepler views](https://github.com/Savaed/gaia/blob/main/docs/img/kepler-90-views.png)

Centroid series are processed in a similar way, but only a local and global view is created for them. Even and odd transits are extracted from the normalized curves and local views of star brightness are created from them. A local view is also created for the second transit brightness.

## General system architecture

![gaia system architecture](https://github.com/Savaed/gaia/blob/main/docs/img/gaia_system_architecture_and_data_flow.png)

## How to

To run this project locally ensure you have **Python 3.11 or newer** installed on your machine:
```sh
$ python --version
# Python 3.11.8
```

Then install [poetry](https://python-poetry.org/) and optionally [tox](https://tox.wiki/en/stable/):
  ```
  $ pip install tox poetry
  ```

Install project dependencies:
```
$ poetry install
```

### Download data

GAIA allows for efficient (asynchronous) downloading of stellar time series, TCE scalar values and stellar parameters from official [NASA](https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html) and [MAST](https://archive.stsci.edu/missions-and-data/kepler/kepler-bulk-downloadsnasa) archives via REST API. The script implements mechanisms to retry the download after error occures and allows to stop and resume at any time without losing download progress. The downloaded data is automatically saved in the specified local location as `.fits` files.
To download data locally, run following script from top level directory:

```sh
$ python -m gaia.scripts.download_data
```

> **NOTE** The amount of data is significant (approx. 280 000 files, 120GB).

### Preprocess data

Data preprocessing allows to extract relevant information from raw files and, in the case of *TCE*, to combine values ​​from several sources into one file. It also changes the data format to one that is easier to further process (default to `.parquet`). The size of the data is also reduced (from 120GB to over 20GB). To preprocess raw data locally run:

```sh
$ python -m gaia.scripts.preprocess_data
```

### Visualize data

The interactive visualizations allow for graphical representation of both *TCE* and stellar scalar data as well as stellar time series. Implemented as a website, the dashboard provides basic operations on charts (filters, zooming, moving plots, selecting specific observations, etc.). To open dashboard web page on *localhost* run:

```sh
$ python -m gaia.scripts.run_dashboard
```

> **NOTE** For the dashboard to work properly, it is **required to pre-process the data** and change it format from `.fits` to `.parquet` using the `preprocess_data.py` script.

### Create final features

Data processing implemented as a PySpark pipeline allows to transform interim data into final features that can be used to train deep learning models. The final data is in the format `.tfrecords`. This operation includes: removing noise from light curves, creating appropriate local and global views, spliting the data into training, validation and test sets, normalizing observations. To create features locally run:

```sh
$ python -m gaia.scripts.create_features
```

or

```sh
$ python -m gaia.scripts.submit_spark_create_features_job
```

### Fit deep learning models

**Not implemented yet.**

## Step-by-step GCP guide

Make sure you have Docker installed on your machine: `docker --version`.

To run part of this project on [Google Cloud Platform (GCP)](https://cloud.google.com/?hl=en), the following steps are required:

  1. Install [gcloud](https://cloud.google.com/sdk/docs/install).

  2. Create a new GCP project or select exiting one:
  ```
  $ gcloud init
  ```

  3. Set up local Application Default Credentials (*ADC*) and create a credential JSON file:
  ```
  $ gcloud auth application-default login
  ```
  Learn more about *ADC* [here](https://cloud.google.com/docs/authentication/application-default-credentials) and [here](https://cloud.google.com/docs/authentication/provide-credentials-adc).

  4. Set up `GOOGLE_APPLICATION_CREDENTIALS` environment variable to provide the location of a credential JSON file. This environment variable is used by GCP Python Client Libraries to communicate with GCP services.
  5. [Enable billing](https://cloud.google.com/billing/docs/how-to/modify-project#enable_billing_for_a_project) for a selected project.
  6. Enable *Service Usage API*.
  7. For data storage enable *Google Cloud Storage API*.
  8. For PySpark data processing enable *Google Cloud Dataproc API*.
  9. For DL models training enable *Google Cloud Vertex AI API*.
  10. For PySpark and DL models custom containers enable *Artifact Registry API*.
  11. Configure Docker to use the Google Cloud CLI to authenticate requests to Artifact Registry in specified region (e.g. *europe-central2*):
  ```sh
  $ gcloud auth configure-docker europe-central2-docker.pkg.dev
  ```

  or add following lines to local Docker config (default location is `~/.docker/config.json`):
  ```json
  "credHelpers": {
    "europe-central2-docker.pkg.dev": "gcloud"
  }
  ```
  12. [Create](https://cloud.google.com/artifact-registry/docs/docker/store-docker-container-images) Artifact Registry container repository.
  13. Build and push Docker image. The Docker image name **MUST** have a format `{region}-docker.pkg.dev/{GCP_project_ID}/{repo}/{image_name}:{tag}` e.g.:
  ```sh
  docker build -t europe-central2-docker.pkg.dev/project-132/test/test-dataproc:tag1 . \
  docker push europe-central2-docker.pkg.dev/project-132/test/test-dataproc:tag1
  ```
  14. Change [Dataproc config](https://github.com/Savaed/gaia/blob/26724d3576ba63355867c76ecc3115ddd973cadb/configs/create_features/kepler_gcp.yaml#L134C1-L134C12) in `gaia/configs/create_features/kepler_gcp.yaml` to use your image.
