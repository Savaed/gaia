import re
from typing import Any

import pytest
import tensorflow.python.framework.errors_impl as tf_errors

from gaia.data.combiners import CombinerError, KeplerFitsCombiner


TEST_FITS_PATHS = ("kepid1-quarter1.fits", "kepid1-quarter2.fits", "kepid2-quarter1.fits")
TEST_FITS_DICT = {"col1": [1, 2, 3]}


@pytest.fixture
def combined_dicts():
    """Return a list of dictionaries which represents combined FITS files for `TEST_FITS_PATHS`."""
    return [
        ("kepid1", {"quarter1": TEST_FITS_DICT, "quarter2": TEST_FITS_DICT}),
        ("kepid2", {"quarter1": TEST_FITS_DICT}),
    ]


@pytest.fixture
def combiner(tmp_path):
    """Return an instance of `KeplerFitsCombiner` with temporal (for each test) metafile path."""
    instance = KeplerFitsCombiner[Any](
        data_dir="a/b/c",
        id_pattern=r"kepid\d",
        quarter_pattern=r"quarter\d",
    )
    # Make sure that metadata is temporal for each test
    instance._meta_path = tmp_path / "test_meta.txt"
    return instance


@pytest.mark.asyncio
async def test_combine__root_dir_not_found(mocker, combiner):
    """Test that CombinerError is raised when no root data dir found."""
    tf_not_found_error = tf_errors.NotFoundError(None, None, "Test directory not found")
    mocker.patch("gaia.data.combiners.tf.io.gfile.listdir", side_effect=tf_not_found_error)
    mocker.patch("gaia.data.combiners.tf.io.gfile.exists", return_value=False)
    with pytest.raises(CombinerError, match="Root data directory"):
        await anext(combiner.combine())


@pytest.fixture
def root_data_dir_exists(mocker):
    """Mock `tf.io.gfile.exists()` to return `True` for root data directory."""
    mocker.patch("gaia.data.combiners.tf.io.gfile.exists", return_value=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,error_msg",
    [
        (ValueError("Fields x not present in the FITS file"), "Fields .* not present"),
        (KeyError("Extension not found"), ".*Extension not found.*"),
    ],
    ids=["missing_fits_fields", "invalid_fits_hdu_extension"],
)
@pytest.mark.usefixtures("root_data_dir_exists")
async def test_combine__fits_files_read_error(error, error_msg, mocker, combiner):
    """Test that CombinerError is raised when there is an error while reading FITS file(s)."""
    mocker.patch("gaia.data.combiners.tf.io.gfile.listdir", return_value=[TEST_FITS_PATHS[0]])
    mocker.patch("gaia.data.combiners.read_fits_table", side_effect=error)
    with pytest.raises(CombinerError, match=error_msg):
        await anext(combiner.combine())


@pytest.mark.asyncio
@pytest.mark.usefixtures("root_data_dir_exists")
async def test_combine__yield_combined_dicts(mocker, combiner, combined_dicts):
    """Test that properly combined files are yield as dictionaries."""
    mocker.patch("gaia.data.combiners.tf.io.gfile.listdir", return_value=TEST_FITS_PATHS)
    mocker.patch("gaia.data.combiners.read_fits_table", side_effect=[TEST_FITS_DICT] * 3)
    results = [(kepid, combined_dict) async for kepid, combined_dict in combiner.combine()]
    assert results == combined_dicts


@pytest.fixture(params=["_kepid_pattern", "_quarter_pattern"], ids=["kepid", "quarter"])
def invalid_pattern_combiner(request, combiner):
    """Return an instance of `KeplerFitsCombiner` with invalivd kepid or quarter pattern."""
    setattr(combiner, request.param, re.compile("^invalid_pattern$"))
    return combiner


@pytest.mark.asyncio
@pytest.mark.usefixtures("root_data_dir_exists")
async def test_combine__kepid_or_quarter_prefix_search_error(mocker, invalid_pattern_combiner):
    """Test that CombinerError is raised when there is a search error (for kepid or quarter)."""
    mocker.patch("gaia.data.combiners.tf.io.gfile.listdir", return_value=TEST_FITS_PATHS)
    with pytest.raises(CombinerError, match="Cannot substract quarter prefix or kepid"):
        await anext(invalid_pattern_combiner.combine())


@pytest.fixture
def combiner_with_meta_kepid1(combiner):
    """Return an instance of `KeplerFitsCombiner` with ID 'kepid1' already processed."""
    combiner._meta_path.write_text("kepid1\n")
    return combiner


@pytest.mark.asyncio
@pytest.mark.usefixtures("root_data_dir_exists")
async def test_combine__skip_already_combined_files(mocker, combiner_with_meta_kepid1):
    """Test that files that were already combined are not reprocessed."""
    mocker.patch("gaia.data.combiners.tf.io.gfile.listdir", return_value=TEST_FITS_PATHS)
    mocker.patch("gaia.data.combiners.read_fits_table", side_effect=[TEST_FITS_DICT])
    result = await anext(combiner_with_meta_kepid1.combine())
    assert result == ("kepid2", {"quarter1": TEST_FITS_DICT})


@pytest.mark.asyncio
@pytest.mark.usefixtures("root_data_dir_exists")
async def test_combine__save_metadata_to_file(mocker, combiner):
    """Test that kepids for combined files are saved in the metadata file."""
    mocker.patch("gaia.data.combiners.tf.io.gfile.listdir", return_value=TEST_FITS_PATHS)
    mocker.patch("gaia.data.combiners.read_fits_table", side_effect=[TEST_FITS_DICT] * 3)
    [_ async for _ in combiner.combine()]  # Just process files. We don't care about the results
    saved_meta_kepids = set(combiner._meta_path.read_text().splitlines())
    assert saved_meta_kepids == {"kepid1", "kepid2"}
