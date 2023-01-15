from unittest.mock import Mock

import pytest

from gaia.data.combiners import CombinerError, FitsCombiner


class TestKeplerFitsCombiner:
    SINGLE_FITS_DICT = {"col1": 1, "col2": 2}

    @pytest.fixture
    def combiner(self):
        """Return `gaia.data.combiners.KeplerFitsCombiner` instance."""
        return FitsCombiner(r"(?=.*)\d$")

    @pytest.fixture
    def paths(self):
        """Return FITS file paths."""
        return ["path1", "path2", "path3"]

    @pytest.mark.asyncio
    async def test_combine__no_quarter_extracted(self, combiner, paths):
        combiner._quarter_pattern = Mock(search=Mock(side_effect=[None] * len(paths)))
        with pytest.raises(CombinerError, match="Quarter prefix extraction failed for all paths"):
            await combiner.combine(paths)

    @pytest.mark.asyncio
    async def test_combine__no_file_read(self, combiner, paths, mocker):
        """Test check whether CombinerError is raised when no FITS file can be read."""
        mocker.patch("gaia.data.combiners.tf.io.gfile.glob", return_value=paths)
        mocker.patch(
            "gaia.data.combiners.read_fits_table",
            side_effect=[Exception("error")] * len(paths),
        )
        with pytest.raises(CombinerError, match="File reading failed for all paths"):
            await combiner.combine(paths)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "prefs,read_outputs,expected",
        [
            (
                [AttributeError(), "2", "3"],
                [SINGLE_FITS_DICT, SINGLE_FITS_DICT],
                {"2": SINGLE_FITS_DICT, "3": SINGLE_FITS_DICT},
            ),
            (
                [AttributeError(), "2", "3"],
                [SINGLE_FITS_DICT, Exception()],
                {"2": SINGLE_FITS_DICT},
            ),
            (
                ["1", "2", "3"],
                [SINGLE_FITS_DICT, Exception(), SINGLE_FITS_DICT],
                {"1": SINGLE_FITS_DICT, "3": SINGLE_FITS_DICT},
            ),
        ],
        ids=["prefix_error", "prefix_and_reading_error", "reading_error"],
    )
    async def test_combine__some_quarters_and_files_not_processed(
        self,
        prefs,
        read_outputs,
        expected,
        combiner,
        paths,
        mocker,
    ):
        """Test check whether a correct dictionary is return even if not all files were read."""
        mocker.patch("gaia.data.combiners.tf.io.gfile.glob", return_value=paths)
        combiner._quarter_pattern = Mock(**{"search.return_value.group.side_effect": prefs})
        mocker.patch("gaia.data.combiners.read_fits_table", side_effect=read_outputs)
        result = await combiner.combine(paths)
        assert result == expected

    @pytest.mark.asyncio
    async def test_combine__return_dictionary_no_error(self, combiner, mocker, paths):
        """Test check whether the correct dictionary is returned when there is no error."""
        mocker.patch("gaia.data.combiners.tf.io.gfile.glob", return_value=paths)
        mocker.patch(
            "gaia.data.combiners.read_fits_table",
            side_effect=[self.SINGLE_FITS_DICT] * len(paths),
        )
        expected = dict.fromkeys(("1", "2", "3"), self.SINGLE_FITS_DICT)
        result = await combiner.combine(paths)
        assert result == expected
