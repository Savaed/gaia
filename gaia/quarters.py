"""Quarter prefixes for Kepler time series features."""

from gaia.enums import Cadence


# Quarter index to filename prefix for long cadence Kepler data.
# Reference: https://archive.stsci.edu/kepler/software/get_kepler.py
_LONG_CADENCE_QUARTER_PREFIXES = (
    ("2009131105131"),
    ("2009166043257"),
    ("2009259160929"),
    ("2009350155506"),
    ("2010078095331", "2010009091648"),
    ("2010174085026"),
    ("2010265121752"),
    ("2010355172524"),
    ("2011073133259"),
    ("2011177032512"),
    ("2011271113734"),
    ("2012004120508"),
    ("2012088054726"),
    ("2012179063303"),
    ("2012277125453"),
    ("2013011073258"),
    ("2013098041711"),
    ("2013131215648"),
)

# Quarter index to filename prefix for short cadence Kepler data.
# Reference: https://archive.stsci.edu/kepler/software/get_kepler.py
_SHORT_CADENCE_QUARTER_PREFIXES = (
    ("2009131110544"),
    ("2009166044711"),
    ("2009201121230", "2009231120729", "2009259162342"),
    ("2009291181958", "2009322144938", "2009350160919"),
    ("2010009094841", "2010019161129", "2010049094358", "2010078100744"),
    ("2010111051353", "2010140023957", "2010174090439"),
    ("2010203174610", "2010234115140", "2010265121752"),
    ("2010296114515", "2010326094124", "2010355172524"),
    ("2011024051157", "2011053090032", "2011073133259"),
    ("2011116030358", "2011145075126", "2011177032512"),
    ("2011208035123", "2011240104155", "2011271113734"),
    ("2011303113607", "2011334093404", "2012004120508"),
    ("2012032013838", "2012060035710", "2012088054726"),
    ("2012121044856", "2012151031540", "2012179063303"),
    ("2012211050319", "2012242122129", "2012277125453"),
    ("2012310112549", "2012341132017", "2013011073258"),
    ("2013017113907", "2013065031647", "2013098041711"),
    ("2013121191144", "2013131215648"),
)


def get_quarter_prefixes(cadence: Cadence) -> tuple[str, ...]:
    """Return Kepler quarter prefixes for a specified observation cadence.

    Args:
        cadence (Cadence): Observations cadence

    Returns:
        tuple[str, ...]: All quarters prefixes
    """
    queraters = (
        _SHORT_CADENCE_QUARTER_PREFIXES
        if cadence is Cadence.SHORT
        else _LONG_CADENCE_QUARTER_PREFIXES
    )
    return tuple(prefix for prefixes in queraters for prefix in prefixes)
