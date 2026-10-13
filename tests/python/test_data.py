import pytest

from trtbench.data import IMAGENETTE_CLASSES


def test_imagenette_synsets_match_torchvision():
    """The folder IDs must be Imagenette's real synsets (a wrong one cost a
    Colab run: n03888605 is not parachute, n03888257 is)."""
    imagenette = pytest.importorskip("torchvision.datasets.imagenette")
    official = {wnid: names[0] for wnid, names in imagenette.Imagenette._WNID_TO_CLASS.items()}
    assert IMAGENETTE_CLASSES == official
