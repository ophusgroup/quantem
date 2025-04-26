import importlib

import h5py

from quantem.core.datastructures import Dataset as Dataset
from quantem.core.datastructures import Dataset4dstem as Dataset4dstem


def read_4dstem(
    file_path: str,
    file_type: str,
):
    """
    File reader for 4D-STEM data

    Parameters
    ----------
    file_path: str
        Path to data
    file_type: str
        The type of file reader needed. See rosettasciio for supported formats
        https://hyperspy.org/rosettasciio/supported_formats/index.html

    Returns
    --------
    Dataset4dstem
    """
    file_reader = importlib.import_module(f"rsciio.{file_type}").file_reader
    imported_data = file_reader(file_path)[0]
    dataset = Dataset4dstem.from_array(
        array=imported_data["data"],
        sampling=[
            imported_data["axes"][0]["scale"],
            imported_data["axes"][1]["scale"],
            imported_data["axes"][2]["scale"],
            imported_data["axes"][3]["scale"],
        ],
        origin=[
            imported_data["axes"][0]["offset"],
            imported_data["axes"][1]["offset"],
            imported_data["axes"][2]["offset"],
            imported_data["axes"][3]["offset"],
        ],
        units=[
            imported_data["axes"][0]["units"],
            imported_data["axes"][1]["units"],
            imported_data["axes"][2]["units"],
            imported_data["axes"][3]["units"],
        ],
    )

    return dataset


def read_2d(
    file_path: str,
    file_type: str,
):
    """
    File reader for images

    Parameters
    ----------
    file_path: str
        Path to data
    file_type: str
        The type of file reader needed. See rosettasciio for supported formats
        https://hyperspy.org/rosettasciio/supported_formats/index.html

    Returns
    --------
    Dataset
    """
    file_reader = importlib.import_module(f"rsciio.{file_type}").file_reader
    imported_data = file_reader(file_path)[0]
    dataset = Dataset.from_array(
        array=imported_data["data"],
        sampling=[
            imported_data["axes"][0]["scale"],
            imported_data["axes"][1]["scale"],
        ],
        origin=[
            imported_data["axes"][0]["offset"],
            imported_data["axes"][1]["offset"],
        ],
        units=[
            imported_data["axes"][0]["units"],
            imported_data["axes"][1]["units"],
        ],
    )

    return dataset


def read_emdfile_to_4dstem(file_path):
    """
    File reader for legacy `emdFile` / `py4DSTEM` files.

    Parameters
    ----------
    file_path: str
        Path to data

    Returns
    --------
    Dataset4dstem
    """
    with h5py.File(file_path, "r") as file:
        dataset = Dataset4dstem.from_array(
            array=file["datacube_root"]["datacube"]["data"],
            sampling=[
                file["datacube_root"]["metadatabundle"]["calibration"]["R_pixel_size"][
                    ()
                ],
                file["datacube_root"]["metadatabundle"]["calibration"]["R_pixel_size"][
                    ()
                ],
                file["datacube_root"]["metadatabundle"]["calibration"]["Q_pixel_size"][
                    ()
                ],
                file["datacube_root"]["metadatabundle"]["calibration"]["Q_pixel_size"][
                    ()
                ],
            ],
            units=[
                file["datacube_root"]["metadatabundle"]["calibration"]["R_pixel_units"][
                    ()
                ],
                file["datacube_root"]["metadatabundle"]["calibration"]["R_pixel_units"][
                    ()
                ],
                file["datacube_root"]["metadatabundle"]["calibration"]["Q_pixel_units"][
                    ()
                ],
                file["datacube_root"]["metadatabundle"]["calibration"]["Q_pixel_units"][
                    ()
                ],
            ],
        )

    return dataset
