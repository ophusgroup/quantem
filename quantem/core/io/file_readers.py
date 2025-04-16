from quantem.core.datastructures import Dataset as Dataset
import numpy as np


def read_4D(
    file_path,
    file_type,
):
    from rsciio.digitalmicrograph import file_reader

    imported_data = file_reader(file_path)[0]
    dataset = Dataset(
        data=imported_data["data"],
        sampling=np.asarray(
            [
                imported_data["axes"][0]["scale"],
                imported_data["axes"][1]["scale"],
                imported_data["axes"][2]["scale"],
                imported_data["axes"][3]["scale"],
            ]
        ),
        origin=np.asarray(
            [
                imported_data["axes"][0]["offset"],
                imported_data["axes"][1]["offset"],
                imported_data["axes"][2]["offset"],
                imported_data["axes"][3]["offset"],
            ]
        ),
        units=np.asarray(
            [
                imported_data["axes"][0]["units"],
                imported_data["axes"][1]["units"],
                imported_data["axes"][2]["units"],
                imported_data["axes"][3]["units"],
            ]
        ),
    )

    return dataset
