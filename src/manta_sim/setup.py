from setuptools import setup, find_packages
import os
from glob import glob

package_name = "manta_sim"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    install_requires=["numpy"],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("../../launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("../../config/*.yaml")),
    ],
    entry_points={
        "console_scripts": [
            "diver_sim = manta_sim.nodes.diver_sim:main",
            "shark_sim = manta_sim.nodes.shark_sim:main",
            "explosive_sim = manta_sim.nodes.explosive_sim:main",
            "vessel_sim = manta_sim.nodes.vessel_sim:main",
        ],
    },
)
