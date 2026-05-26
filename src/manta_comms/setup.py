from setuptools import setup, find_packages

package_name = "manta_comms"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    entry_points={
        "console_scripts": [
            "comms_node = manta_comms.nodes.comms_node:main",
        ],
    },
)
