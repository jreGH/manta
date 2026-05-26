from setuptools import setup, find_packages

package_name = "manta_world_model"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    install_requires=["numpy"],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    entry_points={
        "console_scripts": [
            "world_model_node = manta_world_model.nodes.world_model_node:main",
        ],
    },
)
