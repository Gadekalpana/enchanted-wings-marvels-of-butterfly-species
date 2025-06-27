!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="2jeRSDKAT1COo3g8vyWz")
project = rf.workspace("new-hnx8w").project("butterfly-srgkh")
version = project.version(1)
dataset = version.download("folder")

