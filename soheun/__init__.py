# add current directory (as an absolute directory) to path
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))
