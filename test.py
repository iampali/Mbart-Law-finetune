# import argparse
# from setup_logging import logger
# import torch
# import numpy as np
# import pandas as pd

# logger.info(f"Version of troch {torch.__version__}")
# print(f"gpu avability is  {torch.cuda.is_available()}")
# print(torch.cuda.get_device_name(0))

# print(f"Numpy is {np.__version__} and pandas is {pd.__version__}")


# parser = argparse.ArgumentParser(description="Example script with optional flags")


# parser.add_argument("-n","--name", default="Guest", help="Your name")
# parser.add_argument("-a","--age", type=int, default=18, help="Your age")
# parser.add_argument("-d", "--degree", help="what degree are you pursuing")
# parser.add_argument("-u", "--university", help="Name of your university")
# parser.add_argument("-c", "--city", default="Ajmer", help="Name of your current city")
# args = parser.parse_args()


# logger.info(f"Hello My name is {args.name} and I am {args.age} years old. I am pursing my {args.degree} from {args.university} in {args.city}.")
a = 100
assert a > 100,  "What the fuck is going on"