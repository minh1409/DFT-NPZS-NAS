import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Argument Parser for Testing Progress")

    # Add arguments
    parser.add_argument("--results_path", type=str, default="Example/full_bench.json", help="")

    return parser.parse_args()