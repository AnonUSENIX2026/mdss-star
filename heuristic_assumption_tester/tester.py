import argparse
import csv
import json
from pathlib import Path
import random
import subprocess
import uuid

from math import ceil, floor

import git

instance_type_lookup = {"adversarial": "a"}


def min_t(batch_size, c, ell):
    return ceil(((batch_size) / (c + 1)) + ((c * (ell + 1)) / (c + 1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run tests on our heuristic assumption")
    parser.add_argument(
        "--threads",
        "-th",
        type=int,
        default=1,
        help="The number of threads to run the decoder with",
    )
    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=1,
        help="The number of iterations to run",
    )
    parser.add_argument(
        "--min-n", type=int, default=1000, help="the minimum n value to consider"
    )
    parser.add_argument(
        "--max-n", type=int, default=30000, help="the maximum n value to consider"
    )
    parser.add_argument(
        "--min-c", type=int, default=1, help="The minimum value of c to consider"
    )
    parser.add_argument(
        "--max-c", type=int, default=60, help="The maximum value of c to consider"
    )
    parser.add_argument(
        "--min-ell", type=int, default=1, help="The minimum value of ell to consider"
    )
    parser.add_argument(
        "--max-ell", type=int, default=400, help="The maximum value of ell to consider"
    )
    parser.add_argument(
        "--instance-type",
        "-t",
        type=str,
        default="adversarial",
        help="The types of instances to generate",
    )
    parser.add_argument("logfile", type=str, help="The file to log results to")
    parser.add_argument(
        "cache_dir",
        type=str,
        help="The directory to cache failing instances",
        default="cache",
    )
    parser.add_argument(
        "--low-polys", "-lp", action="store_true", help="Promote low degree polynomials"
    )
    parser.add_argument(
        "--fix-max-degree",
        "-fmd",
        action="store_true",
        help="Fix the max degree term of the polys to 1",
    )
    parser.add_argument("--decoder-command", "-dc", type=str)
    args = parser.parse_args()

    field_names = [
        "id",
        "githash",
        "instance_type",
        "min_n",
        "max_n",
        "min_c",
        "max_c",
        "min_ell",
        "max_ell",
        "n",
        "c",
        "ell",
        "t",
        "is_nice",
        "low_polys",
        "fixed_max_degree",
        "succeeded",
    ]

    current_hash = git.Repo(search_parent_directories=True).head.object.hexsha[:7]

    for _ in range(args.iterations):
        n = random.randint(args.min_n, args.max_n)
        c = random.randint(args.min_c, args.max_c)
        ell = random.randint(args.min_ell, args.max_ell)

        t = min_t(n, c, ell)

        min_sufficient_points = ell + 1
        max_gap_sufficient_dealers = floor(n / min_sufficient_points)
        num_gap_sufficient_dealers = random.randint(0, max_gap_sufficient_dealers)

        sufficient_dealer_counts = []

        if num_gap_sufficient_dealers > 0:
            max_points = floor(n / num_gap_sufficient_dealers)

            for dealer_idx in range(num_gap_sufficient_dealers):
                num_points = random.randint(min_sufficient_points, max_points)
                sufficient_dealer_counts.append(num_points)

        else:
            max_points = -1

        print(
            f"{n=}, {c=}, {ell=}, {t=}, ngsd={num_gap_sufficient_dealers}, mp={max_points}"
        )

        run_id = uuid.uuid4()

        config_name = "temp_config.json"
        instance_name = "temp_instance.json"
        decoder_output_name = "out.json"

        # Write the config file
        config = {"batch_sizes": [n], "c_vals": [c], "threads": args.threads}
        with open(config_name, "w+") as outfile:
            json.dump(config, outfile)

        # Generate the instance
        command = [
            "sage",
            "../instance_generation/instance_generation.sage",
            instance_type_lookup[args.instance_type],
            f"./{instance_name}",
            "-c",
            str(c),
            "-l",
            str(ell),
            "-mf",
            str(n + 1),
        ]
        if args.low_polys:
            command += ["-fldp"]
        if args.fix_max_degree:
            command += ["-fmd"]

        command += ["-e"]
        command += [str(v) for v in sufficient_dealer_counts]
        command += [str(n - sum(sufficient_dealer_counts))]

        subprocess.run(command)

        with open(instance_name) as infile:
            instance = json.load(infile)

        subprocess.run(
            [
                args.decoder_command,
                f"./{instance_name}",
                "--outfile",
                f"./{decoder_output_name}",
            ]
        )

        with open(decoder_output_name) as infile:
            result = json.load(infile)

        if not result["succeeded"]:
            Path(f"./{instance_name}").rename(
                Path(args.cache_dir) / f"instance-{run_id}.json"
            )
            Path(f"./{config_name}").rename(
                Path(args.cache_dir) / f"config-{run_id}.json"
            )
            Path(f"./{decoder_output_name}").rename(
                Path(args.cache_dir) / f"decoder-output-{run_id}.json"
            )

        with open(args.logfile, "a+") as logfile:
            write_header = logfile.tell() == 0

            writer = csv.DictWriter(logfile, fieldnames=field_names)

            if write_header:
                writer.writeheader()

            writer.writerow(
                {
                    "id": run_id,
                    "githash": current_hash,
                    "instance_type": args.instance_type,
                    "min_n": args.min_n,
                    "max_n": args.max_n,
                    "min_c": args.min_c,
                    "max_c": args.max_c,
                    "min_ell": args.min_ell,
                    "max_ell": args.max_ell,
                    "n": n,
                    "c": c,
                    "ell": ell,
                    "t": t,
                    "low_polys": args.low_polys,
                    "fixed_max_degree": args.fix_max_degree,
                    "succeeded": result["succeeded"],
                    "is_nice": instance["parameters"]["is_nice"],
                }
            )
