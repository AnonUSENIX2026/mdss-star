import argparse
import json
import os
from pathlib import Path

from tabulate import tabulate
from tqdm import tqdm

import cache
import gen_subsamples
import instance_generator
import ladder_calculator
import leakage_calculator
import utils


def get_present_ranks(instance, threshold) -> set[int]:
    present_ranks = set()
    for rank, count in instance.most_common():
        if count < threshold:
            break
        else:
            present_ranks.add(rank)
    return present_ranks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Play with STARPLUS/POPSTAR parameters."
    )

    parser.add_argument(
        "-n", "--num_clients", type=int, help="The number of clients", default=100_000
    )
    parser.add_argument(
        "-t", "--threshold", type=int, help="The recovery threshold", default=1_000
    )
    parser.add_argument("-N", type=int, help="The Zipf N param", default=10_000)
    parser.add_argument("-s", type=str, help="The Zipf s param", default="1.03")
    parser.add_argument(
        "-cd",
        "--cache-dir",
        type=str,
        help="The name of directory to store caches",
        default="./cache",
    )
    parser.add_argument(
        "-mrp",
        "--min-recovery-prob",
        type=float,
        help="The minimum probability of recovering a rank for a parameter setting to be acceptable",
        default=0.99,
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        help="The number of iterations to run when simulating recovery probability",
        default=10,
    )
    parser.add_argument(
        "-l",
        "-ell",
        "--ell",
        type=int,
        help="Set the degree of the polynomials, rather than computing the maximum",
    )
    parser.add_argument(
        "--include-threshold",
        action="store_true",
        help="Include in the expected instance a rank that exactly reaches the threshold",
    )
    parser.add_argument(
        "--parallel-subsamples",
        "-ps",
        type=int,
        default=1,
        help="The number of subsamples to run in parallel. There must be > args.mrp probability that *any* succeeds",
    )
    parser.add_argument(
        "--threads",
        "-th",
        type=int,
        default=1,
        help="The number of threads to use when sampling instances",
    )
    parser.add_argument(
        "--linear-cost",
        "--lin",
        action="store_true",
        default=False,
        help=(
            "Use a linear cost function for determining best decoding parameters, i.e. cost=c*n, "
            "rather than cost=c^2 * n"
        ),
    )
    parser.add_argument(
        "--output", type=str, help="A file to write the configuration as json to"
    )
    parser.add_argument(
        "--fix-leading-coeff",
        "-flc",
        action="store_true",
        default=False,
        help="Fix the leading coefficient when computing shares, resulting in 1-lower privacy threshold",
    )

    args = parser.parse_args()

    # Initialize cache dir
    if not os.path.exists(args.cache_dir):
        print("Creating cache directory")
        os.makedirs(args.cache_dir)

    cache_path = (
        Path(args.cache_dir)
        / f"n{args.num_clients}_t{args.threshold}_N{args.N}_s{args.s}.json"
    )

    # Some of these computations require a lot of simulations or samples.
    # To avoid re-doing work, we save previous simulations in a cache,
    # and load it in here.
    print("Loading cache at", cache_path)
    try:
        with open(cache_path, "r") as infile:
            data = json.load(infile)
        my_cache = cache.Cache.from_json(data)
    except FileNotFoundError:
        print("Unable to find cache. Creating a new one.")
        my_cache = cache.Cache([], {})

    # print("Running POPSTAR simulator...")
    # total_leaked, num_hh, num_groups = popstar_simulator.run_simulation(
    #     args.num_clients, args.threshold, args.N, args.s, 16, current_cache
    # )
    # print("Total leaked:", total_leaked)

    # Just hard-code a bunch of batch sizes and c values to try.
    batch_sizes = list(range(1000, 67_000, 1000))
    c_vals = list(range(1, 171))

    # Generate instances
    freqs = utils.freqs_vec(args.N, float(args.s))
    needed_instances = args.iterations - len(my_cache.instances)
    did_update = False
    if needed_instances > 0:
        did_update = True
        for _ in tqdm(range(needed_instances), desc="Generating instances"):
            inst = instance_generator.mp_sample_instance(
                args.threads, freqs, args.num_clients
            )
            my_cache.instances.append(inst)

    did_subsamples_update, my_cache.subsamples = gen_subsamples.gen_subsamples(
        batch_sizes, args.threshold, my_cache.instances, my_cache.subsamples
    )

    did_update = did_update or did_subsamples_update

    if did_update:
        print("Saving updated cache...")
        with open(cache_path, "w+") as outfile:
            json.dump(my_cache.to_json(), outfile)

    max_rank = my_cache.max_present_rank(args.threshold)
    mrp = 1 - ((1 - args.min_recovery_prob).nth_root(args.parallel_subsamples))

    present_ranks_per_instance = list(
        map(lambda inst: get_present_ranks(inst, args.threshold), my_cache.instances)
    )

    if args.ell is None:
        print("Optimizing ell...")
        max_ell = -1
        for test_ell in tqdm(range(args.threshold, 0, -1)):
            # max_ell relates only to decoding the max rank
            results = ladder_calculator.run(
                batch_sizes,
                c_vals,
                test_ell,
                mrp,
                max_rank,
                my_cache.subsamples,
                present_ranks_per_instance,
                min_rank=max_rank,
                quiet=True,
                linear_cost=args.linear_cost,
                num_threads=args.threads,
            )
            if results[0] is not None:
                max_ell = test_ell
                break

        print(
            "Maximum ell that achieves the required reconstruction probability:",
            max_ell,
        )
    else:
        max_ell = args.ell

    leakage_calculator.run(
        args.num_clients,
        args.N,
        float(args.s),
        args.threshold,
        max_ell,
        args.fix_leading_coeff,
    )

    ladder_results = ladder_calculator.run(
        batch_sizes,
        c_vals,
        max_ell,
        mrp,
        max_rank,
        my_cache.subsamples,
        present_ranks_per_instance,
        linear_cost=args.linear_cost,
        num_threads=args.threads,
    )

    rank_info = []
    for rank in range(1, max_rank + 1):
        times_present = 0
        total_count = 0

        for inst in my_cache.instances:
            rank_count = inst[rank]
            if rank_count > args.threshold:
                times_present += 1
                total_count += rank_count

        if times_present > 0:
            rank_info.append(
                (times_present, f"{float(total_count / times_present):0.2f}")
            )
        else:
            rank_info.append((times_present, "n/a"))

    headers = [
        "rank",
        "c",
        "batch size",
        "new t",
        "times present",
        "average count when present",
    ]
    table = []
    for index, result in enumerate(ladder_results):
        if result is not None:
            c_val = result[1]
            batch_size = result[0]
            mt = utils.min_t(batch_size, c_val, max_ell)

            table.append(
                [
                    index + 1,
                    c_val,
                    batch_size,
                    mt,
                    rank_info[index][0],
                    rank_info[index][1],
                ]
            )
        else:
            table.append(
                [index + 1, None, None, None, rank_info[index][0], rank_info[index][1]]
            )

    print("\nDecoding Parameters per rank")
    print(tabulate(table, headers=headers))

    if args.output is not None:
        output_data = {"threads": int(32), "batch_sizes": [], "c_vals": []}
        for result in ladder_results:
            output_data["batch_sizes"].append(int(result[0]))
            output_data["c_vals"].append(int(result[1]))

        with open(args.output, "w+") as outfile:
            json.dump(output_data, outfile, indent=2)
