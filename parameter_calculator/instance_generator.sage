import json
import random
from collections import Counter
from multiprocessing import Pool

from tqdm import tqdm

import utils


def max_present_rank(instance, threshold) -> int:
    max_present = 0
    for rank, count in instance.most_common():
        if count < threshold:
            break
        else:
            if rank > max_present:
                max_present = rank
    return max_present


def instance_to_json(inst):
    data = {}
    for k, v in inst.items():
        data[int(k)] = v
    return data


def json_to_instance(data):
    inst = Counter()
    for k, v in data.items():
        inst[int(k)] = v
    return inst


def sample_from_vec(freqs):
    rand_num = random.random()
    cumulative_sum = 0.0
    for i, freq in enumerate(freqs):
        cumulative_sum += freq
        if rand_num < cumulative_sum:
            return i + 1

    # Freqs may not sum to 1, so we also include a final "not anything else" term
    # This should very rarely be returned.
    return -1


def sample_n_from_vec(n, freqs, enable_tqdm=False):
    c = Counter()
    for _ in tqdm(
        range(n),
        desc="Sampling instance",
        leave=False,
        disable=not enable_tqdm,
        position=1,
    ):
        c.update([sample_from_vec(freqs)])
    return c


def mp_sample_instance(num_threads, freqs, num_reports):
    if num_threads > 1:
        samples_per_thread = num_reports // (num_threads - 1)
    else:
        samples_per_thread = 0
    samples_final = num_reports - samples_per_thread * (num_threads - 1)

    sample_counts = ([samples_per_thread] * (num_threads - 1)) + [samples_final]

    with Pool(num_threads) as pool:
        list_args = [[sc, freqs] for sc in sample_counts]
        list_args[0].append(True)

        fractions = pool.starmap(sample_n_from_vec, list_args)

    total = Counter()
    for c in fractions:
        total += c

    return total


if __name__ == "__main__":
    freqs = utils.freqs_vec(10_000, 1.03)

    instances = []

    for i in tqdm(range(0, 2), desc="Sampling instances"):
        instances.append(instance_to_json(mp_sample_instance(6, freqs, 100_000)))

    with open("out2.json", "w+") as outfile:
        json.dump(instances, outfile)
