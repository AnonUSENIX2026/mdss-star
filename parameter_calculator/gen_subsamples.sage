import random
from collections import Counter

from sage.all import Integer
from tqdm import tqdm


def simulate_decoding(instance: Counter, threshold, batch_size):
    top_counts = []
    instance_copy = instance.copy()
    while instance_copy.most_common(1)[0][1] >= threshold:
        elements = sorted(instance_copy.elements())

        if len(elements) <= batch_size:
            subsample = Counter(elements)
        else:
            subsample = Counter(random.sample(elements, batch_size))

        top_element = subsample.most_common(1)[0]
        top_counts.append(
            {
                "rank": int(top_element[0]),
                "count_in_subsample": int(top_element[1]),
            }
        )
        instance_copy[top_element[0]] = 0
    return top_counts


def gen_subsamples(
    batch_sizes: list[Integer], threshold: int, instances: list, subsamples: dict
) -> tuple[bool, dict]:
    num_instances = len(instances)
    did_update = False

    for batch_size in tqdm(batch_sizes, desc="Processing batch sizes"):
        int_batch_size = int(batch_size)

        if int_batch_size not in subsamples:
            subsamples[int_batch_size] = []

        num_batch_size_subsamples = len(subsamples[int_batch_size])

        if num_instances > num_batch_size_subsamples:
            did_update = True
            for i in tqdm(
                range(num_batch_size_subsamples, num_instances),
                desc="Adding subsamples",
                leave=False,
                position=1,
            ):
                subsamples[int_batch_size].append(
                    simulate_decoding(instances[i], threshold, batch_size)
                )

    return did_update, subsamples
