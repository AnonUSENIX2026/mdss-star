from multiprocessing import Pool

from tqdm import tqdm

import utils


def pr_success(batch_size, c, ell, pr):
    min_t = utils.min_t(batch_size, c, ell)
    return utils.pr_at_least_t_successes(batch_size, min_t, pr)


# Crude, but mostly agrees with what I've seen in practice
def cost(batch_size, c, linear_cost: bool):
    if linear_cost:
        return c * batch_size
    else:
        return (c**2) * batch_size


def find_best_balance(
    min_success_prob, rank, data, c_vals, batch_sizes, quiet, linear_cost
):
    best = None
    for c in tqdm(c_vals, leave=False, position=1, desc="Trying c's", disable=quiet):
        for bs in batch_sizes:
            ps = data[c][bs][rank]

            if ps >= min_success_prob:
                # if bs < 10000 or c < 170:
                if best is None or cost(bs, c, linear_cost) < cost(
                    best[0], best[1], linear_cost
                ):
                    best = (bs, c)
    return best


def derive_success_prob(rank, min_t, trials):
    return len(list(filter(lambda trial: trial[rank] >= min_t, trials))) / len(trials)
    # TODO: Need to only count "successes" as times when the rank is sufficiently present.
    # For now, no need, as max rank is 10 and in the expected instance 10 is always present


def get_recovered_ranks(subsamples, min_t: int) -> list[int]:
    recovered_ranks = []
    for subsample in subsamples:
        if subsample["count_in_subsample"] >= min_t:
            recovered_ranks.append(subsample["rank"])
        else:
            break
    return recovered_ranks


def derive_success_probs(
    iterations, min_t, max_rank, present_ranks_per_instance: list[set]
):
    rank_info: dict[int, dict[str, int | float]] = {}

    for index, iteration in enumerate(iterations):
        recovered_ranks = get_recovered_ranks(iteration, min_t)

        for test_rank in range(1, max_rank + 1):
            if test_rank not in rank_info:
                rank_info[test_rank] = {"present": 0, "recovered": 0}

            if test_rank in present_ranks_per_instance[index]:
                rank_info[test_rank]["present"] += 1

                if test_rank in recovered_ranks:
                    rank_info[test_rank]["recovered"] += 1

    for _, info in rank_info.items():
        pres_count = info["present"]
        rec_count = info["recovered"]

        info["pr_recovered_when_present"] = (
            rec_count / pres_count if pres_count > 0 else 1
        )

    return rank_info


def get_success_data(
    c, batch_sizes, ell, subsamples, min_rank, max_rank, present_ranks_per_instance
):
    c_dict = {}
    for bs in batch_sizes:
        mt = utils.min_t(bs, c, ell)
        bs_dict = {}

        success_probs = derive_success_probs(
            subsamples[bs], mt, max_rank, present_ranks_per_instance
        )

        for rank in range(min_rank, max_rank + 1):
            bs_dict[rank] = success_probs[rank]["pr_recovered_when_present"]

        c_dict[bs] = bs_dict
    return c_dict


def get_success_data_batch(
    c_vals,
    batch_sizes,
    ell,
    subsamples,
    min_rank,
    max_rank,
    present_ranks_per_instance,
    print_progress=False,
) -> dict:
    success_data = {}
    for c in tqdm(
        c_vals,
        disable=not print_progress,
        desc="Analyzing decoder success probabilities",
    ):
        c_dict = get_success_data(
            c,
            batch_sizes,
            ell,
            subsamples,
            min_rank,
            max_rank,
            present_ranks_per_instance,
        )
        success_data[c] = c_dict
    return success_data


def get_success_data_threaded(
    c_vals,
    num_threads,
    batch_sizes,
    ell,
    subsamples,
    min_rank,
    max_rank,
    present_ranks_per_instance,
) -> dict:

    result = {}
    chunks = utils.chunk_array(c_vals, len(c_vals) // num_threads)
    with Pool(num_threads) as pool:
        list_args = [
            [
                chunk,
                batch_sizes,
                ell,
                subsamples,
                min_rank,
                max_rank,
                present_ranks_per_instance,
            ]
            for chunk in chunks
        ]
        list_args[0].append(True)
        results = pool.starmap(get_success_data_batch, list_args)

    for r in results:
        result.update(r)

    return result


def run(
    batch_sizes,
    c_vals,
    ell,
    min_success_prob,
    max_rank: int,
    subsamples,
    present_ranks_per_instance: list[set],
    min_rank=1,
    quiet=False,
    linear_cost=False,
    num_threads=8,
):
    # success_data = {}
    #
    # for c in tqdm(
    #     c_vals, disable=quiet, desc="Analyzing decoder success probabilities"
    # ):
    #
    #     success_data[c] = get_success_data(
    #         c,
    #         batch_sizes,
    #         ell,
    #         subsamples,
    #         min_rank,
    #         max_rank,
    #         present_ranks_per_instance,
    #     )

    success_data = get_success_data_threaded(
        c_vals,
        num_threads,
        batch_sizes,
        ell,
        subsamples,
        min_rank,
        max_rank,
        present_ranks_per_instance,
    )

    results = []
    for rank in tqdm(
        range(min_rank, max_rank + 1),
        desc="Optimizing c values for ranks",
        disable=quiet,
    ):
        results.append(
            find_best_balance(
                min_success_prob,
                rank,
                success_data,
                c_vals,
                batch_sizes,
                quiet,
                linear_cost,
            )
        )

    return results
