import math
from collections import Counter

from sage.all import binomial, ceil, erf, sqrt


def min_t(batch_size, c, ell):
    return ceil(((batch_size) / (c + 1)) + ((c * (ell + 1)) / (c + 1)))


def pr_rank(N, s, rank):
    # Given Zipf parameters N and s, return the probability that a
    # sampled element is the element of rank `i`.
    H_N = sum(1 / (k**s) for k in range(1, N + 1))
    return (1 / (H_N)) * (1 / (rank**s))


def sanitize_json(j):
    if type(j) is dict:
        return {str(k): sanitize_json(v) for k, v in j.items()}
    elif type(j) is list:
        return [int(v) for v in j]
    else:
        return int(j)


def freqs_vec(N, s):
    H_N = sum(1 / (k**s) for k in range(1, N + 1))
    return [(1 / (H_N)) * (1 / (rank**s)) for rank in range(1, N)]


def pr_alts_full(n, t, p):
    total = 0
    for i in range(t):
        adder = (binomial(n, i) * (p**i)) * ((1 - p) ** (n - i))
        if math.isnan(adder):
            adder = 0
        total += adder

    prob_less_than_t = total
    return 1 - prob_less_than_t


def pr_rank_given_at_least_t_and_not_higher(support, pool_size, s, rank, t):
    # Given Zipf parameters `support` and `s`, return the probability that a sampled element is the element
    # of rank `rank`, conditioned on the element *not* being any element of higher rank, and on at
    # least `t` of the element existing in the pool of `pool_size` elements.

    # The probability of sampling an element of rank `rank`
    p_ri = pr_rank(support, s, rank)

    p_rh = sum([pr_rank(support, s, rank) for rank in range(1, rank)])
    p_not_rh = 1 - p_rh

    p_alti_given_ri = pr_at_least_t_successes(pool_size - 1, t - 1, p_ri)
    p_alti_given_not_ri = pr_at_least_t_successes(pool_size - 1, t, p_ri)

    p_ri_over_p_not_rh = p_ri / p_not_rh

    p_alti_given_not_rh = (p_ri_over_p_not_rh * p_alti_given_ri) + (
        (1 - p_ri_over_p_not_rh) * p_alti_given_not_ri
    )

    return min((p_alti_given_ri * p_ri) / (p_alti_given_not_rh * p_not_rh), 1)


def pr_alts_approx(n, t, p):
    """
    Returns an approximation of the probability of at least `t` successes out of `n` trials
    with success probability `p`, using the normal approximation for large `n`.

    Args:
        n (int): Total number of trials.
        t (int): Minimum number of successes.
        p (float): Probability of success in each trial.

    Returns:
        float: Approximate probability of at least `k` successes out of `n` trials.
    """

    # Mean and standard deviation of the approximating normal distribution
    mu = n * p
    sigma = sqrt(n * p * (1 - p))

    # Convert the binomial P(X >= k) to a normal P(X' >= k - 0.5) for continuity correction
    z = (t - 0.5 - mu) / sigma

    # Using the error function (erf) to compute the cumulative probability
    # The survival function for the standard normal distribution
    prob = 0.5 * (1 - erf(z / sqrt(2)))

    return prob


def pr_at_least_t_successes(N, t, p):
    if float(p) == 0:
        return 0
    if N > 8000:
        return pr_alts_approx(N, t, p)
    else:
        return pr_alts_full(N, t, p)


# Compute the max rank we care about seeing
# All ranks have some probability of occurring at least t times, but we don't need to see
# the recovery probability of all of them. This takes a cuttoff, so only ranks with a probability
# of occurring at least t times of `probability_cutoff` or greater make it in.
def find_max_rank(num_clients, N, s, t, probability_cutoff):
    max_rank = 1
    while max_rank <= N:
        freq = pr_rank(N, s, max_rank)
        pr_at_least_t = float(pr_at_least_t_successes(num_clients, t, freq))
        if pr_at_least_t < probability_cutoff:
            break
        max_rank += 1
    max_rank -= 1

    return max_rank


def gen_expected_instance(N, s, num_clients):
    # print(3 ** s)

    H_N = sum(1 / (k**s) for k in range(1, int(N) + 1))

    rank_counter = Counter()

    current_rank = 1
    total_made = 0
    while total_made < num_clients:
        pr = (1 / (H_N)) * (1 / (current_rank**s))
        expected_num = ceil(pr * num_clients)
        rank_counter.update({current_rank: expected_num})
        total_made += expected_num
        current_rank += 1
    return rank_counter


def str_obj(o):
    if type(o) is list:
        return [str_obj(i) for i in o]
    elif type(o) is dict:
        return {str(k): str_obj(v) for (k, v) in o.items()}
    else:
        return str(o)


def chunk_array(arr, chunk_size):
    return [arr[i : i + chunk_size] for i in range(0, len(arr), chunk_size)]
