import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sage.all import GF, Integer, PolynomialRing, ceil, shuffle
from tqdm import tqdm


def mdss_threshold(n, c, ell):
    return ceil((n + (c * ell)) / (c + 1))


def serialize_poly(poly, degree):
    return [int(poly[i]) for i in range(degree + 1)]


def load_poly(pR, poly):
    return pR(list(poly))


@dataclass
class Instance:
    field: Any
    pR: Any
    n: int
    c: int
    ell: int
    agreement: int
    present_dealers: dict[int, list[list[Any]]]
    codeword: list[tuple[Integer, list[Integer]]]
    symbol_to_dealer: list[int]
    is_nice: bool = field(init=False)

    @staticmethod
    def calc_if_nice(threshold, symbol_to_dealer):

        if len(symbol_to_dealer) == 0:
            return True

        c = Counter(symbol_to_dealer)

        prev_count = c.most_common(1)[0][1] + 2

        for _, count in c.most_common():
            if count < threshold:
                break
            if prev_count - count <= 1 or count == threshold:
                return False

            prev_count = count

        return True

    def __post_init__(self):
        threshold = mdss_threshold(self.n, self.c, self.ell)
        self.is_nice = Instance.calc_if_nice(threshold, self.symbol_to_dealer)

    def to_serializable(self):
        return {
            "parameters": {
                "field_size": len(self.field),
                "variable": str(self.pR.gens()[0]),
                "n": int(self.n),
                "c": int(self.c),
                "ell": int(self.ell),
                "agreement": int(self.agreement),
                "is_nice": self.is_nice,
            },
            "present_dealers": {
                int(rank): list(
                    map(lambda poly: serialize_poly(poly, self.ell), poly_list)
                )
                for rank, poly_list in self.present_dealers.items()
            },
            "codeword": list(
                map(
                    lambda symbol: [
                        int(symbol[0]),
                        list(map(int, symbol[1])),
                    ],
                    self.codeword,
                )
            ),
            "symbol_to_dealer": list(map(int, self.symbol_to_dealer)),
        }

    @staticmethod
    def from_json(data):
        params = data["parameters"]
        f = GF(params["field_size"])
        pR = PolynomialRing(f, params["variable"])

        return Instance(
            f,
            pR,
            params["n"],
            params["c"],
            params["ell"],
            params["agreement"],
            {
                dealer_id: list(map(lambda poly: load_poly(pR, poly), poly_list))
                for dealer_id, poly_list in data["present_dealers"].items()
            },
            [
                [f(symbol[0]), [f(yc) for yc in symbol[1]]]
                for symbol in data["codeword"]
            ],
            data["symbol_to_dealer"],
        )


def random_non_zero_element(field):
    elt = field.random_element()
    while elt == 0:
        elt = field.random_element()
    return elt


def gen_unique_elements(field, n):
    evaluation_points = []
    for i in range(n):
        elt = field.random_element()
        while elt in evaluation_points:
            elt = field.random_element()
        evaluation_points.append(elt)
    return evaluation_points


def random_polynomial(pR, degree):
    return pR.random_element(degree=degree)


def polynomial_with_random_coefficients(pR, degree):
    # Samples a random coefficient for each power of x
    # As opposed to just pR.random_element(degree=degree), which always has a non-zero coefficient
    # for x^{degree}. Also as opposed to pR.random_element(degree=(-1, degree)), which samples the
    # degree uniformly first, i.e. overweights lower degree polys.

    result = pR(0)
    var = pR.gens()[0]
    for current_deg in range(degree + 1):
        result += pR.base_ring().random_element() * (var**current_deg)
    return result


def polynomial_with_random_coefficients_and_fixed_max_degree(pR, degree):
    # Fixes the term of x^degree to be 1, uniformly samples coefficients for the rest

    var = pR.gens()[0]

    return polynomial_with_random_coefficients(pR, degree - 1) + (var**degree)


def random_polynomial_up_to_degree(pR, degree):
    # Uniformly samples a degree, then gives a poly of that degree.
    return pR.random_element(degree=(-1, degree))


def gen_adversarial_instance(
    field, pR, ell, c, agreement, evals, favor_low_deg_polys=False, fix_max_degree=False
):
    evals = sorted(evals, reverse=True)
    m = len(evals) - 1  # number of polynomial sets
    n = sum(evals)
    f_list = [
        [
            (
                polynomial_with_random_coefficients_and_fixed_max_degree(pR, ell)
                if fix_max_degree
                else (
                    polynomial_with_random_coefficients(pR, ell)
                    if not favor_low_deg_polys
                    else random_polynomial_up_to_degree(pR, ell)
                )
            )
            for i in range(c)
        ]
        for j in range(m)
    ]
    eval_points = gen_unique_elements(field, n)
    codeword = []
    i = 0
    for k in range(m):
        for j in range(evals[k]):
            codeword.append(
                (
                    k + 1,
                    [eval_points[i], list(fi(eval_points[i]) for fi in f_list[k]), k],
                )
            )
            i += 1
    for j in range(evals[-1]):
        codeword.append(
            (
                -1,
                [eval_points[i], list(field.random_element() for fi in range(c)), -1],
            )
        )
        i += 1

    present_ranks = {}
    for rank in range(1, len(evals)):
        if evals[rank - 1] < agreement:
            break
        present_ranks[rank] = f_list[rank - 1]

    random.shuffle(codeword)

    symbol_to_dealer = [x[0] for x in codeword]
    codeword = [x[1] for x in codeword]

    return Instance(
        field, pR, n, c, ell, agreement, present_ranks, codeword, symbol_to_dealer
    )


def gen_noise_points(field, c, n):
    return [
        [field.random_element(), [field.random_element() for _ in range(c)], -1]
        for _ in range(n)
    ]


def gen_simulated_instance(
    field,
    pR,
    ell,
    c=1,
    num_stalkers=3,
    anonymity_epoch=4,
    delta=4,
    detection_time=3600,
    deletion_percent=5,
    num_noise_points=0,
    previous_windows=23,
):
    if anonymity_epoch % delta != 0:
        raise ValueError(
            f"delta ({delta}) must divide the anonymity epoch ({anonymity_epoch})."
        )
    if detection_time % anonymity_epoch != 0:
        raise ValueError(
            f"anonymity_epoch ({anonymity_epoch}) must divide detection time "
            f"({detection_time})."
        )
    broadcasts_per_anonymity_epoch = anonymity_epoch // delta
    num_anonymity_epochs = detection_time // anonymity_epoch

    used_eval_points = set()
    for _ in range(previous_windows * num_anonymity_epochs):
        used_eval_points.add(random_non_zero_element(field))

    stalker_broadcasts = []
    for stalker in range(num_stalkers):
        # Create a polynomial for the stalker
        stalker_polys = [random_polynomial(pR, ell) for _ in range(c)]
        for _ in range(num_anonymity_epochs):
            # Generate the evaluation point for this anonymity epoch
            eval_point = random_non_zero_element(field)

            if eval_point not in used_eval_points:
                broadcast = [
                    eval_point,
                    [poly(eval_point) for poly in stalker_polys],
                    stalker,
                ]
            else:
                broadcast = [eval_point, [field.random_element() for _ in range(c)], -1]
            used_eval_points.add(eval_point)

            for _ in range(broadcasts_per_anonymity_epoch):
                # Simulate the channel dropping a percentage of broadcasts by only
                # appending the broadcast with a 1 - deletion_percent chance
                if random.randint(1, 100) > deletion_percent:
                    stalker_broadcasts.append(broadcast)

    noise_points = gen_noise_points(field, c, num_noise_points)

    all_broadcasts = stalker_broadcasts + noise_points
    shuffle(all_broadcasts)

    return all_broadcasts


def sample_from_vec(freqs):
    """
    Given a discrete probability distribution `freqs`, sample an item accordingly. 1-indexed to
    match use in Zipfian distributions.

    Args:
        freqs (list[float]): The probability of sampling each value

    Returns:
        int: A value sampled according to the passed frequencies
    """

    rand_num = random.random()
    cumulative_sum = 0.0
    for i, freq in enumerate(freqs):
        cumulative_sum += freq
        if rand_num < cumulative_sum:
            return i + 1


def get_shuffled_sequential_elements(field, start, end):
    elements = [field(e) for e in range(start, end)]
    shuffle(elements)
    return elements


def gen_zipfian_instance(
    field, pR, ell, c, agreement, support, s, n, fix_max_degree=False, **tqdm_args
):
    """
    Generates an instance based on a Zipfian distribution of client inputs

    Args:
        field (FiniteField): The field to draw values from
        pR (PolynomialRing): The polynomial ring clients sample polynomials from
        ell (int): The degree of the polynomials
        c (int): The number of polynomials a client submits points on in each report
        agreement (int): The minimum number of points submitted for the same value at which that
            value should be considered "present" in the input (and therefore need to be
            reconstructed).
        support (int): The support of the Zipfian distribution. The number of possible client
            values.
        s (float): The s parameter for the Zipfian distribution
        n (int): The total number of points in the instance

    Returns:
        tuple[
            dict[int, list[Polynomial]],
            list[
                tuple[Integer, list[Integer], Integer]
            ]
        ]: A generated instance. First tuple element is the present Zipf ranks and their associated
        polynomials. Second is the instance itself. A list of points, each of which also has the
        value associated with it as the final entry.
    """
    normalization_const = sum(1 / (k**s) for k in range(1, support + 1))

    freqs = [(1 / (k**s)) / normalization_const for k in range(1, support + 1)]
    eval_points = get_shuffled_sequential_elements(field, 1, n + 1)

    rank_to_polys = {}
    codeword = []
    symbol_to_dealer = []
    for eval_point in tqdm(eval_points, **tqdm_args):
        rank = sample_from_vec(freqs)

        if rank not in rank_to_polys:
            if not fix_max_degree:
                polys = [random_polynomial(pR, ell) for _ in range(c)]
            else:
                polys = [
                    polynomial_with_random_coefficients_and_fixed_max_degree(pR, ell)
                    for _ in range(c)
                ]
            rank_to_polys[rank] = polys

        codeword.append(
            [eval_point, list(poly(eval_point) for poly in rank_to_polys[rank])]
        )
        symbol_to_dealer.append(rank)

    present_ranks = {}
    rank_counts = Counter(symbol_to_dealer)

    for rank, polys in rank_to_polys.items():
        if rank_counts[rank] >= agreement:
            present_ranks[rank] = polys

    return Instance(
        field, pR, n, c, ell, agreement, present_ranks, codeword, symbol_to_dealer
    )


def gen_malicious_client_instance(field, pR, ell, c, agreement, evals, **tqdm_args):
    """
    Generates an instance where there is a maliciously crafted input point.
    Malicious input is of the form (x, y_1, y_2,..., r,..., y_c), where y_i is an honestly
    generated point, and r is a randomly sampled value.
    """
    n = sum(evals)
    rank_to_polys = [
        [random_polynomial(pR, ell) for _ in range(c)] for _ in range(len(evals))
    ]

    eval_points = gen_unique_elements(field, n)

    i = 0
    codeword = []
    symbol_to_dealer = []
    for dealer, count in enumerate(evals):
        for _ in range(count):
            eval_point = eval_points[i]
            point = [
                eval_point,
                list(poly(eval_point) for poly in rank_to_polys[dealer]),
            ]

            codeword.append(point)
            symbol_to_dealer.append(dealer)
            i += 1

    present_ranks = {}
    rank_counts = Counter(symbol_to_dealer)
    for dealer, polys in enumerate(rank_to_polys):
        if rank_counts[dealer] >= agreement:
            present_ranks[dealer] = polys

    return Instance(
        field, pR, n, c, ell, agreement, present_ranks, codeword, symbol_to_dealer
    )


def run_zipf(field, pR, args):
    return gen_zipfian_instance(
        field,
        pR,
        args.ell,
        args.c,
        args.agreement,
        args.support,
        args.s,
        args.N,
        args.fix_max_degree,
    )


def run_malicious(field, pR, args):
    agreement = mdss_threshold(sum(args.evals), args.c, args.ell)

    return gen_malicious_client_instance(
        field, pR, args.ell, args.c, agreement, args.evals
    )


def run_adversarial(field, pR, args):
    agreement = mdss_threshold(sum(args.evals), args.c, args.ell)

    return gen_adversarial_instance(
        field,
        pR,
        args.ell,
        args.c,
        agreement,
        args.evals,
        args.favor_low_deg_polys,
        args.fix_max_degree,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate a MDSS instances")

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("filename", help="The file to write the instance to")
    parent_parser.add_argument(
        "-c", help="The MDSS c parameter. Default 200.", type=int, default=200
    )
    parent_parser.add_argument(
        "-l",
        "-ell",
        "--ell",
        help="The MDSS ell parameter. Default 349.",
        type=int,
        default=349,
    )
    parent_parser.add_argument(
        "--min-field",
        "-mf",
        help="The minimum field size. Default 100,001",
        type=int,
        default=100_001,
    )
    parent_parser.add_argument(
        "-v",
        "--variable",
        help="The polynomial variable. Default x.",
        type=str,
        default="x",
    )

    subparsers = parser.add_subparsers()

    zipf_parser = subparsers.add_parser(
        "zipf",
        aliases=["z"],
        help="Generate a Zipfian instance",
        parents=[parent_parser],
    )
    zipf_parser.set_defaults(func=run_zipf)
    zipf_parser.add_argument(
        "-N",
        help="The number of points in the instance. Default 100,000.",
        type=int,
        default=100_000,
    )
    zipf_parser.add_argument(
        "-t",
        "--agreement",
        help="The number of matching points required for a value to be considered present. Default 1000.",
        type=int,
        default=1000,
    )
    zipf_parser.add_argument(
        "-u",
        "--support",
        help="The support of the Zipfian instance. Default 10,000.",
        type=int,
        default=10_000,
    )
    zipf_parser.add_argument(
        "-s", help="The Zipf s parameter. Default 1.03.", type=float, default=1.03
    )
    zipf_parser.add_argument(
        "--fix-max-degree",
        "-fmd",
        help="Fix the coefficient of the max degree term in each poly to be 1",
        action="store_true",
    )

    malicious_parser = subparsers.add_parser(
        "malicious",
        aliases=["m"],
        help="Generate a malicious instance",
        parents=[parent_parser],
    )
    malicious_parser.set_defaults(func=run_malicious)
    malicious_parser.add_argument(
        "-e",
        "--evals",
        nargs="*",
        help="The number of points each malicious client should contribute",
        type=int,
    )

    adversarial_parser = subparsers.add_parser(
        "adversarial",
        aliases=["a"],
        help="Generate an adversarial instance",
        parents=[parent_parser],
    )
    adversarial_parser.set_defaults(func=run_adversarial)
    adversarial_parser.add_argument(
        "-e",
        "--evals",
        nargs="*",
        help="The number of points from each dealer",
        type=int,
    )
    adversarial_parser.add_argument(
        "--favor-low-deg-polys",
        "-fldp",
        help="Overweight polynomials of lower degree",
        action="store_true",
    )
    adversarial_parser.add_argument(
        "--fix-max-degree",
        "-fmd",
        help="Fix the coefficient of the max degree term in each poly to be 1",
        action="store_true",
    )

    args = parser.parse_args()

    field = GF(Integer(args.min_field).next_prime())
    pR = PolynomialRing(field, args.variable)

    inst = args.func(field, pR, args)

    with open(args.filename, "w+") as outfile:
        json.dump(inst.to_serializable(), outfile)
