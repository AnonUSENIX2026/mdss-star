import argparse

from sage.all import floor

import leakage_calculator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", help="The number of clients", type=int, default=100000)
    parser.add_argument("-N", help="The Zipf N parameter", type=int, default=10000)
    parser.add_argument("-s", help="The Zipf s parameter", type=float, default=1.03)
    parser.add_argument("-t", help="The threshold", type=int, default=1000)
    parser.add_argument(
        "--max-leakage",
        "-ml",
        help="The maximum number of reports allowed to leak",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--allow-low-degree",
        "-ald",
        action="store_true",
        help="Allow low degree polynomials",
    )
    parser.add_argument(
        "--max-ell", "-me", type=int, default=1000, help="The max value of ell to try"
    )

    args = parser.parse_args()

    min_ell = 1
    max_ell = args.max_ell

    while max_ell - min_ell > 1:
        print("gap:", max_ell - min_ell)

        current_ell = floor((max_ell + min_ell) / 2)
        lc, nrl = leakage_calculator.get_leakage(
            args.n, args.N, args.s, args.t, current_ell, not args.allow_low_degree
        )
        if lc > args.max_leakage:
            min_ell = current_ell
        else:
            max_ell = current_ell

    if (
        leakage_calculator.get_leakage(
            args.n, args.N, args.s, args.t, min_ell, not args.allow_low_degree
        )[0]
        <= args.max_leakage
    ):
        print("DONE", min_ell)
    else:
        print("DONE", max_ell)
