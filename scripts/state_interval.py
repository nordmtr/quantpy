import json
from argparse import ArgumentParser
from pprint import pprint

import numpy as np

import quantpy as qp


def main(args=None):
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="path to input data file",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        type=str,
        help="path to output file",
        required=False,
    )
    parser.add_argument(
        "--no-ci",
        default=False,
        action="store_true",
        help="removes confidence intervals",
    )
    args = parser.parse_args(args)

    with open(args.input, "r") as fp:
        input_data = json.load(fp)

    output = dict()

    results = np.asarray(input_data["outcomes"])
    povm_matrix = np.asarray(input_data["povm_matrix"])

    n_qubits = int(np.log2(povm_matrix.shape[-1]) / 2)
    state = qp.qobj.fully_mixed(n_qubits)
    tmg = qp.StateTomograph(state)
    tmg.experiment(1000, povm_matrix)
    tmg.results = results
    output["state"] = list(tmg.point_estimate(physical=False).bloch)

    if not args.no_ci:
        if "target_state" in input_data:
            target_state = qp.Qobj(input_data["target_state"])
            interval = qp.MomentFidelityStateInterval(tmg, target_state=target_state)
            interval.setup()
            (fidelity_min, fidelity_max), _ = interval(input_data["conf_levels"])
            output["fidelity_min"] = list(np.maximum(fidelity_min, 0))
            output["fidelity_max"] = list(np.minimum(fidelity_max, 1))
        else:
            interval = qp.MomentInterval(tmg)
            interval.setup()

        dist = interval.cl_to_dist(input_data["conf_levels"])
        output["hs_radius"] = list(dist)
    if args.output:
        with open(args.output, "w") as fp:
            json.dump(output, fp, indent=4)
        return
    pprint(output)


if __name__ == "__main__":
    main()
