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
    args = parser.parse_args(args)

    with open(args.input, "r") as fp:
        input_data = json.load(fp)

    output = dict()

    results = np.asarray(input_data["outcomes"])
    povm_matrix = np.asarray(input_data["povm_matrix"])

    n_qubits = int(np.log2(povm_matrix.shape[-1]) / 2)
    channel = qp.channel.depolarizing(n_qubits=n_qubits)
    input_states = [qp.Qobj(bloch) for bloch in input_data["input_states"]]
    tmg = qp.ProcessTomograph(channel, input_states=input_states)
    tmg.experiment(1000, "proj-set")
    tmg.results = results
    output["process"] = list(tmg.point_estimate(cptp=False).choi.bloch)

    if "target_process" in input_data:
        target_process = qp.Channel(qp.Qobj(input_data["target_process"]))
        interval = qp.MomentFidelityProcessInterval(tmg, target_process=target_process)
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
