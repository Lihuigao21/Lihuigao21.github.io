"""Plot the recorded numerical controls; this does not rerun dynamics."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    data = json.loads(args.data.read_text(encoding="utf-8"))
    q, dt, comparison = data["quantum_controls"], data["timestep_controls"], data["comparison"]
    labels = ["Quantum\nstep", "Quantum\ngrid", "Quantum\nbox", "CPS W\nstep", "CPS half\nstep", "W\nbasis", "Half\nbasis"]
    values = [q[x]["population_max_difference"] for x in ("dt", "grid", "box")]
    values += [dt[x]["population_dt_difference"] for x in ("W", "half")]
    values += [comparison[x]["population_basis_max"] for x in ("W", "half")]
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig, ax = plt.subplots(figsize=(9.5,3.8), constrained_layout=True)
    ax.bar(labels, values, color=["#64748b"]*3+["#0077AA","#C65D24"]*2)
    ax.axhline(2e-4,color="black",ls=":",label="Declared population tolerance")
    ax.set(yscale="log",ylim=(1e-14,1e-3),ylabel="Maximum population difference",
        title="Discretization and representation controls")
    ax.grid(axis="y",alpha=.2)
    ax.legend(loc="lower right",fontsize=9)
    args.output.mkdir(parents=True,exist_ok=True)
    for suffix in ("png","svg"):
        path=args.output/f"cps-basis-controls.{suffix}"
        fig.savefig(path,dpi=180)
        if suffix=="svg":
            text="\n".join(x.rstrip() for x in path.read_text(encoding="utf-8").splitlines())+"\n"
            path.write_bytes(text.encode("utf-8"))
    plt.close(fig)


if __name__=="__main__":
    main()
