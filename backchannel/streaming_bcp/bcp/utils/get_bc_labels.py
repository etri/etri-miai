import argparse
import os
import bcp.const.const as C

parser = argparse.ArgumentParser(description="Get BC labels")
parser.add_argument("--dataset_name", type=str, default="swbd", help="Dataset")
parser.add_argument(
    "--category_label", type=str, default="merge", help="Whether to expand labels"
)
parser.add_argument("--output", type=str, default="label.txt", help="Output file")
args = parser.parse_args()

bc_categories = ["<blank>", "<unk>"]

const_bc_categories = (
    C.SWBD_BC_CATEGORIES if args.dataset_name == "swbd" else C.BC_CATEGORIES
)

if args.category_label == "merge":
    for x in const_bc_categories.values():
        if x[0] in bc_categories:
            continue
        bc_categories.append(x[0])
elif args.category_label == "binary":
    for x in const_bc_categories.values():
        if x[2] in bc_categories:
            continue
        bc_categories.append(x[2])
else:
    bc_categories += [x[1] for x in const_bc_categories.values()]

if os.path.exists(args.output):
    with open(args.output, "r") as f:
        lines = [x.strip() for x in f.readlines() if x.strip() != ""]

        if len(lines) == len(bc_categories) and all(
            [x == y for x, y in zip(lines, bc_categories)]
        ):
            print("Labels already exist")
            exit(0)
        else:
            print("Labels already exist but not the same. Overwriting...")

else:
    dirname = os.path.dirname(args.output)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

with open(args.output, "w") as f:
    f.write("\n".join(bc_categories))
    f.write("\n")
print(
    f"Labels are saved to {args.output} successfully. Detailed labels are as follows:"
)
print("\n".join(bc_categories))
