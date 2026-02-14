from pathlib import Path
import pickle
import pandas as pd
from collections import defaultdict


def parse_results(root_dir, classes):
    folds_dirs = sorted(list(root_dir.glob("fold_*")))

    results = []
    attentions = defaultdict(lambda: {})

    for fold_dir in folds_dirs:
        fold = fold_dir.name.split("_")[1]
        seeds_dirs = sorted(list(fold_dir.glob("seed_*")))

        for seed_dir in seeds_dirs:
            seed = seed_dir.name.split("_")[1]

            with open(seed_dir / "predictions.pckl", "rb") as f:
                res = pickle.load(f)
            # flatten scores and logits+
            scores = res["scores"].T
            logits = res["logits"].T

            res.update(
                {f"score_{subt_name}": scr for subt_name, scr in zip(classes, scores)}
            )
            res.update(
                {f"logit_{subt_name}": lgt for subt_name, lgt in zip(classes, logits)}
            )

            del res["scores"]
            del res["logits"]

            res = pd.DataFrame(res)
            res["fold"] = fold
            res["seed"] = seed

            if "attention" in res:
                attentions[int(fold)][int(seed)] = dict(
                    zip(res["samples"], res["attention"])
                )
                del res["attention"]
            else:
                attentions = None

            results.append(res)

    results = pd.concat(results)

    if attentions:
        attentions = dict(attentions)

    return {"results": results, "attentions": attentions}
