# ============================================================
# ORGANOS HIV MORGAN FP REPRODUCIBLE
# ============================================================

!pip -q install rdkit

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ============================================================
# CONFIG
# ============================================================

SEED = 42
np.random.seed(SEED)

FP_RADIUS = 2
FP_BITS = 2048

BRANCH = 4096
ECHO = 16
DEPTH = 12

QUERY_COUNT = 100

# ============================================================
# LOAD HIV
# ============================================================

URL = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/HIV.csv"

df = pd.read_csv(URL)

df = df[["smiles", "HIV_active"]].dropna()

df.columns = [
    "smiles",
    "label"
]

df["label"] = (
    df["label"]
    .astype(int)
)

df = df[
    df["label"].isin([0,1])
]

print("Rows:", len(df))
print("Actives:", int(df["label"].sum()))

# ============================================================
# MORGAN FP
# ============================================================

def morgan_fp(smiles):

    mol = Chem.MolFromSmiles(
        str(smiles)
    )

    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=FP_RADIUS,
        nBits=FP_BITS
    )

    arr = np.zeros(
        (FP_BITS,),
        dtype=np.uint8
    )

    DataStructs.ConvertToNumpyArray(
        fp,
        arr
    )

    return arr

fps = []
labels = []

for _,row in tqdm(
    df.iterrows(),
    total=len(df)
):

    fp = morgan_fp(
        row["smiles"]
    )

    if fp is None:
        continue

    fps.append(fp)
    labels.append(
        row["label"]
    )

X = np.vstack(fps).astype(
    np.float32
)

y = np.array(labels)

N = len(y)

print("Usable:", N)

# ============================================================
# QUERY SET
# ============================================================

active_idx = np.where(
    y == 1
)[0]

query_idx = np.random.choice(
    active_idx,
    size=min(
        QUERY_COUNT,
        len(active_idx)
    ),
    replace=False
)

# ============================================================
# TANIMOTO
# ============================================================

def tanimoto_scores(q,m):

    inter = m @ q

    union = (
        m.sum(axis=1)
        + q.sum()
        - inter
    )

    return inter / np.maximum(
        union,
        1e-9
    )

# ============================================================
# THERMAL ECHO
# ============================================================

def thermal_echo(scores):

    pool = np.arange(
        len(scores)
    )

    for _ in range(DEPTH):

        if len(pool) <= BRANCH:
            break

        order = pool[
            np.argsort(
                scores[pool]
            )[::-1]
        ]

        chunks = np.array_split(
            order,
            BRANCH
        )

        survivors = []

        for ch in chunks:

            if len(ch) == 0:
                continue

            survivors.append(
                ch[
                    np.argmax(
                        scores[ch]
                    )
                ]
            )

        pool = np.array(
            survivors,
            dtype=np.int32
        )

    order = pool[
        np.argsort(
            scores[pool]
        )[::-1]
    ]

    main = order[:1]

    echo = order[
        1:1+ECHO
    ]

    active = np.unique(
        np.concatenate(
            [main,echo]
        )
    )

    ranked = active[
        np.argsort(
            scores[active]
        )[::-1]
    ]

    return ranked,len(active)

# ============================================================
# METRICS
# ============================================================

def hit_at_k(
    labels,
    ranked,
    k
):
    ranked = ranked[:k]

    return float(
        labels[ranked].sum() > 0
    )

def ndcg_at_k(
    labels,
    ranked,
    k
):

    ranked = ranked[:k]

    gains = labels[
        ranked
    ]

    dcg = np.sum(
        gains /
        np.log2(
            np.arange(
                2,
                len(gains)+2
            )
        )
    )

    ideal = np.sort(
        labels
    )[::-1][:k]

    idcg = np.sum(
        ideal /
        np.log2(
            np.arange(
                2,
                len(ideal)+2
            )
        )
    )

    if idcg == 0:
        return 0

    return dcg/idcg

# ============================================================
# RUN
# ============================================================

hit10 = []
hit50 = []
ndcg10 = []
active_cnt = []

for qi in tqdm(query_idx):

    q = X[qi]

    scores = tanimoto_scores(
        q,
        X
    )

    scores[qi] = -1

    ranked,active = (
        thermal_echo(
            scores
        )
    )

    hit10.append(
        hit_at_k(
            y,
            ranked,
            min(
                10,
                len(ranked)
            )
        )
    )

    hit50.append(
        hit_at_k(
            y,
            ranked,
            min(
                50,
                len(ranked)
            )
        )
    )

    ndcg10.append(
        ndcg_at_k(
            y,
            ranked,
            min(
                10,
                len(ranked)
            )
        )
    )

    active_cnt.append(
        active
    )

print("\n" + "="*80)
print("HIV MORGAN FP RESULT")
print("="*80)

print("Molecules:", N)
print("Avg Active:", np.mean(active_cnt))
print("Compression:", N/np.mean(active_cnt))
print("Hit@10:", np.mean(hit10))
print("Hit@50:", np.mean(hit50))
print("nDCG@10:", np.mean(ndcg10))