import pandas as pd

import sdsModels as sdsm


sequences = [
    [
        # Four sequences. State 1 always emits 1. State 2 always emits 2.
        (1, 1, 1),
        (1, 2, 2),
        (1, 1, 1),
        (1, 2, 2),

        (2, 1, 1),
        (2, 2, 2),
        (2, 1, 1),
        (2, 2, 2),

        (3, 1, 1),
        (3, 2, 2),
        (3, 1, 1),
        (3, 2, 2),

        (4, 1, 1),
        (4, 2, 2),
        (4, 1, 1),
        (4, 2, 2)
    ],
    [
        # Four sequences. State 1 emits 1 with 50% pr, 2 with 50% pr. State 2 same. Same order of state/emm in each
        # sequence.
        (1, 1, 1),
        (1, 2, 2),
        (1, 1, 2),
        (1, 2, 1),

        (2, 1, 1),
        (2, 2, 2),
        (2, 1, 2),
        (2, 2, 1),

        (3, 1, 1),
        (3, 2, 2),
        (3, 1, 2),
        (3, 2, 1),

        (4, 1, 1),
        (4, 2, 2),
        (4, 1, 2),
        (4, 2, 1)
    ],
    [
        # Four sequences. Mixed emissions.
        (1, 1, 1),
        (1, 2, 2),
        (1, 1, 3),
        (1, 2, 4),

        (2, 1, 4),
        (2, 2, 3),
        (2, 1, 2),
        (2, 2, 1),

        (3, 1, 2),
        (3, 2, 4),
        (3, 1, 3),
        (3, 2, 1),

        (4, 1, 3),
        (4, 2, 4),
        (4, 1, 1),
        (4, 2, 2)
    ],
    [
        # Each state with unique emission sequence.
        (1, 1, 1),
        (1, 1, 2),
        (1, 1, 3),

        (2, 2, 3),
        (2, 2, 2),
        (2, 2, 1),

        (3, 3, 1),
        (3, 3, 3),
        (3, 3, 2),

        (4, 4, 2),
        (4, 4, 3),
        (4, 4, 1),

        (5, 5, 2),
        (5, 5, 1),
        (5, 5, 3),

        (6, 6, 3),
        (6, 6, 1),
        (6, 6, 2)

    ],
    [
        # Two identical sequences.
        (1, 1, 1),
        (1, 2, 2),
        (1, 3, 3),

        (2, 1, 1),
        (2, 2, 2),
        (2, 3, 3)
    ],
    [
        # "Missing" states 1 [x] 3 [x] [x] 6
        (1, 1, 1),
        (1, 3, 2),
        (1, 6, 3),

        (2, 1, 1),
        (2, 3, 2),
        (2, 6, 3)
    ],
    [
        # "Missing" emissions 1 [x] 3 [x] [x] 6
        (1, 1, 1),
        (1, 2, 3),
        (1, 3, 6),

        (2, 1, 1),
        (2, 2, 3),
        (2, 3, 6)
    ],
    [
        # "Missing" states & emissions 1 [x] 3 [x] [x] 6
        (1, 3, 1),
        (1, 1, 3),
        (1, 6, 6),

        (2, 3, 1),
        (2, 1, 3),
        (2, 6, 6)
    ],
    [
        # Multiple emissions
        (1, 1, 1, 6),
        (1, 2, 3, 1),
        (1, 3, 6, 3),

        (2, 1, 1, 6),
        (2, 2, 3, 1),
        (2, 3, 6, 3)
    ],
    [
        # Multiple emissions, unique per sequence
        (1, 1, 1, 4, 7),
        (1, 2, 2, 5, 8),
        (1, 3, 3, 6, 9),

        (2, 1, 10, 13, 16),
        (2, 2, 11, 14, 17),
        (2, 3, 12, 15, 18)
    ]
]

for seq in sequences:
    num_features = len(seq[0]) - 2
    head = ["label", "rating"]
    feats = []
    for f in xrange(0, num_features):
        feats.append(str(f))
    data = pd.DataFrame(seq, columns=head + feats)
    exp = sdsm.Experiment(data=data)
    states = data["rating"].unique()
    exp.addModel(sdsm.Hmm({'states': states}, verbose=True))
    exp.generateResults(feats, cvMethod="loo")
    exp.printResults(['model', 'accuracy', 'r2'])
