# %%
# Load correlation matrix and write to "db" file
# IN PROGRESS
import pandas as pd

import numpy as np

corrs = pd.read_parquet("data/correlations/Correlation Matrix.parquet").to_numpy()

index_to_neuron_id = lambda x: "{}_FC1_{}".format(x // 1024, x % 1024)
neuron_id_to_index = lambda neuron_id: int(neuron_id.split("_FC1_")[0]) * 1024 + int(
    neuron_id.split("_FC1_")[1]
)

res = []
for i in range(corrs.shape[0]):
    sorted_indices = np.argsort(corrs[i])[::-1]
    prev_layer_index = (i // 1024) * 1024
    top_indices_upstream = sorted_indices[sorted_indices < prev_layer_index][:10]
    top_scores = corrs[i][top_indices_upstream]
    res.extend(
        [
            (index_to_neuron_id(i), index_to_neuron_id(index), score, "LOWER")
            for index, score in zip(top_indices_upstream, top_scores)
        ]
    )

    next_layer_index = ((i // 1024) + 1) * 1024
    top_indices_downstream = sorted_indices[sorted_indices > next_layer_index][:10]
    top_scores = corrs[i][top_indices_downstream]
    res.extend(
        [
            (index_to_neuron_id(i), index_to_neuron_id(index), score, "HIGHER")
            for index, score in zip(top_indices_downstream, top_scores)
        ]
    )

    top_indices_same_layer = sorted_indices[
        (sorted_indices >= prev_layer_index)
        & (sorted_indices <= next_layer_index)
        & (sorted_indices != i)
    ][:10]
    top_scores = corrs[i][top_indices_same_layer]
    res.extend(
        [
            (index_to_neuron_id(i), index_to_neuron_id(index), score, "SAME")
            for index, score in zip(top_indices_same_layer, top_scores)
        ]
    )


df_corrs = pd.DataFrame(
    res, columns=["startNeuronId", "endNeuronId", "corr", "layerLocation"]
)
len(df_corrs)

df_corrs.to_csv("db_outputs/neuron-corrs.csv", index=False)
