# %%
import pandas as pd

import numpy as np

corrs = pd.read_parquet("data/correlations/Correlation Matrix.parquet").to_numpy()


# %%
ref_layer = 5
ref_neuron = 100

index_to_neuron_id = lambda x: "{}_FC1_{}".format(x // 1024, x % 1024)
neuron_id_to_index = lambda neuron_id: int(neuron_id.split("_FC1_")[0]) * 1024 + int(
    neuron_id.split("_FC1_")[1]
)

# index_to_neuron_id(1023)
neuron_id_to_index("5_FC1_10000")


res = []
for i in range(corrs.shape[0]):
    sorted_indices = np.argsort(corrs[i])[::-1]
    min_index = (i // 1024) * 1024
    top_indices_upstream = sorted_indices[sorted_indices < min_index][:10]
    top_scores = corrs[i][top_indices_upstream]
    res.extend(
        [
            (index_to_neuron_id(i), index_to_neuron_id(index), score, True)
            for index, score in zip(top_indices_upstream, top_scores)
        ]
    )

    max_index = ((i // 1024) + 1) * 1024
    top_indices_downstream = sorted_indices[sorted_indices > max_index][:10]
    top_scores = corrs[i][top_indices_downstream]
    res.extend(
        [
            (index_to_neuron_id(i), index_to_neuron_id(index), score, False)
            for index, score in zip(top_indices_downstream, top_scores)
        ]
    )

df_corrs = pd.DataFrame(
    res, columns=["startNeuronId", "endNeuronId", "corr", "isUpstream"]
)
len(df_corrs)

# %%
df_corrs.to_csv("db_outputs/neuron-corrs.csv", index=False)
