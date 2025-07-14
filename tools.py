import pandas as pd

def sample_personas(df, sample_sizes, random_state=42):
    """
    Samples a custom number of rows from the DataFrame for each specified opinion.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - sample_sizes (dict): A dictionary where keys are opinions (-2, -1, 0, 1, 2)
                           and values are the number of rows to sample for that opinion.
    - random_state (int): Random state for reproducibility.

    Returns:
    - pd.DataFrame: A DataFrame containing the sampled rows.
    """
    sampled_dfs = []
    # Iterate over each target opinion
    for opinion in [-2,-1, 0, 1,2]:
        subset = df[df["Opinion"] == opinion]
        n = sample_sizes.get(opinion, 0)
        if n > len(subset):
            print(f"Warning: Only {len(subset)} rows available for opinion {opinion}, but {n} requested.")
            sampled = subset
        else:
            sampled = subset.sample(n=n, random_state=random_state)
        sampled_dfs.append(sampled)
    return pd.concat(sampled_dfs, ignore_index=True)
