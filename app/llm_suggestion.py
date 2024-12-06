import openai

def get_llm_suggestions(api_key, result_df, num_samples=3):
    """
    Use OpenAI's API to suggest the best samples to test in the lab.

    Args:
        api_key (str): The user's OpenAI API key.
        result_df (pd.DataFrame): The dataframe with results (Utility, Novelty, etc.).
        num_samples (int): Number of best samples to suggest.

    Returns:
        list: List of suggested samples.
    """
    prompt = f"""
    You are an expert in material discovery. From the following table, suggest the {num_samples} best samples to test in the lab.
    Use criteria such as high utility, low uncertainty, and high novelty for selection. Respond with the sample IDs only.

    Table:
    {result_df.to_string(index=False)}

    Suggest {num_samples} samples to test:
    """

    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for material discovery."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )

    # Parse response
    suggestions = response["choices"][0]["message"]["content"]
    return suggestions.strip().split("\n")
