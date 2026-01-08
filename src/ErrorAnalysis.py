def load_errors(
        path: str
        ) -> set[str]:
    """
    Reads an error file produced during classification e.g., false positives or false negatives
    and extracts only the sentence text.
    Duplicate sentences could be removed later by storing them in a set.

    Parameters
    ----------
    path : str
        Path to the error log file containing misclassified examples.

    Returns
    -------
    set of str
        A set of all unique sentences that were misclassified by the model.
    """
    sentences = set()
    with open(path, "r") as file:
        for line in file:
            # Select only sentence part
            sentence = line.split("):", 1)[1].strip()
            sentences.add(sentence)
    return sentences

if __name__ == "__main__":

    # Load all errors
    nb_fn = load_errors("error/nb_fn.txt")
    nb_fp = load_errors("error/nb_fp.txt")
    rd_fn = load_errors("error/rd_fn.txt")
    rd_fp = load_errors("error/rd_fp.txt")

    # Combine Naive Bayes errors
    nb_total_errors = nb_fn | nb_fp

    # Combine Rule-based Dictionary errors
    rd_total_errors = rd_fn | rd_fp

    # Compare
    nb_only = nb_total_errors - rd_total_errors
    rd_only = rd_total_errors - nb_total_errors
    common_errors = nb_total_errors & rd_total_errors

    print("Naive Bayes total errors:", len(nb_total_errors))
    print("Dictionary total errors :", len(rd_total_errors))

    print("\nErrors only Naive Bayes has:", len(nb_only))
    print("Errors only Dictionary has :", len(rd_only))
    print("Errors both have:", len(common_errors))

    # Write to files
    with open("error/common_errors.txt", "w") as file:
        for sentence in common_errors:
            file.write(sentence + "\n")
    with open("error/nb_only_errors.txt", "w") as file:
        for sentence in nb_only:
            file.write(sentence + "\n")
    with open("error/rd_only_errors.txt", "w") as file:
        for sentence in rd_only:
            file.write(sentence + "\n")