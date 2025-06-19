import os
import re
import sys
import json
import yaml
import random
import hashlib
import itertools
from collections import OrderedDict


STANCES = ["A", "B"]
POLITICAL_STANCES = {
    "A": "Left",
    "B": "Right"
}
BASE_METRICS_DIR = "[PATH TO ALIGNSCORE AND BLEURT CHECKPOINTS]"
CHAT_MODELS = ["gpt-3.5-turbo", 
               "gpt-4", 
               "gpt-4-turbo", 
               "gpt-4o", 
               "gpt-4o-mini",
               "o1-mini",
               "o1"]
GENERATION_MODELS = ["llama", "qwen", "mistral"]

log_codes = {
    "error": "\033[91m[ERROR]\033[0m",
    "warning": "\033[93m[WARNING]\033[0m",
    "status": "\033[92m[STATUS]\033[0m",
}

openai_context = {
  "gpt-3.5-turbo": {
    "context": 16385,
    "max_out": 4096
  },
  "gpt-3.5-turbo-0125": {
    "context": 16385,
    "max_out": 4096
  },
  "gpt-3.5-turbo-0301": {
    "context": 4097,
    "max_out": 4097
  },
  "gpt-3.5-turbo-0613": {
    "context": 4097,
    "max_out": 4097
  },
  "gpt-3.5-turbo-1106": {
    "context": 16385,
    "max_out": 4096
  },
  "gpt-3.5-turbo-16k": {
    "context": 16385,
    "max_out": 16385
  },
  "gpt-3.5-turbo-16k-0613": {
    "context": 16385,
    "max_out": 16385
  },
  "gpt-4": {
    "context": 8192,
    "max_out": 8192
  },
  "gpt-4-0125-preview": {
    "context": 128000,
    "max_out": 4096
  },
  "gpt-4-0314": {
    "context": 8192,
    "max_out": 8192
  },
  "gpt-4-0613": {
    "context": 8192,
    "max_out": 8192
  },
  "gpt-4-1106-preview": {
    "context": 128000,
    "max_out": 4096
  },
  "gpt-4-1106-vision-preview": {
    "context": 128000,
    "max_out": 4096
  },
  "gpt-4-32k": {
    "context": 32768,
    "max_out": 32768
  },
  "gpt-4-32k-0314": {
    "context": 32768,
    "max_out": 32768
  },
  "gpt-4-32k-0613": {
    "context": 32768,
    "max_out": 32768
  },
  "gpt-4-turbo-preview": {
    "context": 128000,
    "max_out": 4096
  },
  "gpt-4-vision-preview": {
    "context": 128000,
    "max_out": 4096
  },
  "gpt-4o-mini": {
    "context": 128000,
    "max_out": 16384
  },
}

#################
# PRETTY PRINTS #
#################
def print_error(string, flush=False):
    print(f"{log_codes['error']:<20}{string}", flush=flush)


def print_status(string, flush=False, clear=False):
    if clear:
        # Clear previous line
        sys.stdout.write("\033[F\033[K")

    print(f"{log_codes['status']:<20}{string}", flush=flush)

def pretty_print_dict(dictionary):
    max_key_length = max([len(i) for i in dictionary.keys()]) + 2
        
    print("="*80)
    for key, value in dictionary.items():
        print(f"{key:<{max_key_length}}: {value}")
    print("="*80)
    
    
####################
# HELPER FUNCTIONS #
####################
def powerset(lst):
    """
    Generate the power set of a list (all possible subsets),
    including the empty subset.
    """
    return itertools.chain.from_iterable(
        itertools.combinations(lst, r) for r in range(len(lst)+1)
    )

def generate_combinations_with_counts_single(a):
    """
    Specialized function for a single list.
    Returns all non-empty subsets and their counts.
    """
    results = []
    for r in range(1, len(a)+1):
        for combo in itertools.combinations(a, r):
            # combo is a tuple of chosen elements from a
            # count is just how many we chose from this single list
            results.append( (list(combo), (len(combo),)) )
    return results

def generate_combinations_with_counts_two(a, b):
    """
    Specialized function for two lists.
    Returns all non-empty combinations (excluding the case where both are empty)
    and their counts as (count_from_a, count_from_b).
    """
    results = []
    subsets_a = list(powerset(a))
    subsets_b = list(powerset(b))
    for sub_a in subsets_a:
        for sub_b in subsets_b:
            # Exclude the case where both subsets are empty
            if sub_a or sub_b:
                combined = list(sub_a) + list(sub_b)
                counts = (len(sub_a), len(sub_b))
                results.append((combined, counts))
    return results

def generate_combinations_with_counts_three(a, b, c):
    """
    Specialized function for three lists.
    Returns all non-empty combinations (excluding the case where all are empty)
    and their counts as (count_from_a, count_from_b, count_from_c).
    """
    results = []
    subsets_a = list(powerset(a))
    subsets_b = list(powerset(b))
    subsets_c = list(powerset(c))
    for sub_a in subsets_a:
        for sub_b in subsets_b:
            for sub_c in subsets_c:
                # Exclude the case where all three subsets are empty
                if sub_a or sub_b or sub_c:
                    combined = list(sub_a) + list(sub_b) + list(sub_c)
                    counts = (len(sub_a), len(sub_b), len(sub_c))
                    results.append((combined, counts))
    return results

def generate_combinations_with_counts_general(list_of_lists):
    """
    General function for N lists (where N >= 4, but works for any N).
    Returns all non-empty combinations and their counts.
    """
    # First, build a list of all subsets (including empty) for each list
    all_subsets = [list(powerset(lst)) for lst in list_of_lists]
    
    results = []
    # all_subsets is a list of lists-of-subsets, one for each original list
    # By using product, we get every possible combination of picking one subset from each list
    for combo_of_subsets in itertools.product(*all_subsets):
        # combo_of_subsets is a tuple like (subset_from_list1, subset_from_list2, ...)
        # We skip if *all* chosen subsets are empty
        if any(subset for subset in combo_of_subsets):
            combined = []
            counts = []
            for subset in combo_of_subsets:
                combined += subset
                counts.append(len(subset))
            results.append((combined, tuple(counts)))
    return results

def generate_combinations_with_counts(list_of_lists):
    """
    Wrapper function that decides which specialized function to use
    based on the number of lists provided.
    """
    n = len(list_of_lists)
    
    if n == 0:
        # No lists? Return empty result
        return []
    elif n == 1:
        return generate_combinations_with_counts_single(list_of_lists[0])
    elif n == 2:
        return generate_combinations_with_counts_two(list_of_lists[0], list_of_lists[1])
    elif n == 3:
        return generate_combinations_with_counts_three(list_of_lists[0], 
                                                       list_of_lists[1], 
                                                       list_of_lists[2])
    else:
        # 4 or more lists, use the general approach
        return generate_combinations_with_counts_general(list_of_lists)

def remove_numbered_lists(text):
    """
    Removes numbered lists (e.g., '1.', '2)', 'I.', 'a)', etc.) from a string.

    Parameters:
        text (str): The input string containing potential numbered lists.

    Returns:
        str: The string with numbered lists removed.
    """
    # Pattern to match numbered list formats
    pattern = r"""
        (?:             # Non-capturing group for the list number format
            ^\s*        # Start of line with optional leading spaces
            (\d+|       # Digits (e.g., 1, 2, 123)
             [ivxlcdm]+| # Roman numerals (e.g., i, ii, iii, iv, etc.)
             [a-zA-Z])   # Letters (e.g., a, b, c)
            [\.\)\:]    # Followed by '.', ')' or ':'
            \s+         # At least one space after the number
        )
        (.+?)           # Content after the list number
        (?=$|\n)        # End of line or newline
    """
    # Remove matches from the text
    clean_text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.VERBOSE | re.MULTILINE)
    return clean_text.strip()


def is_box_empty_except(box, excluded_key):
    # Create a copy of the Box without the excluded key
    filtered_box = {k: v for k, v in box.items() if k != excluded_key}
    # Check if the remaining items are empty
    return len(filtered_box) == 0


def clean_summary(summary: str, stance: str) -> tuple:
    """
    Cleans and reformats a summary based on the given stance.

    Depending on the stance ("left" or "right"), this function extracts the relevant
    portion of the summary and replaces stance-specific terms with "The article".
    It also removes unnecessary newlines and verifies that the processed summary
    starts with "The article".

    Args:
        summary (str): The summary text to be cleaned.
        stance (str): The stance to process, either "left" or "right".

    Returns:
        tuple: A tuple containing:
            - str: The cleaned summary, or an empty string if an error occurs.
            - bool: True if the cleaned summary starts with "The article", False otherwise.

    Raises:
        ValueError: If the stance is invalid or required keywords are not found.
    """
    try:
        if stance == "left":
            if "The Left" not in summary:
                raise ValueError("The keyword 'The Left' is not present in the summary.")
            summary = summary[summary.index("The Left"):]
            summary = summary.replace("The Left", "The article")
        elif stance == "right":
            if "The Right" not in summary:
                raise ValueError("The keyword 'The Right' is not present in the summary.")
            summary = summary[summary.index("The Right"):]
            summary = summary.replace("The Right", "The article")
        else:
            raise ValueError("Invalid stance. Must be 'left' or 'right'.")

        if "\n\n" in summary:
            summary = summary[summary.index("\n\n") + 2:]

        starts_with_article = summary.startswith("The article")
        return summary, starts_with_article

    except Exception as e:
        return "", False


alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'

def split_into_sentences(text):
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead 
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    if "Sen." in text: text = text.replace("Sen.","Sen<prd>")        
    text = re.sub(r"\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences

def set_seeds(seed):
    import numpy as np
    import torch    
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_yaml_hash(yaml_content, length=16):
    if type(yaml_content) == str and yaml_content.lower().endswith(('.yaml', '.yml')):
        loaded_yaml = yaml.safe_load(yaml_content)
        normalized_yaml_str = yaml.dump(loaded_yaml, sort_keys=True)
    else:
        normalized_yaml_str = str(yaml_content)
        
    hash_obj = hashlib.sha256(normalized_yaml_str.encode())
    pseudorandom_string = hash_obj.hexdigest()[:length]
    
    return pseudorandom_string

def partition(obj, num_partitions):
    """
    Splits an iterable into a specified number of partitions.

    This function divides an iterable into a specified number of roughly equal partitions.
    If the iterable cannot be evenly divided, the remaining elements are distributed among the partitions.

    Args:
        obj (iterable): The iterable to be partitioned.
        num_partitions (int): The number of partitions to divide the iterable into.

    Returns:
        list of lists: A list containing the partitions, each of which is a list of elements.
    """
    # Calculate the base chunk size and the remainder
    chunks = len(obj) // num_partitions
    remainder = len(obj) % num_partitions
    
    chunk_list = []
    start = 0
    
    for i in range(num_partitions):
        # Calculate the size for this partition, adding 1 if remainder > 0
        chunk_size = chunks + (1 if i < remainder else 0)
        chunk_list.append(obj[start:start + chunk_size])
        start += chunk_size
    
    return chunk_list

def raise_error(error_msg):
    """
    Prints an error message in red and stops the program.

    This function prints the provided error message to the console with ANSI escape codes
    to display the message in red, indicating an error. It then raises a SystemExit exception
    to terminate the program.

    Args:
        error_msg (str): The error message to be displayed.

    Raises:
        SystemExit: Always raised to terminate the program after printing the error message.

    Example:
        if not is_json_file("path/to/file.json"):
            raise_error("The provided file is not a valid JSON file.")
    """
    print(f"[\033[91mERROR\033[0m]: {error_msg}")
    raise SystemExit

def is_json_file(file_path: str) -> bool:
    """
    Checks if the given file is a valid JSON file.

    Args:
        file_path (str): The path to the file to be checked.

    Returns:
        bool: True if the file is a valid JSON file, False otherwise.

    Raises:
        ValueError: If the file is not a valid JSON file.
    """
    try:
        with open(file_path, 'r') as file:
            json.load(file)
        return True
    except json.JSONDecodeError:
        raise ValueError("The file is not a valid JSON file.")
    except FileNotFoundError:
        raise ValueError("The file does not exist.")


def is_yaml_file(file_path: str) -> bool:
    """Check if a given string represents a path to a YAML file.

    This function verifies if the provided string ends with a valid YAML file
    extension (e.g., .yaml or .yml) and checks if it points to an existing file.

    Args:
        file_path (str): The path to check.

    Returns:
        bool: True if the path is to a YAML file, False otherwise.

    Examples:
        >>> is_yaml_file("config.yaml")
        True
        >>> is_yaml_file("/path/to/file.txt")
        False
        >>> is_yaml_file("/nonexistent/config.yml")
        False
    """
    valid_extensions = {".yaml", ".yml"}
    return os.path.isfile(file_path) and os.path.splitext(file_path)[1].lower() in valid_extensions


def has_config_yaml(folder_path: str) -> bool:
    """Check if the given path is a folder and contains a config.yaml file.

    Args:
        folder_path (str): The path to the folder to check.

    Returns:
        bool: True if the folder exists and contains a config.yaml file, False otherwise.

    Examples:
        >>> has_config_yaml("/path/to/folder")
        True
        >>> has_config_yaml("/nonexistent/folder")
        False
    """
    if not os.path.isdir(folder_path):
        return False

    config_path = os.path.join(folder_path, "config.yaml")
    return os.path.isfile(config_path)


def model_type(value):
    """
    Validate that the input follows the format [MODEL_NAME]:[VERSION].
    
    Args:
        value (str): The input string to be validated.
    
    Returns:
        str: The validated input string.
    """
    if not re.match(r'^[^:]+:[^:]+$', value):
        raise_error("Invalid format in argparse. Expected format: [MODEL_NAME]:[VERSION].")
    return value

# Custom loader to preserve the order
def ordered_load(stream, Loader=yaml.SafeLoader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)

# Custom dumper to preserve the order
def ordered_dump(data, stream=None, Dumper=yaml.SafeDumper, **kwargs):
    class OrderedDumper(Dumper):
        pass
    def _dict_representer(dumper, data):
        return dumper.represent_mapping(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items())
    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwargs)

def random_subset(strings):
    # Return an empty list if only one string is in the list
    if len(strings) <= 1:
        return []
    
    # Choose a random length for the subset that is less than the length of the original list
    subset_length = random.randint(1, len(strings) - 1)
    
    # Get a random subset of the specified length
    subset = random.sample(strings, subset_length)
    
    return subset
  
def get_perspectives(text):
    """
    Extracts perspectives labeled as 'Left' and 'Right' from the given text.

    This function assumes the input text follows a specific structure:
        Instruction
        Left:
        [text]
        Right:
        [text]

    The text following "Left:" is extracted as the "left" perspective, and
    the text following "Right:" is extracted as the "right" perspective.
    
    Args:
        text (str): The input text containing labeled perspectives.

    Returns:
        dict: A dictionary with 'left' and 'right' keys containing respective
              perspectives as strings.
    """
    lines = text.split("\n")
    left_start = lines.index("Left: ") + 1
    right_start = lines.index("Right:") + 1

    left = "\n".join(line for line in lines[left_start:right_start - 1] if line).strip()
    right = "\n".join(line for line in lines[right_start:] if line).strip()
    
    return {"left": left, "right": right}

# For Binning
import numpy as np

def sturges_rule(data):
    """
    Calculate the number of bins using Sturges' Formula.
    Args:
        data (list or numpy array): The dataset.
    Returns:
        int: Number of bins.
    """
    n = len(data)
    if n <= 0:
        raise ValueError("Data must contain at least one value.")
    return int(np.ceil(np.log2(n) + 1))


def rice_rule(data):
    """
    Calculate the number of bins using the Rice Rule.
    Args:
        data (list or numpy array): The dataset.
    Returns:
        int: Number of bins.
    """
    n = len(data)
    if n <= 0:
        raise ValueError("Data must contain at least one value.")
    return int(np.ceil(2 * n**(1/3)))


def freedman_diaconis_rule(data):
    """
    Calculate the number of bins using the Freedman-Diaconis Rule.
    Args:
        data (list or numpy array): The dataset.
    Returns:
        int: Number of bins.
    """
    if len(data) < 2:
        raise ValueError("Data must contain at least two values.")
    
    # Compute IQR
    data = np.array(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    if iqr == 0:
        raise ValueError("IQR is zero; Freedman-Diaconis Rule is not applicable.")
    
    # Bin width
    n = len(data)
    bin_width = 2 * (iqr / n**(1/3))
    if bin_width == 0:
        raise ValueError("Bin width is zero; Freedman-Diaconis Rule is not applicable.")
    
    # Number of bins
    data_range = data.max() - data.min()
    return int(np.ceil(data_range / bin_width))


def scotts_rule(data):
    """
    Calculate the number of bins using Scott's Rule.
    Args:
        data (list or numpy array): The dataset.
    Returns:
        int: Number of bins.
    """
    if len(data) < 2:
        raise ValueError("Data must contain at least two values.")
    
    # Compute standard deviation
    data = np.array(data)
    std_dev = np.std(data)
    
    if std_dev == 0:
        raise ValueError("Standard deviation is zero; Scott's Rule is not applicable.")
    
    # Bin width
    n = len(data)
    bin_width = 3.5 * (std_dev / n**(1/3))
    if bin_width == 0:
        raise ValueError("Bin width is zero; Scott's Rule is not applicable.")
    
    # Number of bins
    data_range = data.max() - data.min()
    return int(np.ceil(data_range / bin_width))
