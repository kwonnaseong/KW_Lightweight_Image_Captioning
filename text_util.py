import string

# Load the document in memory
def load_doc(filename):
    file = open(filename, 'r')
    all_text = file.read()
    file.close()
    return all_text


# Extract descriptions from loaded document
def load_descriptions(doc):
    description_map = dict()

    # Process per line
    for line in doc.split('\n'):

        # White space split
        tokens = line.split()

        if len(line) < 2:
            continue

        # Image ID, Image Description
        image_id, image_desc = tokens[0], tokens[1:]

        # Removing filename from Image ID
        image_id = image_id.split('.')[0]

        # De-tokenize Description by converting back to string
        image_desc = ' '.join(image_desc)

        # If needed, create list.
        if image_id not in description_map:
            description_map[image_id] = list()

        # Store description
        description_map[image_id].append(image_desc)
    return description_map


# Clean the descriptions
def clean_descriptions(descriptions):

    # Translation table
    table = str.maketrans('', '', string.punctuation)

    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):

            desc = desc_list[i]

            # Create tokens
            desc = desc.split()

            # Lower Case
            desc = [word.lower() for word in desc]

            # Remove punctuation
            desc = [w.translate(table) for w in desc]

            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word) > 1]

            # Remove tokens with numbers
            desc = [word for word in desc if word.isalpha()]

            # Convert to string
            desc_list[i] = ' '.join(desc)


# Convert Description to vocabulary
def to_vocabulary(descriptions):

    # List of Descriptions
    all_desc = set()

    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]

    return all_desc


# Save descriptions
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


def main():

    filename = 'data/Flickr8k_text/Flickr8k.token.txt'

    # Load
    doc = load_doc(filename)

    # Parse
    descriptions = load_descriptions(doc)
    print('Loaded:\t' + str(len(descriptions)))

    # Clean
    clean_descriptions(descriptions)

    # Summarize
    vocabulary = to_vocabulary(descriptions)
    print('Vocabulary Size:\t' + str(len(vocabulary)))

    # Save
    save_descriptions(descriptions, 'data/descriptions.txt')

if __name__ == "__main__":
    main()
