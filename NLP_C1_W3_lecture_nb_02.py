import pandas as pd # Library for Dataframes
import numpy as np # Library for math functions
import pickle # Python object serialization library. Not secure
import matplotlib.pyplot as plt # Import matplotlib

word_embeddings = pickle.load( open( "word_embeddings_subset.p", "rb" ) )
len(word_embeddings) # there should be 243 words that will be used in this assignment

def vec(w, word_embeddings=word_embeddings):
    if w in word_embeddings:
        return word_embeddings[w]
    else:
        return

def plot_words(words):
    bag2d = np.array([vec(word) for word in words]) # Convert each word to its vector representation

    fig, ax = plt.subplots(figsize = (10, 10)) # Create custom size image

    col1 = 3 # Select the column for the x axis
    col2 = 2 # Select the column for the y axis

    # Print an arrow for each word
    for word in bag2d:
        ax.arrow(0, 0, word[col1], word[col2], head_width=0.005, head_length=0.005, fc='r', ec='r', width = 1e-5)

    ax.scatter(bag2d[:, col1], bag2d[:, col2]); # Plot a dot for each word

    # Add the word label over each dot in the scatter plot
    for i in range(0, len(words)):
        ax.annotate(words[i], (bag2d[i, col1], bag2d[i, col2]))

    plt.show()

    return

def plot_sentence(words):

    bag2d = np.array([vec(word) for word in words]) # Convert each word to its vector representation

    # Calculate a vector as a sum of the words
    # then go and find the closest matching word
    doc2vec = np.sum(bag2d, axis = 0)
    closest = find_closest_word(doc2vec)

    fig, ax = plt.subplots(figsize = (10, 10)) # Create custom size image

    col1 = 3 # Select the column for the x axis
    col2 = 2 # Select the column for the y axis

    # Print an arrow for each word
    for word in bag2d:
        ax.arrow(0, 0, word[col1], word[col2], head_width=0.005, head_length=0.005, fc='r', ec='r', width = 1e-5)

    ax.scatter(bag2d[:, col1], bag2d[:, col2]); # Plot a dot for each word

    # Print an arrow for the calculated word
    ax.arrow(0, 0, doc2vec[col1], doc2vec[col2], head_width=0.005, head_length=0.005, fc='g', ec='g', width= 1e-5)
    ax.annotate(closest, (doc2vec[col1], doc2vec[col2]))

    # Add the word label over each dot in the scatter plot
    for i in range(0, len(words)):
        ax.annotate(words[i], (bag2d[i, col1], bag2d[i, col2]))

    plt.show()

    return(closest)


# Operating on word embeddings
# In the next cell, we make a beautiful plot for the word embeddings of some words. Even if plotting the dots gives
# an idea of the words, the arrow representations help to visualize the vector's alignment as well.

words = ['oil', 'gas', 'happy', 'sad', 'city', 'town', 'village', 'country', 'continent', 'petroleum', 'joyful']
plot_words(words)

# Word distance
# Now plot the words 'sad', 'happy', 'town', and 'village'. In this same chart, display the vector from
# 'village' to 'town' and the vector from 'sad' to 'happy'. Let us use NumPy for these linear algebra operations.

words = ['sad', 'happy', 'town', 'village']

bag2d = np.array([vec(word, word_embeddings) for word in words]) # Convert each word to its vector representation

fig, ax = plt.subplots(figsize = (10, 10)) # Create custom size image

col1 = 3 # Select the column for the x axe
col2 = 2 # Select the column for the y axe

# Print an arrow for each word
for word in bag2d:
    ax.arrow(0, 0, word[col1], word[col2], head_width=0.0005, head_length=0.0005, fc='r', ec='r', width = 1e-5)

# print the vector difference between village and town
village = vec('village', word_embeddings)
town = vec('town', word_embeddings)
diff = town - village
ax.arrow(village[col1], village[col2], diff[col1], diff[col2], fc='g', ec='g', width = 1e-5)

# print the vector difference between village and town
sad = vec('sad', word_embeddings)
happy = vec('happy', word_embeddings)
diff = happy - sad
ax.arrow(sad[col1], sad[col2], diff[col1], diff[col2], fc='b', ec='b', width = 1e-5)

ax.scatter(bag2d[:, col1], bag2d[:, col2]); # Plot a dot for each word

# Add the word label over each dot in the scatter plot
for i in range(0, len(words)):
    ax.annotate(words[i], (bag2d[i, col1], bag2d[i, col2]))

plt.show()

# Linear algebra on word embeddings

print(np.linalg.norm(vec('town', word_embeddings))) # Print the norm of the word town
print(np.linalg.norm(vec('sad', word_embeddings))) # Print the norm of the word sad

# Predicting capitals
capital = vec('France', word_embeddings) - vec('Paris', word_embeddings)
country = vec('Madrid', word_embeddings) + capital
print(country[0:5]) # Print the first 5 values of the vector

# print difference between country and spain
diff = country - vec('Spain', word_embeddings)
print(diff[0:10])

# Create a dataframe out of the dictionary embedding. This facilitate the algebraic operations
keys = word_embeddings.keys()
data = []
for key in keys:
    data.append(word_embeddings[key])

embedding = pd.DataFrame(data=data, index=keys)
# Define a function to find the closest word to a vector:
def find_closest_word(v, k = 1):
    # Calculate the vector difference from each word to the input vector
    diff = embedding.values - v
    # Get the norm of each difference vector.
    # It means the squared euclidean distance from each word to the input vector
    delta = np.sum(diff * diff, axis=1)
    # Find the index of the minimun distance in the array
    i = np.argmin(delta)
    # Return the row name for this item
    return embedding.iloc[i].name

embedding.head()

find_closest_word(country)

# Predicting other Countries
find_closest_word(vec('Italy') - vec('Rome') + vec('Berlin'))
print(find_closest_word(vec('Beijing') + capital))
print(find_closest_word(vec('Lisbon') + capital)) # doesn't always work

# Representing a sentence as a vector
doc = "Spain city king petroleum"
words = ['Spain', 'city', 'king', 'petroleum']

vdoc = [vec(x) for x in doc.split(" ")]
doc2vec = np.sum(vdoc, axis = 0)
print(find_closest_word(doc2vec))

words = ['Spain', 'city', 'king', 'petroleum']
print(plot_sentence(words))
