import random
from Bio import pairwise2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def generate_sequence(n, is_protein=False):
    alphabet = "ACDEFGHIKLMNPQRSTVWY" if is_protein else "ACGT"
    return ''.join(random.choice(alphabet) for _ in range(n))

def mutate_sequence(sequence, mutation_rate):
    mutated_sequence = list(sequence)
    for i in range(len(mutated_sequence)):
        if random.random() < mutation_rate:
            mutated_sequence[i] = random.choice("ACDEFGHIKLMNPQRSTVWY")
    return ''.join(mutated_sequence)

def compute_alignment_score(seq1, seq2):
    alignments = pairwise2.align.globalxx(seq1, seq2)
    alignment_score = alignments[0].score
    return alignment_score

def create_similar_pairs(num_pairs, seq_length, mutation_rate_range=(0.1, 0.3), is_protein=False):
    pairs = []
    for _ in range(num_pairs):
        seq1 = generate_sequence(seq_length, is_protein)
        mutation_rate = random.uniform(*mutation_rate_range)
        seq2 = mutate_sequence(seq1, mutation_rate)
        alignment_score = compute_alignment_score(seq1, seq2)
        pairs.append((seq1, seq2, alignment_score))
    return pairs

def build_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_dataset(pairs):
    X_data, y_data = [], []
    for pair in pairs:
        seq1, seq2, alignment_score = pair
        dot_plot_matrix = compute_dotplot_matrix(seq1, seq2)
        X_data.append(dot_plot_matrix)
        y_data.append(alignment_score)
    X_data = np.array(X_data)
    X_data = X_data.reshape(X_data.shape[0], X_data.shape[1], X_data.shape[2], 1)  # Reshape for CNN
    y_data = np.array(y_data)
    return X_data, y_data

def compute_dotplot_matrix(seq1, seq2):
    rows, cols = len(seq1), len(seq2)
    dot_plot = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if seq1[i] == seq2[j]:
                dot_plot[i][j] = 1
    return dot_plot

if __name__ == "__main__":
    num_pairs = 1000
    seq_length = 128

    # Generate similar pairs of DNA sequences with varying alignment scores
    pairs = create_similar_pairs(num_pairs, seq_length)

    # Create the dataset for training and testing the deep learning model
    X_data, y_data = create_dataset(pairs)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    # Build the deep learning model
    input_shape = (seq_length, seq_length, 1)  # Update this based on your dotplot matrix dimensions
    model = build_model(input_shape)

    # Variables to keep track of the best pair and best alignment score
    best_pair = None
    best_alignment_score = float('-inf')

    # Train the model and keep track of the best pair and best alignment score
    for epoch in range(10):  # Replace '10' with the desired number of epochs
        model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))

        # Evaluate the model on the test set
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        # Find the pair with the best alignment score so far
        best_pair_index = np.argmax(y_test)  # Get the index of the highest alignment score in y_test
        if y_test[best_pair_index] > best_alignment_score:
            best_alignment_score = y_test[best_pair_index]
            best_pair = pairs[best_pair_index]

        print(f"Epoch {epoch+1}, Mean Squared Error (MSE): {mse}")
    
    print()
    print("Best Pair with Highest Alignment Score:")
    print(f"Sequence 1: {best_pair[0]}")
    print(f"Sequence 2: {best_pair[1]}")
    print(f"Alignment Score: {best_alignment_score}")

    # Evaluate the model on the test set
    loss = model.evaluate(X_test, y_test)
    print(f"Test loss: {loss}")
    
    first_seq1, first_seq2, _ = pairs[0]
    dot_plot = compute_dotplot_matrix(first_seq1, first_seq2)
    plt.imshow(dot_plot, cmap='gray', aspect='auto', origin='upper')
    plt.xticks(range(len(first_seq2)), first_seq2)
    plt.yticks(range(len(first_seq1)), first_seq1)
    plt.xlabel("Sequence t")
    plt.ylabel("Sequence s")
    plt.title("Dot Plot: Similarity of Sequences s and t")
    plt.show()

    # Predict the best alignment score for the first pair using the trained model
    dot_plot_input = dot_plot.reshape(1, dot_plot.shape[0], dot_plot.shape[1], 1)
    predicted_alignment_score = model.predict(dot_plot_input)[0][0]
    
    print(f"Best Alignment Score (Predicted): {predicted_alignment_score}")

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse}")
