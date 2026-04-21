# Required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
import os

# Show current working directory
# print(os.getcwd())


# Train : entrainement : the rest of the data
# Val : validation 0.1 => 10 % of the Train+val data
# Test : test après validation 0.2 => 20 % of full data

def train_lstm_model(X, Y, input_shape, test_size=0.2, val_size=0.1, epochs=20, batch_size=16, threshold=0.5):
    """
    Cette fonction permet de
        1. Séparer les données en ensembles d'entraînement et de test.
        2. Créer un modèle LSTM avec 64 unités, qui prédit une sortie à chaque pas de temps (séquence complète).
        3. Compiler le modèle avec la fonction de perte `binary_crossentropy` et l'optimiseur `adam`.
        4. Entraîner le modèle sur les données d'entraînement, en validant sur les données de test.
        5. Faire des prédictions sur les données de test.
        6. Appliquer un seuil (par défaut 0.5) pour transformer les prédictions en classes binaires (0 ou 1).
        7. Retourner les données test, les vraies valeurs, les prédictions continues et binaires, l'historique d'entraînement ainsi que le model lui même.

    How to use it:
        import sys
        sys.path.append("../../../00_Swiss_knife")
        from train_lstm_model import train_lstm_model

        Data should be cleaned before the training phase :
        # Normalisation
        scaler = MinMaxScaler() 
        df_temp_norm = pd.DataFrame(scaler.fit_transform(df_temp), columns=df_temp.columns, index=df_temp.index)
        df_Dtemp_norm = pd.DataFrame(scaler.fit_transform(df_Dtemp), columns=df_Dtemp.columns, index=df_Dtemp.index)
        df_co2_norm = pd.DataFrame(scaler.fit_transform(df_co2), columns=df_co2.columns, index=df_co2.index)

        # Combine in one dataframe
        df_combined = pd.concat([df_temp_norm,df_Dtemp_norm,df_co2_norm, df_Occ], axis=1, join='inner') # inner join : Only matching rows (intersection)

        # Convert each of them into array (numpy)
        temp = df_combined.iloc[:, 0:144].to_numpy()  # Tableau purement numérique
        Dtemp = df_combined.iloc[:, 144:288].to_numpy()
        co2 = df_combined.iloc[:, 288:432].to_numpy()

        # Create tensor for the features (+2 features)
        X = np.stack((temp, Dtemp, co2), axis=2)

        # RMQ : if you want to use only one feature...(no stack)
        # X = df_combined.iloc[:, :144].to_numpy().reshape((-1, 144, 1))
        # (-1 = deduction auto du nb de lignes, 144 = nb de colonnes, 1 = nombre de features)

        # Groundtruth dataset
        Y = df_combined.iloc[:, -144:].to_numpy()
 
    X_train, X_val, X_test, Y_train, Y_val, Y_test, Y_pred, Y_pred_binary, history, metrics, model = train_lstm_model(
        X,
        Y,
        input_shape=(144, 3),
        epochs=100,
        batch_size=16,
        threshold=0.5
        )
    
    
    Parameters
    ----------
     - X : features tensor
     - Y : Groundtruth
     - input_shape :
         (144,3)
             144 : nb of 10min in a day
             3 : nb of features
     - test_size=0.2 : amount of test data (20%)
     - val_size=0.1 : amount of validation data
     - epochs=20 : Hyperparameter...for tunning
     - batch_size=16 : Hyperparameter...for tuning
     - threshold=0.5 : actually the model does not directly predict binary values...

    returns
     -------
    - X_train
     - X_val
     - X_test
     - Y_train
     - Y_val
     - Y_test
     - Y_pred
     - Y_pred_binary
     - history
     - metrics
     - model
    
    """

    # Étape 1 : Split en train+val et test
    X_temp, X_test, Y_temp, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=42, shuffle=True
    )

    # Étape 2 : Split train+val en train et val
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_temp, Y_temp, test_size=val_size, random_state=42, shuffle=True
    )

    # Définition du modèle
    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=True, input_shape=input_shape),
        TimeDistributed(Dense(1, activation='sigmoid'))
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Entraînement avec validation
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, Y_val),
        verbose=1
    )

    # DERNIERE PARTIE DE LA FONCTION QUE JE NE COMPRENDS PAS TRès BIEN
    # Prédiction sur le test set
    Y_pred = model.predict(X_test) # ça c'est l'utilisation du modele proprement dit : sortie = model.predict(entrée)
    Y_pred_binary = (Y_pred > threshold).astype(int) # Cette ligne permet d'obtenir des valeurs binaire (1 ou 0 selon le threshold)
    # dans la prédiction proprement dit ce n'est pas directement des binaires que le modèle donne)

    from sklearn.metrics import classification_report
    Y_test_flat = Y_test.reshape(-1)
    Y_pred_binary_flat = Y_pred_binary.reshape(-1)

    # Y_test_flat: le vecteur 1D des vraies étiquettes de test (0 ou 1 pour une classification binaire).
    # Y_pred_binary_flat: le vecteur 1D des prédictions binaires de ton modèle (0 ou 1).

    print(classification_report(Y_test_flat, Y_pred_binary_flat))
    # résume plusieurs métriques importantes pour la classification
        # Precision : proportion de prédictions positives correctes par rapport à toutes les prédictions positives.
        # Recall (ou Sensitivity) : proportion de vrais positifs correctement identifiés.
        # F1-score : moyenne harmonique de la précision et du rappel.
        # Support : nombre d’échantillons réels de chaque classe.

    test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=0) # ça c'est pour sortir l'accuracy du modèle

    # test_loss measures how confident your predictions are, while test_accuracy measures how often your model got the answer exactly right.
    # During training: the optimizer minimizes loss, so low loss is the goal.
    # During evaluation: accuracy (or other metrics like F1-score, precision, recall) tells how well the model actually predicts the correct labels.
    # Accuracy just cares if the predicted class is correct (0 or 1), ignoring probabilities.
    # Loss (like binary crossentropy) cares how confident the model is in its predictions.
   
    print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")


    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    # Compute metrics
    precision = precision_score(Y_test_flat, Y_pred_binary_flat)
    recall = recall_score(Y_test_flat, Y_pred_binary_flat)
    f1 = f1_score(Y_test_flat, Y_pred_binary_flat)
    accuracy = accuracy_score(Y_test_flat, Y_pred_binary_flat)
    
    metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "loss": test_loss
    }
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test, Y_pred, Y_pred_binary, history, metrics, model

# Hyperparameters adjustement :
    # LSTM hyperparameters are basically all the “knobs” you can tune before training:
    # Architecture: layers, units, dropout, bidirectional
    # Training: epochs, batch size, learning rate, optimizer, loss function
    # Input: sequence length, feature dimension
    # Tuning these properly is critical for good performance.

#-------------------------------------------- NOT SURE OF ANYTHING BELOW----------------------------------------------------------------

# -------------------------
# HELPER FUNCTION: CREATE SLIDING WINDOWS (MANY-TO-ONE)
# -------------------------
def create_sliding_window_many_to_one(X, Y, window_size=24, step=1):
    """
    Create overlapping sequences for LSTM using a sliding window,
    where each sequence predicts **only the next timestep** (many-to-one).

    Parameters:
    - X: features, shape (num_samples (tensor raw = days), timesteps (tensor col), num_features)
    - Y: targets, shape (num_samples, timesteps) or (num_samples, timesteps, 1)
    - window_size: number of timesteps per input sequence
    - step: step size for sliding window

    Returns:
    - X_windows: array of shape (num_windows, window_size, num_features)
    - Y_windows: array of shape (num_windows, 1)  (next timestep)
    """
    X_windows = []
    Y_windows = []

    for sample_idx in range(X.shape[0]):
        for i in range(0, X.shape[1] - window_size):
            X_windows.append(X[sample_idx, i:i+window_size, :])  # input sequence
            Y_windows.append(Y[sample_idx, i+window_size])        # next timestep target

    X_windows = np.array(X_windows)
    Y_windows = np.array(Y_windows)

    # Ensure Y has shape (num_windows, 1) for compatibility with Dense output
    if len(Y_windows.shape) == 1:
        Y_windows = np.expand_dims(Y_windows, axis=1)

    return X_windows, Y_windows


# -------------------------
# LSTM TRAINING FUNCTION (MANY-TO-ONE)
# -------------------------
def train_lstm_model_many_to_one(X, Y, input_shape, test_size=0.2, val_size=0.1, epochs=20, batch_size=16, threshold=0.5):
    """
    Train a many-to-one LSTM model (input sequence predicts next timestep).

    Steps:
    1. Split data into train/val/test sets
    2. Build LSTM model with Dense output
    3. Compile and train
    4. Predict next timestep and calculate metrics
    """
    # -------------------------
    # Step 1: Split data
    # -------------------------
    X_temp, X_test, Y_temp, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=42, shuffle=True
    )

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_temp, Y_temp, test_size=val_size, random_state=42, shuffle=True
    )

    # -------------------------
    # Step 2: Build LSTM model
    # -------------------------
    model = Sequential([
        LSTM(64, activation='tanh', input_shape=input_shape),
        Dense(1, activation='sigmoid')  # output for next timestep
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # -------------------------
    # Step 3: Train model
    # -------------------------
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, Y_val),
        verbose=1
    )

    # -------------------------
    # Step 4: Predict on test set
    # -------------------------
    Y_pred = model.predict(X_test)
    Y_pred_binary = (Y_pred > threshold).astype(int)

    # Flatten for metrics
    Y_test_flat = Y_test.flatten()
    Y_pred_binary_flat = Y_pred_binary.flatten()

    # Classification report
    from sklearn.metrics import classification_report
    print(classification_report(Y_test_flat, Y_pred_binary_flat))

    # Accuracy and loss
    test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")

    # Compute additional metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    precision = precision_score(Y_test_flat, Y_pred_binary_flat)
    recall = recall_score(Y_test_flat, Y_pred_binary_flat)
    f1 = f1_score(Y_test_flat, Y_pred_binary_flat)
    accuracy = accuracy_score(Y_test_flat, Y_pred_binary_flat)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "loss": test_loss
    }

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, Y_pred, Y_pred_binary, history, metrics, model

# X_train, X_val, X_test, Y_train, Y_val, Y_test, Y_pred, Y_pred_binary, history, metrics, model = train_lstm_model_many_to_one(
# X_sw,
# Y_sw,
# input_shape=(window_size, 1),  # input length = window_size, 1 feature
# epochs=50,
# batch_size=16,
# threshold=0.5
# )



# Example of application

# # -------------------------
# # MAIN SCRIPT
# # -------------------------

# # Normalize your data
# scaler = MinMaxScaler()
# df_co2_norm = pd.DataFrame(scaler.fit_transform(df_co2), columns=df_co2.columns, index=df_co2.index)

# df_combined = pd.concat([df_co2_norm, df_occ], axis=1, join='inner')
# # df_combined.info()

# X = df_combined.iloc[:,:144].to_numpy().reshape((-1, 144, 1))
# Y = df_combined.iloc[:,-144:].to_numpy()

# # -------------------------
# # APPLY SLIDING WINDOW (MANY-TO-ONE)
# # -------------------------
# window_size = 24  # input sequence length (e.g., 4 hours)
# step = 1          # slide by 1 timesteps (e.g., 10 min )
# X_sw, Y_sw = create_sliding_window_many_to_one(X, Y, window_size, step)

# print("New X shape:", X_sw.shape)  # (num_samples, 24, 1)
# print("New Y shape:", Y_sw.shape)  # (num_samples, 1)
