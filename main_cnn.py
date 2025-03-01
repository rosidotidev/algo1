import ai.cnn_model as cm
import pandas as pd
import data.data_enricher as de
import ai.ml_trainers as mt

def main():
    try:
        ticker="MSFT"
        data = pd.read_csv(f"../data/{ticker}.csv", index_col="Date", parse_dates=True)
        data = de.add_technical_indicators_v1(data)
        data = mt.forward_looking_label(data, look_ahead=15, threshold_buy=0.10, threshold_sell=0.05)
        sh_data = data['Target'].value_counts()
        print(sh_data)
    except FileNotFoundError:
        print("File 'ticker_data.csv' not found.")
        return

    # Estrai un campione di dati prima di dividere i dati
    sample_size = 10  # Numero di righe per ogni classe
    sample_0 = data[data['Target'] == 0].sample(sample_size)
    sample_1 = data[data['Target'] == 1].sample(sample_size)
    sample_2 = data[data['Target'] == 2].sample(sample_size)
    sample_data = pd.concat([sample_0, sample_1, sample_2])

    # Rimuovi le righe del campione dal DataFrame originale
    data = data.drop(sample_data.index)

    # Calcola la forma dell'input
    input_shape = (data.shape[1] - 1, 1)

    # Costruisci il modello CNN
    cnn_model = cm.build_cnn(input_shape)

    # Addestra il modello CNN
    cm.train_cnn(cnn_model, data)

    # Prepara i dati di test dal campione
    X_sample = sample_data.drop('Target', axis=1).values
    y_sample = sample_data['Target'].values
    X_sample = X_sample.reshape(X_sample.shape[0], X_sample.shape[1], 1)

    # Testa il modello sul campione
    loss, accuracy = cnn_model.evaluate(X_sample, y_sample, verbose=0)
    print(f"Sample Test - Loss: {loss}")
    print(f"Sample Test - Accuracy: {accuracy}")

if __name__ == "__main__":
    main()