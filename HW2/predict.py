import argparse
import time

from sklearn.metrics import confusion_matrix, accuracy_score
from torch import nn

from utils import *


def eval(df, model, criterion):
    # Convert the sentiment_class from set to list
    sentiment_class = list(label_dict.keys())
    y_pred, y_actual = [], []

    test_tweets = df.content.tolist()
    test_labels = df.emotion.tolist()

    # predict
    with torch.no_grad():
        for i in range(len(test_tweets)):
            sentence, label = test_tweets[i], test_labels[i]
            inputs = prepare_sequence(sentence, model.vocab)
            hidden = model.init_hidden()
            for j in range(len(inputs)):
                class_scores, hidden = model(inputs[j], hidden)
            # for word i. The predicted tag is the maximum scoring tag.
            y_pred.append(sentiment_class[(class_scores.max(dim=1)[1].numpy())[0]])
            y_actual.append(str(label))

    print(sentiment_class)
    print(confusion_matrix(y_actual, y_pred, labels=sentiment_class))
    print(accuracy_score(y_actual, y_pred))


if __name__ == '__main__':
    t = time.time()
    # Parsing script arguments
    parser = argparse.ArgumentParser(description='Process input')
    parser.add_argument('input_file', type=str, help='Input folder path, containing images')
    args = parser.parse_args()

    print('Loading Data...')
    df = load_data(args.input_file)
    print('------------------- Stage 1 completed -------------------')

    print('Preprocessing...')
    df = preprocess(df)
    print('------------------- Stage 2 completed -------------------')

    # stage 3: load model
    print('Loading Model...')
    model = torch.load('model.pkl')
    print('------------------- Stage 3 completed -------------------')

    # stage 4: evaluating
    print('Evaluating...')
    criterion = nn.NLLLoss()
    eval(df, model, criterion)
    print('------------------- Stage 4 completed -------------------')

    # stage 4: saving csv
    print('Exporting to prediction.csv...')
    # prediction_df.to_csv("prediction.csv", index=True, header=False)
    print('------------------- Stage 4 completed -------------------')

    print('Finished!')
    print(f'Total Time Taken: {time.time() - t}')
