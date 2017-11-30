import numpy as np
import data_loader
import models
import argparse
import sys


def main():

    qa_data = data_loader.load_questions_answers(1, 'Data')
    qa_train = qa_data['training']
   
    fc7_features, image_id_list = data_loader.load_fc7_features('Data', 'train')
    image_id_map = {}
    for i in xrange(len(image_id_list)):
        image_id_map[image_id_list[i]] = i

    sentence_train = np.ndarray((len(qa_train), qa_data['max_question_length']), dtype='int32');
    answer_train = np.zeros((len(qa_train), len(qa_data['answer_vocab'])));
    fc7_train = np.ndarray((len(qa_train), 4096));
  
    for i in range(0, len(qa_train)):
        sentence_train[i, :] = qa_train[i]['question'][:]
        answer_train[i, qa_train[i]['answer']] = 1.0
        fc7_index = image_id_map[qa_train[i]['image_id']]
        fc7_train[i, :] = fc7_features[fc7_index][:]

    qa_val = qa_data['validation']

    
   

    sentence_val = np.ndarray((len(qa_val), qa_data['max_question_length']), dtype='int32');
    answer_val = np.zeros((len(qa_val), len(qa_data['answer_vocab'])));
    
     
    for i in range(0, len(qa_val)):
        sentence_val[i, :] = qa_val[i]['question'][:]
        answer_val[i, qa[i]['answer']] = 1.0
        fc7_train[i, :] = fc7_features[fc7_index][:]
        

    model = models.vis_lstm()
    X_train = [fc7_train, sentence_train]
    X_val = [fc7_train, sentence_val]
    model_path = 'weights/model_1.h5'
   

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, answer_train,
              nb_epoch=args.num_epochs,
              batch_size=args.batch_size,
              validation_data=(X_val, answer_val),
              verbose=1)

    model.save(model_path)


if __name__ == '__main__': main()
