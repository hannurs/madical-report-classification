#!/bin/sh
echo "build voc"
C:/Users/Hania/AppData/Local/Programs/Python/Python39/python.exe build_vocabulary.py
echo "extract features"
C:/Users/Hania/AppData/Local/Programs/Python/Python39/python.exe data_extraction.py
echo "train & evaluate"
C:/Users/Hania/AppData/Local/Programs/Python/Python39/python.exe naive_bayes_train_evaluate.py
# python validate.py