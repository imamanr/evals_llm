#!/usr/bin/env /opt/homebrew/bin/bash

export AWS_ACCESS_KEY_ID="ASIAVRUVQ64SZLWPVAOG"
export AWS_SECRET_ACCESS_KEY="WB6ZcJhgluDLEnyLMXdKabcJBOeL+LbRRsqJ4PyC"
export AWS_SESSION_TOKEN="IQoJb3JpZ2luX2VjELD//////////wEaCXVzLXdlc3QtMiJIMEYCIQDrV31dSV9vdsvVqfLZ0dwqGuILJZXhSSKYlhMxfCQKXQIhAInmqXKYAgVmLiwUt9XaKU/KCYO0/9M8mdF7LoxYivaLKocDCDkQABoMMzgxNDkxOTM1MDEzIgwow1Xe8lHziqIh5cIq5ALn4NkWGJIsNNPx+XF/qOX89gCSyTlesj+P5ZdhNJmwMy2aPzWSt+VL1gReIyb9jLH3Tx3p/Mr5PGurUGuGK5cReUUL5sGsPDMUrbffkCRLHWJgmZ9lnAiw8elEsccWrUTMhLFfaLA/p+rX55eoCaLjHMrcwjrvDkssuB3iLwRHEDNg34dqHnRccclnzpCIEFr4OZeIFesEZJpM2DCXQEQYJZiE6RVKeu4d5sUssjYnEIvRAo5LSyJkA3/kXIRe6TvbiitKi758nOmvDgMkKw3QsZgifj7XwWw2JPILzGO7583Bb724VqUOSgmVmT63x/tcFbWG3jZXgqLOhAxWiZ/KF/JSBeYERda2ai/kkgMWvovx9ZqVkHYydUqq+P6/yzTLnqrRE7glGmZtn/qmVHGrCoTSXrbMDM5ByJHoaU4/xla13fXr2tQiMbQkbU3rMdTJCAETk+5K4DBgqTbvx7A+rqmSvDCOnuSyBjqlAfzToG6ixN+OuBAuSK4tNEQlJXpq1GTfKwKbAGL65B9y13uqyhW2LG8mKsrnWzXz2Jg+miqInJydvXtDDuBSwYqCZLuATU4pFi/GjyUEO5FjXJOqSlhMA5GwapTAAW8SM2zB+Q5GGzjE2dh25A1cvqPMaRQv4ljVw8cFSZDxxEJgYAyTEM1Y1mHCY1QC737e0pLsNgMIcigeB2GvSIG9ZwpMOPLPAw=="


python evaluate.py -m haiku_text -d intent_classfication -ddir assets/data -mt1 F1_score -mt2 latency -vd bedrock

python evaluate.py -m haiku -d lvm_logs -ddir ~/Documents/Rabbit/Datasets/ -mt1 Clip -mt2 latency -vd bedrock

python evaluate.py -m hf_llm -d sample_conv -ddir ./assets/data_r1/ -mt1 sentenceT -mt2 latency -vd huggingface

python evaluate.py -m fireworks -d lvm_logs -ddir ~/Documents/Rabbit/Datasets/ -mt1 Clip -mt2 latency -vd fireworks