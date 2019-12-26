# adapted from https://github.com/udacity/sagemaker-deployment/blob/master/Tutorials/IMDB%20Sentiment%20Analysis%20-%20XGBoost%20-%20Web%20App.ipynb

import boto3
import librosa
import numpy as np
import json

def extract_mfcc(wav_file, coeffs = 40):
        """Extract MFCC (Mel-frequency cepstral coefficients) from a file.
        Args:
           wav_file: name of a .wav audio file 
        Returns:
            mfccs: mfccs
        """
        X, sample_rate = librosa.load(wav_file) 
        # extract mfcc feature from data
        mfcc = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=coeffs).T,axis=0) 
    
        return mfcc

def lambda_handler(event, context):
    
    class_mapping_dict = {'angry': 0, 'fear': 1, 'happy': 2, 'neutral': 3, 'sad': 4}
    
    mfcc = extract_mfcc(event['body'])
    input = {'instances': [mfcc]}    
    
    # The SageMaker runtime is what allows to invoke the created endpoint.
    runtime = boto3.Session().client('sagemaker-runtime')

    # use the SageMaker runtime to invoke the endpoint
    response = runtime.invoke_endpoint(EndpointName = 'tensorflow-inference-2019-12-26-08-24-13-228', # The name of the endpoint
                                       ContentType = 'application/json', # The data format that is expected
                                       Body = json.dumps(input) # The preprocessed audio file

    # The response is an HTTP response whose body contains the result of our inference
    result = response['Body'].read().decode('utf-8')
    # thansform to json                                   
    result = json.loads(result)                                   
    # extract an array of probability predictions
    result = result['predictions'][0]
    
    # get the index of a maximum value
    max_idx = np.argmax(result)
    # get key of a value equal to the index of a maximum value
    # sourced from https://stackoverflow.com/questions/8023306/get-key-by-value-in-dictionary
    result_label = list(class_mapping_dict.keys())[list(class_mapping_dict.values()).index(max_idx)]

    return {
        'statusCode' : 200,
        'headers' : { 'Content-Type' : 'text/plain', 'Access-Control-Allow-Origin' : '*' },
        'body' : str(result_label)
    }