import argparse
import io, os, sys

# Imports the Google Cloud client library
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

from trans_to_phon import *
from ctcmodel_main import *
from viterbi_align import *
from phoneme_list import *

client = speech.SpeechClient()

## requirements
#(1) pip install google-cloud-speech
#(2) git clone --recursive https://github.com/parlance/ctcdecode.git
# cd ctcdecode
# pip install .
#(3) Setting up gcloud:
# https://cloud.google.com/speech-to-text/docs/quickstart-gcloud

def get_gloud_transcript(file_name):
	with io.open(file_name, 'rb') as audio_file:
		content = audio_file.read()
		audio = types.RecognitionAudio(content=content)

	config = types.RecognitionConfig(
		encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
		language_code='en-US')

	# Detects speech in the audio file
	response = client.recognize(config, audio)
	result = response.results[0]
	transcript = result.alternatives[0].transcript
	return transcript

def run_main(file):
	label_map = PHONEME_MAP + [' ']
	phoneme_list = PHONEME_LIST + [' ']

	transcript = get_gloud_transcript(file)
	# dictionary = word_phon_dict("/home/hyd/fisher_complete_trans/f1+f2.complete.fordecode.dict")	
	dictionary = word_phon_dict("cmudict.0.6d.wsj0.forfalignment")
	phonemes = convert_sentence_trans_phones(transcript, dictionary)
	phonemes_id = [PHONEME_LIST.index(l) for l in phonemes]
	# print(phonemes_id)
	feature_input = get_feature(file)
	# print(feature_input.shape)
	if (feature_input.shape[0]< 50 or len(phonemes_id) < 2): # Too small audio or couldnt break into phonemes 
		return 
	phonemes_len = len(PHONEME_MAP)
	model = CTCPhonemeSegmentor(47,512,3)
	# model = model#.cuda()
	checkpoint = torch.load('18.model')
	model.load_state_dict(checkpoint['model_state_dict'])
	model.eval()
	log_softmax_logits, softmax_logits, lens = model([feature_input])
	logits = log_softmax_logits.detach().cpu().squeeze().transpose(1,0)
	# print(logits)
	output_sequence = viterbi_align(logits, phonemes_id)
	blank_id = logits.shape[0]-1
	time_stamps_useful = list(map(lambda i: i if output_sequence[i]!=blank_id else -1, np.arange(len(output_sequence))))
	time_stamps_not_blank = np.array(list(filter(lambda i: i!=-1, time_stamps_useful)))
	time_stamps_not_blank = time_stamps_not_blank/100.
	output_sequence_not_blank = np.array(list(filter(lambda i: i!=blank_id, output_sequence)))
	phone_pred = []
	for j in output_sequence_not_blank:
		phone_pred.append(phoneme_list[j])
	return phone_pred, time_stamps_not_blank


if __name__ == '__main__':
	parser = argparse.ArgumentParser( prog='gcloud_main.py', description='Takes in the input wav file and outputs \
																	(1) a list of phonemes and (2) time stamps for each phoneme')
	parser.add_argument('-infile', required=True, help='The input file to process')
	args = parser.parse_args()
	output_sequence, time_stamps= run_main(args.infile)
	print("Predicted phonemes: ")
	print(output_sequence)
	print("At timestamps: (in seconds) ")
	print(time_stamps)
	print(len(output_sequence), len(time_stamps))
	## time_stamps[0] is when the first phoneme ends. Duration (0 -> time_stamps[0])
	## time_stamps[1] is when the second phoneme ends. Duration (time_stamps[0] -> time_stamps[1])
	## and so on ...