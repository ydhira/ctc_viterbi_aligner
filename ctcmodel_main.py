import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from ctcdecode import CTCBeamDecoder
from phoneme_list import *
import stringdist
import csv
from deepspeechpytorch.decoder import GreedyDecoder
from phoneme_list import *
import argparse
from scipy.io import wavfile
from python_speech_features import logfbank, fbank 

# mel log , window shift = 100 ms, window size = 25 ms.

def get_feature(file_name):
	rate, data = wavfile.read(file_name)
	output, _ = fbank(data,samplerate=rate, winlen=0.025625,
					  winstep=0.01, nfilt=40, nfft=512,
					  lowfreq=100, highfreq=3800, winfunc=np.hamming)
	# print("mellog output shape: ", output.shape)
	output = np.log(output)
	return output

class ER:
	def __init__(self):
		self.label_map = PHONEME_MAP + [' ']
		self.phoneme_list = PHONEME_LIST + [' ']
		self.decoder = CTCBeamDecoder(
			labels=self.label_map, 
			blank_id=phonemes_len,
			log_probs_input=True,
			beam_width=200
		)
		self.greedy_decoder = GreedyDecoder(
							labels=self.label_map,
							blank_index=phonemes_len)

	def __call__(self, prediction, target=None, test=False):
		return self.forward(prediction, target)

	def forward(self, prediction, target):
		logits = prediction[0] # (logits, len)
		feature_lengths = prediction[1].int()
		labels = target
		logits = torch.transpose(logits, 0, 1)
		logits = logits.cpu()
		# beam decoder
		output, scores, timesteps, out_seq_len = self.decoder.decode(probs=logits, seq_lens=feature_lengths)

		############# GREEDY DECODE ##########################
		_, max_probs = torch.max(logits, 2)
		strings, offsets = self.greedy_decoder.decode(probs = logits)
		predictions = []
		time_stamps = []
		ls = 0
		for i in range(len(strings)):
			pred = strings[i][0]
			phone_pred = []
			for j in pred:
				phone_pred.append(self.phoneme_list[self.label_map.index(j)])
			predictions.append(phone_pred)
			time_stamps.append(offsets[i][0].float()/100)
			if target != None:
				true = "".join(self.label_map[l] for l in labels[i])			
				ls += stringdist.levenshtein(strings[i][0], true)
		return predictions, time_stamps, ls/len(strings)

def val_model(model, feature_input, test=False):
	with torch.no_grad():
		model.eval()
		error_rate = ER()		
		log_softmax_logits, softmax_logits, lens = model([feature_input])
		pred = error_rate((logits,lens))
		return pred

class CTCPhonemeSegmentor(nn.Module):
	def __init__(self, vocab_size, hidden_size, nlayers): # 47, 128, 1
		super(CTCPhonemeSegmentor,self).__init__()
		self.vocab_size=vocab_size
		self.hidden_size = hidden_size
		self.nlayers=nlayers
		self.embed = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=3, stride=(1, 1), padding=(1, 1), bias=False),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Conv2d(16, 32, kernel_size=3, stride=(1, 1), padding=(1, 1), bias=False),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=(1, 1), bias=False),
			nn.ReLU(),
			nn.Dropout(0.3)
			)
		self.rnn = nn.LSTM(input_size = 1280, hidden_size=hidden_size, num_layers=nlayers, bidirectional=True, dropout=0.1) # 1 layer, batch_size = False
		self.fc1 = nn.Linear(2*hidden_size,1024)
		self.do = nn.Dropout(p=0.2)
		self.relu = nn.ReLU()
		self.scoring = nn.Linear(1024,vocab_size)

	def forward(self, seq_list): # (max_len , bs , 40)
		lens = [f.shape[0] for f in seq_list]
		 # features = [torch.from_numpy(f).cuda() for f in seq_list]
		features = [torch.from_numpy(f) for f in seq_list]
		input_features = rnn.pad_sequence(features).float()
		input_features = input_features.unsqueeze(1).permute(2,1,3,0) # (bs, 1, 40, max_len) bsx1xmax_lenx40
		# print("---->", input_features.size())
		input_features = self.embed(input_features)
		# print("after cnn", input_features.size()) #(bs, out_ch, 20, max_len)
		n, c, h, w = input_features.size()
		input_features = input_features.view(n, c*h, w).permute(2, 0, 1)
		# print("after permute",input_features.size()) #(max_seq, bs, out_ch * 20)
		packed_input = rnn.pack_padded_sequence(input_features, lens) # packed version
		output_packed, hidden = self.rnn(packed_input)
		# print("output_padded", output_padded.shape)
		out, lengths = rnn.pad_packed_sequence(output_packed)
		out = self.fc1(out)
		out = self.relu(out)
		out = self.do(out)
		logits = self.scoring(out) #concatenated logits
		out = nn.functional.log_softmax(logits, dim=2)
		out2 = nn.functional.softmax(logits, dim=2)

		# print("out: ",out.shape)
		del input_features
		for f in features:
			del f
		del packed_input
		del output_packed
		return out, out2, lengths #return concatenated logits ##return lens also

if __name__ == '__main__':
	parser = argparse.ArgumentParser( prog='ctcmodel_main.py', description='Takes in the input wav file and outputs \
																	(1) a list of phonemes and (2) time stamps for each phoneme')
	parser.add_argument('-infile', required=True, help='The input file to process')
	args = parser.parse_args()
	feature_input = get_feature(args.infile)

	phonemes_len = len(PHONEME_MAP)
	model = CTCPhonemeSegmentor(47,512,3)
	model = model.cuda()
	checkpoint = torch.load('18.model')
	model = model.load_state_dict(checkpoint['model_state_dict'])
	output = val_model(model, feature_input)
	# log_softmax_logits, softmax_logits, lens = model([feature_input])
	print("Predicted phonemes: ")
	print(output[0][0])
	print("At timestamps: (in seconds) ")
	print(output[1][0])



