import os

def get_trans_files(dirname):
	trans_files = []
	for root, dirs, files in os.walk(dirname):
		for name in files:
			if name.endswith(".txt"):
				trans_files.append(os.path.join(root, name))

	return trans_files

def word_phon_dict(file):
	'''
	returns a dictionary with word as key and value as list of phonemes. 
	'''
	fd = open(file, 'r').readlines()
	d = {}
	for line in fd:
		line_split = line.split(" ")
		phones = line_split[1:]
		word = line_split[0].lower()
		d[word] = (" ").join(phones).rsplit()
	return d

def convert_sentence_trans_phones(sentence, d):
	'''takes in transcript and dictionary of word to list of phonemes.
	returns the phone sequence in the sentence '''
	d_keys = d.keys()
	phones=[]
	sentence = sentence.replace("\n", "")
	sentence = sentence.lower()
	sentence_split = sentence.split(" ")
	for word in sentence_split:

		if word in d_keys:	
			# print(word, d[word])
			phones.extend(d[word])
	return phones

def convert_trans_phones(files, d):
	'''
	files a list of files where each file contains transcript. d is the dictionary of word to phonemes. 
	returns a dictionary where key is filename and value is the list containing sequence of phonemes in the file. 
	'''
	d_keys = d.keys()
	file_phones = {}
	for file in files:
		phones = []
		lines = open(file, "r").readlines()
		sentence = ""
		for line in lines:
			sentence += line.replace("\n", "")
		sentence = sentence.lower()
		sentence_split = sentence.split(" ")
		for word in sentence_split:
			if word in d_keys:	
				# print(word, d[word])
				phones.extend(d[word])
			else:
				print("%s not in dictionary " % word)
		file_phones[file] = phones
		print(file)
		# print(phones)
		break
	return file_phones

def write_phones(file_phones, in_dir, out_dir):
	files = file_phones.keys()
	for f in files:
		out_file = f.replace(in_dir, out_dir)
		dirname = os.path.dirname(out_file)
		if not os.path.exists(dirname):
			os.makedirs(dirname) 
		phones = (" ").join(file_phones[f])
		with open(out_file, 'w') as fw:
			fw.write(phones)
		# print("writing to : ", out_file)
		# break


if __name__ == "__main__":
	in_dir = "/home/hyd/DL-Spring19/Hw3/hw3p2/code/transcripts/data"
	out_dir = "/home/hyd/DL-Spring19/Hw3/hw3p2/code/phonemes/"

	trans_files = get_trans_files(in_dir)
	print("len trans_files: ", len(trans_files))
	d = word_phon_dict("/home/hyd/fisher_complete_trans/f1+f2.complete.fordecode.dict")	
	file_phones = convert_trans_phones(trans_files, d)
	# write_phones(file_phones, in_dir,  out_dir)

