def extract_attr(train_set):
	file1 = open("imdb.vocab", "r")
	file2 = open('imdbEr.txt', 'r')
	file = open('attr_words.txt', "w")

	word = {}

	count = 0
	for line in file2:
		line1 = file1.readline().strip()
		word[line.strip()] = count
		count =count + 1

	keylist = list(word.keys())


	# print(keyp)
	# print(keyn)
	p = []
	n = []

	count = 0
	for key in keylist:
		if count == 5000:
			break
		if float(key) >= 2.068 and float(key) <= 5:
			file.write(str(word[key])+"\n")
			p.append(key)
			count = count + 1
		if float(key) <= -1.41 and float(key) >= -5:
			file.write(str(word[key])+"\n")
			n.append(key)
			count = count + 1

	print(len(p))
	print(len(n))
	print(count)


def train_set_read(filepath):
	train_set = []
	with open(filepath,'r') as file:
		for line in file:
			train_set.append(line.strip())


	return train_set


train_set = train_set_read("./train/train_dataset.txt")
extract_attr(train_set)