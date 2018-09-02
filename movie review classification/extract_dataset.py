file = open("./train/labeledBow.feat", "r")
file1 = open("./train/train_dataset.txt", "w")

neg_count = 0
pos_count = 0

for line in file:
	tokens = line.split()
	if int(tokens[0]) >= 7 and pos_count < 500:
		pos_count = pos_count + 1
		print(line)
		file1.write(line)
	if int(tokens[0]) <= 4 and neg_count < 500:
		neg_count = neg_count + 1
		print(line)
		file1.write(line)