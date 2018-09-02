import math
import pickle
import random as rn
import timeit
import tree_class as tc
import numpy as np

def read_dataset_and_attributes(attribute_filename, training_set_filename):

	attr_list = []
	reviews_list = {}

	if attribute_filename:
		with open(attribute_filename, "r") as file:
			for line in file:
				tokens = line.strip().split()
				attr_list.append(int(tokens[1]))

	count = 0
	with open(training_set_filename, "r") as file:
		for line in file:
			tokens = line.strip().split()
			reviews_list[count] = (int(tokens[0]), line)
			count = count + 1

	return attr_list, reviews_list


def check(var, test_data):

	if var:
		if var.label:
			return var.label

		else:
			if str(var.attr) + ":" in test_data:
				label = check(var.left, test_data)
			else:
				label = check(var.right, test_data)

			return label

	return None


def CheckModel(var, filepath):

	_, test_data = read_dataset_and_attributes(None, filepath)

	false_count = 0

	for data in test_data.keys():
		actual_label = None
		if test_data[data][0] >= 7:
			actual_label = 1
		else:
			actual_label = -1

		if actual_label != check(var, test_data[data][1]):
			false_count = false_count + 1

	print(filepath, " error: ", false_count/len(test_data))
	return false_count/len(test_data)


def checkError(var, test_data):
	false_count = 0
	for data in test_data.keys():
		actual_label = None
		if test_data[data][0] >= 7:
			actual_label = 1
		else:
			actual_label = -1

		if actual_label != check(var, test_data[data][1]):
			false_count = false_count + 1

	# print("validation error: ", false_count/len(test_data))
	return false_count/len(test_data)


def DecisionTree(attr_list, reviews_list):
	rn.shuffle(attr_list)
	rn.shuffle(reviews_list)
	level_tree = {}
	print("start making tree")
	m = open('./model.pickle', 'wb')
	var = tc.decision_tree(list(reviews_list.keys()), None, None)
	start = timeit.default_timer()
	var.MakeTree(reviews_list, attr_list)
	print(timeit.default_timer() - start)
	pickle.dump(var, m)
	m.close()
	if var:
		depth = 0
		var.printTree(depth, level_tree)

		for level in sorted(level_tree.keys()):
			print(level," : ", end=" ")
			for value in level_tree[level]:
				print(value.attr, "(", value.label, ")", end="\t")

			print("\n")

	CheckModel(var, "./train/train_dataset.txt")
	CheckModel(var, "./test/test_dataset.txt")
	

def EarlyStoppingDT(attr_list, reviews_list, depth, percentage_review):

	level_tree = {}
	print("start making tree")
	m = open('./model_earlystopping.pickle', 'wb')
	var = tc.decision_tree(list(reviews_list.keys()), None, None)
	var.MakeTree(reviews_list, attr_list, depth=depth, percentage_review=percentage_review)
	pickle.dump(var, m)
	m.close()
	if var:
		depth = 0
		var.printTree(depth, level_tree)

		for level in sorted(level_tree.keys()):
			print(level," : ", end=" ")
			for value in level_tree[level]:
				print(value.attr, end="\t")

			print("\n")

	CheckModel(var, "./train/train_dataset.txt")
	CheckModel(var, "./test/test_dataset.txt")


def noise_add(noise_percentage, attr_list, reviews_list):

	total_noise = (noise_percentage/100) * len(reviews_list)

	while total_noise > 0:

		r = rn.randint(0, len(reviews_list) - 1)
		if reviews_list[r][0] >= 7:
			reviews_list[r] = (-1,reviews_list[r][1])
		else:
			reviews_list[r] = (1,reviews_list[r][1])
		total_noise = total_noise - 1


	level_tree = {}
	print("start making tree")
	m = open('./noise_model.pickle', 'wb')
	var = tc.decision_tree(list(reviews_list.keys()), None, None)
	var.MakeTree(reviews_list, attr_list)
	pickle.dump(var, m)
	m.close()
	if var:
		depth = 0
		var.printTree(depth, level_tree)

		for level in sorted(level_tree.keys()):
			print(level," : ", end=" ")
			for value in level_tree[level]:
				print(value.attr, "\t", value.label, end="\t")

			print("\n")

	CheckModel(var, "./train/train_dataset.txt")
	CheckModel(var, "./test/test_dataset.txt")


def pruning(reviews_list):
	_, test_data = read_dataset_and_attributes(None, "./test/test_dataset.txt")
	model = open("model.pickle","rb")
	var = pickle.load(model)
	error = 1
	level = {}
	var.printTree(0, level)
	print(var.parent)
	for l in sorted(level.keys()):
		print(l," : ", end=" ")
		for value in level[l]:
			print(value.attr, "(", value.label, ")", end="\t")

		print("\n")

	for i in reversed(sorted(level.keys())):
		for node in level[i]:
			if node:
				print(node.error)
				# if not node.error:
				left = node.left
				right = node.right
				label = node.label
				node.left = None
				node.right = None
				p, n = node.return_pos_neg_count(reviews_list)
				if p > n:
					node.label = 1
				else:
					node.label = -1
				print(var)
				e = checkError(var, test_data)
				print("error: ", e, " node: ", node.attr)
				if e > 0.3:
					node.left = left
					node.right = right
					node.label = label
					node.errorless_parent()	

	level = {}
	var.printTree(0, level)
	for l in sorted(level.keys()):
		print(l," : ", end=" ")
		for value in level[l]:
			print(value.attr, "(", value.label, ")", end="\t")

		print("\n")

	CheckModel(var, "./train/train_dataset.txt")
	CheckModel(var, "./test/test_dataset.txt")


def predict(tree, test_data):

	if tree:
		if tree.label:
			return tree.label

		else:
			if str(tree.attr) + ":" in test_data:
				label = check(tree.left, test_data)
			else:
				label = check(tree.right, test_data)

			return label

	return None


def feature_bagging(attr_list, reviews_list, number_of_trees):

	_, test_data = read_dataset_and_attributes(None, "./test/test_dataset.txt")
	forest = []
	for i in range(number_of_trees):
		print(i)
		rn.shuffle(attr_list)
		var = tc.decision_tree(list(reviews_list.keys()), None, None)
		var.MakeTree(reviews_list, attr_list[0:1000])
		forest.append(var)


	error = 0
	for data in test_data.keys():
		actual_label = None
		prediction = None
		if test_data[data][0] >= 7:
			actual_label = 1
		else:
			actual_label = -1

		pred = 0

		for tree in forest:
			p = predict(tree, test_data[data][1])
			if p == 1:
				pred = pred + 1

		if len(forest) - pred > pred:
			prediction = -1
		else:
			prediction = 1

		if actual_label != prediction:
			error = error + 1

	print("error: ", error/len(test_data))


attr_list, reviews_list = read_dataset_and_attributes('attr_words.txt', './train/train_dataset.txt')
for attr in attr_list:
	flag = False
	for review in reviews_list:
		if str(attr) + ":" in reviews_list[review][1]:
			flag = True
			break

	if not flag:
		attr_list.remove(attr)

print(len(attr_list))
DecisionTree(attr_list, reviews_list)
# pruning(reviews_list)
# EarlyStoppingDT(attr_list, reviews_list,  50, 2)
# noise_add(.5, attr_list, reviews_list)
# noise_add(1, attr_list, reviews_list)
# noise_add(5, attr_list, reviews_list)
# noise_add(10, attr_list, reviews_list)
# feature_bagging(attr_list, reviews_list, 20)