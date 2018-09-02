'''
	Shailendra Kumar Gupta 2016csb1059
'''
'''Libraries required for different procesing'''
import math
import pickle
import random as rn
import timeit
import ct as tc
import numpy as np
import copy
import sys




'''
	assuming that attribute file contains word index line by line.
	training set as given.
'''
def read_dataset_and_attributes(attribute_filename, training_set_filename):

	print("Reading Files")
	attr_list = np.array([],dtype=int)
	reviews_list = {}

	count = 0
	with open(training_set_filename, "r") as file:
		for line in file:
			tokens = line.strip().split()
			reviews_list[count] = (int(tokens[0]), line)
			count = count + 1

	if attribute_filename:
		with open(attribute_filename, "r") as file:
			for line in file:
				tokens = line.strip()
				for review in reviews_list:
					if tokens+":" in reviews_list[review][1]:
						attr_list = np.insert(attr_list, 0 ,int(tokens))
						break

	
	print("Done.")
	print(len(attr_list))
	return attr_list, reviews_list

''' 
	prints the tree given the root node.
'''
def PrintTree(root):
	print("Level Vise Printing of Tree: ")
	if root:
		depth = 0
		level_tree = {}
		root.printTree(depth, level_tree)

		for level in sorted(level_tree.keys()):
			print(level," : ", end=" ")
			for value in level_tree[level]:
				print(value.attr, "(", value.label, ")", end="\t")

			print("\n")

	else:
		print("Tree is empty")


'''
	return the matrix [len(attr), len(revieslist)] to store presence of an attr in a review.
'''
def return_mat(attr_list, reviews_list):
	rn.shuffle(attr_list)

	mat = np.empty((len(attr_list), len(reviews_list)), dtype=int)
	# print(mat[0][])

	for i in range(len(attr_list)):
		count = 0
		for review in reviews_list:
			if str(attr_list[i])+":" in reviews_list[review][1]:
				mat[i][count] = 1
			else:
				mat[i][count] = -1
			count = count + 1

	return mat	


'''
	check the error of the model
'''
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


'''
	to make the decision tree.
'''
def DecisionTree(attr_list, reviews_list):
	level_tree = {}
	print("start making tree")
	m = open('./model.pickle', 'wb')
	mat = return_mat(attr_list, reviews_list)
	var = tc.decision_tree(list(reviews_list.keys()), None, None)
	start = timeit.default_timer()
	prev = []
	var.MakeTree(mat, attr_list, prev)
	print(timeit.default_timer() - start)
	pickle.dump(var, m)
	m.close()
	PrintTree(var)

	CheckModel(var, "./train/train_dataset.txt")
	CheckModel(var, "./test/test_dataset.txt")
	
'''
	to make early stopped decision tree.
'''
def EarlyStoppingDT(mat, attr_list, reviews_list, depth, percentage_review, ratio):

	level_tree = {}
	print("start making tree")
	m = open('./model_earlystopping.pickle', 'wb')
	var = tc.decision_tree(list(reviews_list.keys()), None, None)
	prev = []
	var.MakeTree(mat, attr_list, prev, reviews_list=reviews_list, depth=depth, percentage_review=percentage_review, ratio=ratio)
	pickle.dump(var, m)
	m.close()
	PrintTree(var)

	CheckModel(var, "./train/train_dataset.txt")
	CheckModel(var, "./test/test_dataset.txt")


'''
	noise decision tree.
'''
def noise_add(noise_percentage, attr_list, reviews_list):

	total_noise = (noise_percentage/100) * len(reviews_list)

	while total_noise > 0:

		r = rn.randint(0, len(reviews_list) - 1)
		if reviews_list[r][0] >= 7:
			reviews_list[r] = (-1,reviews_list[r][1])
		else:
			reviews_list[r] = (1,reviews_list[r][1])
		total_noise = total_noise - 1

	mat = return_mat(attr_list, reviews_list)

	level_tree = {}
	print("start making tree")
	m = open('./noise_model.pickle', 'wb')
	var = tc.decision_tree(list(reviews_list.keys()), None, None)
	prev = []
	var.MakeTree(mat, attr_list, prev)
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


'''
	pruned tree.
'''
def pruning(reviews_list):
	_, test_data = read_dataset_and_attributes(None, "./test/test_dataset.txt")
	model = open("model.pickle","rb")
	var = pickle.load(model)
	error = 1
	level = {}
	var.printTree(0, level)
	for l in sorted(level.keys()):
		print(l," : ", end=" ")
		for value in level[l]:
			print(value.attr, "(", value.label, ")", end="\t")

		print("\n")
	maxerror = checkError(var, test_data)
	print(maxerror)
	for i in reversed(sorted(level.keys())):
		for node in level[i]:
			if node:
				# if not node.error:
				left = node.left
				right = node.right
				label = node.label
				node.left = None
				node.right = None
				p, n = node.return_pos_neg_count()
				if p > n:
					node.label = 1
				else:
					node.label = -1
				e = checkError(var, test_data)
				print(e)
				if e > maxerror:
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

'''
	predict data given the model.
'''
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


def feature_bagging(attr_list, reviews_list, test_data, number_of_trees):

	
	forest = []
	for i in range(number_of_trees):
		print(i)
		rn.shuffle(attr_list)
		var = tc.decision_tree(list(reviews_list.keys()), None, None)
		mat = return_mat(attr_list, reviews_list)
		prev = []
		var.MakeTree(mat, attr_list, prev)
		forest.append(var)


	error = 0
	for data in test_data:
		actual_label = test_data[data][0]
		prediction = None
		pred = 0

		for tree in forest:
			p = predict(tree, test_data[data][1])
			if p == 1:
				pred = pred + 1

		if (len(forest) - pred) > pred:
			prediction = -1
		else:
			prediction = 1

		if actual_label != prediction:
			error = error + 1

	print("error: ", error/len(test_data))





attr_list, reviews_list = read_dataset_and_attributes('attr_words.txt', './train/train_dataset.txt')
rn.shuffle(attr_list)

if sys.argv[1] == '2':

	depth = input('Enter the depth for early stopping: ')
	ratio = input('Enter the percentage +/-ve ratio value for early stopping: ')
	DecisionTree(attr_list, reviews_list, mat)
	EarlyStoppingDT(mat, attr_list, reviews_list,  depth=int(depth), percentage_review=0, ratio=float(ratio))
elif sys.argv[1] == '3':
	noise_add(.5, attr_list, reviews_list)
	noise_add(1, attr_list, reviews_list)
	noise_add(5, attr_list, reviews_list)
	noise_add(10, attr_list, reviews_list)
elif sys.argv[1] == '4':
	pruning(reviews_list)
elif sys.argv[1] == '5':
	_, test_data = read_dataset_and_attributes(None, sys.argv[2])
	for data in test_data:
		if test_data[data][0] >= 7:
			test_data[data] = (1,test_data[data][1])
		else:
			test_data[data] = (-1,test_data[data][1])

	attr_list_copy = copy.deepcopy(attr_list)
	feature_bagging(attr_list[0:1000], reviews_list, test_data, 1)
	number_of_trees = 0
	for i in range(5):
		new_attr_list = copy.deepcopy(attr_list_copy)
		number_of_trees = number_of_trees + 5
		rn.shuffle(new_attr_list)
		print(new_attr_list)
		feature_bagging(new_attr_list[0:1000], reviews_list, test_data, number_of_trees)	
	feature_bagging(attr_list[0:1000], reviews_list, 5)
	feature_bagging(attr_list[0:1000], reviews_list, 10)
	feature_bagging(attr_list[0:1000], reviews_list, 15)
	feature_bagging(attr_list[0:1000], reviews_list, 20)
	feature_bagging(attr_list[0:1000], reviews_list, 25)

