import math
import timeit

class decision_tree(object):
	"""
		This is the Node class representing the nodes of tree
		left -> left child of node
		right -> right child of node
	"""
	def __init__(self, reviews_index_list, attr, parent):
		self.parent = parent
		self.left = None
		self.right = None
		self.label = None
		self.reviews_index_list = reviews_index_list
		self.attr = attr
		self.error = False


	def printTree(self, depth, level_tree):
		"""		
			To print the level order of the tree.
		"""
		if self:
			if self.left:
				self.left.printTree(depth+1, level_tree)

			if depth in level_tree.keys():
				level_tree[depth].append(self)
			else:
				level_tree[depth] = [self]

			if self.right:
				self.right.printTree(depth+1, level_tree)


	def return_pos_neg_count(self, reviews_list):
		""" Returns the number of POSITIVE and NEGATIVE Reviews count."""
		pos_count = 0
		neg_count = 0

		for review in self.reviews_index_list:
			if reviews_list[review][0] >= 7:
				pos_count = pos_count + 1
			elif reviews_list[review][0] <= 4:
				neg_count = neg_count + 1

		return pos_count, neg_count


	def return_entropy(self, reviews_list):
		""" 
			Returns the ENTROPY of the given NODE of the DECISION TREE using below formula:
			E(Node) = - pp * log2(pp) - np * log2(np)

			where pp and np -> probability of positive and negative reviews respectively.	
		"""
		pos_count, neg_count = self.return_pos_neg_count(reviews_list)
		# Total Count of reviews
		total = pos_count + neg_count

		if total == 0:
			return 0

		pos_p = pos_count/total
		neg_p = neg_count/total

		if pos_p > 0:
			pos_p = -(pos_p * (math.log(pos_p)/math.log(2)))

		if neg_p > 0:
			neg_p = -(neg_p * (math.log(neg_p)/math.log(2)))	

		return pos_p + neg_p


	def return_ig(self, attr, reviews_list):
		"""
			Returns Information gain for given attribute.
		"""
		left = []
		right = []

		for review in self.reviews_index_list:
			if str(attr)+":" in reviews_list[review][1]:
				left.append(review)
			else:
				right.append(review)
		# left = [review for review in self.reviews_index_list if str(attr)+":" in reviews_list[review][1]]
		# right = [review for review in self.reviews_index_list if review not in left]

		entropy = self.return_entropy(reviews_list)
		children_entropy = 0

		if len(left) + len(right) > 0:
			self.left = decision_tree(left, None, self)
			self.right = decision_tree(right, None, self)
			left_entropy = self.left.return_entropy(reviews_list)
			right_entropy = self.right.return_entropy(reviews_list)
			lp = len(left)/(len(left) + len(right))
			rp = len(right)/(len(left) + len(right))

			children_entropy = (left_entropy * lp + right_entropy * rp)


		return entropy - children_entropy, left, right


	def return_max_attr(self, review_list, attr_list):
		"""
			Returns attribute to be placed at current node i.e. attribute with maximum information gain.
		"""
		max_attr = ""
		max_ig = 0
		max_left = None
		max_right = None

		for attr in attr_list:
			ig, left, right = self.return_ig(attr, review_list)
			if ig > max_ig:
				max_ig = ig
				max_attr = attr
				max_left = left
				max_right = right

		print(max_ig)
		if max_attr in attr_list:
			self.attr = max_attr
			attr_list.remove(max_attr)

		if max_ig == 0:
			self.left = None
			self.right = None
		else:
			self.left = decision_tree(max_left, None, self)
			self.right = decision_tree(max_right, None, self)


	def MakeTree(self, reviews_list, attr_list, depth=None, percentage_review=None):

		p, n = self.return_pos_neg_count(reviews_list)

		depth_count = 1
		if depth:
			depth_count = depth

		per_rev = 0
		if percentage_review:
			per_rev = (percentage_review/100) * len(reviews_list)

		if attr_list and len(self.reviews_index_list) > per_rev and depth_count > 0:
			if p != 0 and n != 0:
				self.return_max_attr(reviews_list, attr_list)

				if not self.left or not self.right:
					if p > n:
						self.label = 1
					else:
						self.label = -1

				else:
					if depth :
						self.left.MakeTree(reviews_list, attr_list, depth=depth-1, percentage_review=percentage_review)
						self.right.MakeTree(reviews_list, attr_list, depth=depth-1, percentage_review=percentage_review)
					else:
						self.left.MakeTree(reviews_list, attr_list)
						self.right.MakeTree(reviews_list, attr_list)

			else:

				if p == 0:
					self.label = -1
				else:
					self.label = +1
		else:

			if p > n:
				self.label = 1
			else:
				self.label = -1


	def errorless_parent(self):

		self.error = True
		if self.parent:
			self.parent.errorless_parent()