import mysqldb
import nltk
import getpass

# A struct of sorts that holds information field values
# for a given post (e.g. body, author_id), depending on the query.
class Post:
	pass

# An interface to a forum database. Performs database queries
# and returns formatted results.
class Forum:
	_host = 'datastage.stanford.edu'
	_user = 'akshayka'
	_db = 'EdxForum'

	def __init__(self, passwd='', host=_host, user=_user, db=_db):
		if (passwd == ''):
			passwd = getpass.getpass("Password: ")
		self.db = mysqldb.MySQLDB(host, user=user, passwd=passwd, db=db)

	# Returns a list of Post structures, where each Post contains information
	# about a given post retrieved from the specificed table.
	#
	# More concretely, for each field in attr, there exists a corresponding
	# member variable in Post with the same name that contains that field's value.
	#
	# For example, if post_contents were called with
	# 	attr=['author_id', 'body'],
	# then each post p in posts would have
	#	p.author_id
	#	p.body
	# as valid members variables.
	#
	# args:
	# 	One of query or (table, course_name) are required.
	#	The latter args are meant to abstract away the sql command,
	#	but raw sql queries are accepted. 	
	#	NB: attr _must_ correspond to the selected attributes in
	#		the query.
	def post_contents(self, attr, table='', course_name='', query='', \
					verbose=False):
		posts = []

		# Construct the query if one was not supplied.
		if (query == ''):
			fields = ','.join(attr)
			query = 'select ' + fields + ' from ' + table + \
				' where course_display_name="' + course_name + '"'

		# Retrieve the posts from the query.
		for result in self.db.query(query):
			if (verbose):
				result_str = ''
				for entry in result: result_str += str(entry) + ' '
				print result_str

			post = Post()
			for i in range(0, len(attr)):
				setattr(post, attr[i], result[i])
			posts.append(post)
		return posts

	def course_names(self, table=''):
		course_names = []
		for result in self.db.query('select distinct course_display_name ' \
			'from contents;'):	
			course_names.append(result[0])
		return course_names

class Format:
	def __init__(self):
		self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
	def split_sentences(self, post):
		  return self.sent_detector.tokenize(post.strip())
